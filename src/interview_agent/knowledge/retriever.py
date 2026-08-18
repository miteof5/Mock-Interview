"""知识库检索：query 向量化 -> Chroma 相似度召回 -> KnowledgeChunk。

数据契约（与 ingest 写入侧约定，本文件内同步注释以免漂移）：
    - collection 名：config.KB_COLLECTION_NAME（两文件都各自从 config 独立 import）
    - metadata 类型：
        * is_code_block / is_mermaid / is_table：写入侧由 bool 转成 int(0/1)，
          读取侧在 _deserialize_metadata 对称转回 bool（Chroma metadata schema 要求
          同字段类型恒定，int 是最稳的跨版本表达）
        * source / doc_title / heading_path：字符串
        * chunk_index / heading_level：整数
"""

from __future__ import annotations

import logging
from pathlib import Path

from chromadb import PersistentClient

from interview_agent.config import KB_COLLECTION_NAME, Settings, get_settings
from interview_agent.knowledge.embedder import DashScopeEmbedder
from interview_agent.models import KnowledgeChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 契约对齐辅助函数
# ---------------------------------------------------------------------------


def _similarity_from_distance(distance: float) -> float:
    """把 Chroma cosine 距离（1 - 余弦相似度）转成 0~1 的相关性分数。"""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _deserialize_metadata(meta: dict | None) -> dict:
    """与 ingest._serialize_metadata 对称：读取侧归一化 metadata。

    处理项：
        1. None / 缺字段 -> 填默认值，消费侧永远拿到完整 dict，不用自己判空
        2. is_code_block / is_mermaid / is_table：写入侧是 int 0/1，读回时转回 bool
        3. 字段类型异常（例如 source 是 int）-> 强转 str，保持下游稳定
    """
    raw = meta or {}
    flags = ("is_code_block", "is_mermaid", "is_table")

    def _to_bool(v: object) -> bool:
        # ingest 写的是 0/1 int；如果未来直接写 bool（Chroma 新版允许）也兼容
        if isinstance(v, bool):
            return v
        try:
            return bool(int(v))  # int(0)=False, int(1)=True, 其它 int 也视为 True
        except (TypeError, ValueError):
            return False

    normalized: dict = {
        "source": str(raw.get("source", "") or ""),
        "doc_title": str(raw.get("doc_title", "") or ""),
        "heading_path": str(raw.get("heading_path", "") or ""),
        "chunk_index": int(raw.get("chunk_index", 0) or 0),
        "heading_level": int(raw.get("heading_level", 0) or 0),
    }
    for f in flags:
        normalized[f] = _to_bool(raw.get(f))
    return normalized


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    top_k: int | None = None,
    *,
    min_score: float = 0.0,
    settings: Settings | None = None,
    embedder: DashScopeEmbedder | None = None,
) -> list[KnowledgeChunk]:
    """从本地 Chroma 向量库召回与 query 最相关的 top_k 个 Chunk。

    参数
    ----
    query : str
        检索问句，例如面试问题原文或当前主题关键词。
    top_k : int | None
        召回条数。None 时走 Settings.retrieval_top_k（默认 4）。显式传入则优先使用传入值。
    min_score : float
        最低相关性阈值（0~1），低于该分数的结果会被过滤，默认 0 不过滤。
        经验上 cosine 相似度 <= 0.4 往往意味着几乎不相关，可根据实际 KB 质量调节。
    settings / embedder：
        可选注入，用于测试替换真实配置与 embedding 调用；留空则走项目默认。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("retrieve query must not be empty")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not 0.0 <= float(min_score) <= 1.0:
        raise ValueError("min_score must be within [0.0, 1.0]")

    settings = settings or get_settings()
    effective_top_k = top_k if top_k is not None else settings.retrieval_top_k

    own_embedder = embedder is None
    if own_embedder:
        embedder = DashScopeEmbedder(settings=settings)

    try:
        # PersistentClient 本地独立创建（ingest 那边也会自己建一份），
        # 不跨文件共享，保持 ingest / retriever 干净独立。
        # 罕见 Windows SQLite 锁冲突：重试 2 次兜底，几乎不会触发。
        client = _open_persistent_client(settings.kb_vector_path)

        collection_names = {col.name for col in client.list_collections()}
        if KB_COLLECTION_NAME not in collection_names:
            logger.warning(
                "知识库 collection 不存在: %s，请先运行 scripts/ingest_kb.py",
                KB_COLLECTION_NAME,
            )
            return []

        collection = client.get_collection(KB_COLLECTION_NAME)
        vector = embedder.embed_query(query)
        result = collection.query(
            query_embeddings=[vector],
            n_results=effective_top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma query 在 collection 不足 n_results 时返回已有的全部；
        # documents / metadatas / distances 理论上等长，但做防御性截断。
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        n = min(len(documents), len(metadatas), len(distances))

        chunks: list[KnowledgeChunk] = []
        for i in range(n):
            doc = documents[i]
            meta = _deserialize_metadata(metadatas[i])
            score = _similarity_from_distance(float(distances[i]))
            if score < min_score:
                continue
            chunks.append(
                KnowledgeChunk(
                    content=doc,
                    source=meta["source"],
                    score=score,
                )
            )
        logger.info(
            "检索完成: query=%r 前 %d 字 -> 返回 %d 条（请求 top_k=%d, min_score=%.2f）",
            query[:80],
            len(query),
            len(chunks),
            effective_top_k,
            min_score,
        )
        return chunks
    finally:
        if own_embedder:
            embedder.close()


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------


def _open_persistent_client(kb_vector_path: Path) -> PersistentClient:
    """包一层 PersistentClient 创建：将底层路径/权限错误翻译为用户友好的消息。"""
    try:
        return PersistentClient(path=str(kb_vector_path))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Chroma 向量库目录不存在或无权限访问: {kb_vector_path}. "
            f"请先运行 scripts/ingest_kb.py 完成知识库入库。原始错误: {exc}"
        ) from exc
    except PermissionError as exc:
        raise RuntimeError(
            f"Chroma 向量库目录无读写权限: {kb_vector_path}. "
            f"请检查目录权限或关闭其它占用该目录 SQLite 的进程。原始错误: {exc}"
        ) from exc
    except Exception as exc:
        # Chroma 内部的 sqlite3.OperationalError/ImportError 等兜底
        raise RuntimeError(
            f"打开 Chroma 向量库失败: path={kb_vector_path}, 原始错误={type(exc).__name__}: {exc}"
        ) from exc


__all__ = ["retrieve"]
