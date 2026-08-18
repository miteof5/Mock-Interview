"""知识库入库：Markdown 切块 -> 向量化 -> 写入本地 Chroma。"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from chromadb import PersistentClient

from interview_agent.config import (
    KB_COLLECTION_NAME,
    Settings,
    get_settings,
)
from interview_agent.knowledge.chunker import Chunk, MarkdownChunker
from interview_agent.knowledge.embedder import DashScopeEmbedder

# (KB_COLLECTION_NAME 来自 config.py，写入侧与读取侧 retriever.py 共用同一常量定义；
#  此处重导出是为了对外 API 稳定：外部历史代码从 ingest 取常量时仍然可用。)
__all__ = [
    "KB_COLLECTION_NAME",
    "collect_markdown_files",
    "ingest_markdown_file",
    "ingest_markdown_files",
]

logger = logging.getLogger(__name__)


def collect_markdown_files(kb_path: Path) -> list[Path]:
    """递归扫描知识库目录，返回排序后的 .md 文件列表。"""
    if not kb_path.exists():
        raise FileNotFoundError(f"知识库目录不存在: {kb_path}")
    files = [
        path
        for path in kb_path.rglob("*.md")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(kb_path).parts)
    ]
    return sorted(files)


def _chunk_id(source: str, index: int) -> str:
    """基于来源路径和块序号生成稳定 ID，重复入库时可覆盖更新。"""
    raw = f"{source}:{index}".encode()
    return hashlib.sha1(raw).hexdigest()


def _serialize_metadata(meta: dict[str, str | int | bool]) -> dict[str, str | int]:
    """Chroma metadata schema 强约束且不可变：bool 转 int 防御性写入。

    chunker 产出的 metadata 含 is_code_block / is_mermaid / is_table 三个 bool 字段，
    chromadb<0.5 会拒绝 bool 类型；即便 >=0.5 也要求同一 collection 内字段类型恒定。
    统一转成 int（0/1）写入，读取侧无需特殊处理。
    """
    return {k: int(v) if isinstance(v, bool) else v for k, v in meta.items()}


def ingest_markdown_files(
    files: list[Path],
    *,
    settings: Settings | None = None,
    chunker: MarkdownChunker | None = None,
    embedder: DashScopeEmbedder | None = None,
    reset: bool = False,
) -> int:
    """将 Markdown 文件切块、向量化并写入本地 Chroma，返回入库 Chunk 数。"""
    settings = settings or get_settings()
    chunker = chunker or MarkdownChunker()
    own_embedder = embedder is None
    if own_embedder:
        embedder = DashScopeEmbedder(settings=settings)

    try:
        # ---- 第一步：切块 ----
        chunks: list[Chunk] = []
        for path in files:
            file_chunks = chunker.chunk_file(path)
            logger.info("切块完成: %s -> %d 个 Chunk", path, len(file_chunks))
            chunks.extend(file_chunks)
        if not chunks:
            logger.warning("没有可入库的 Chunk，直接结束")
            return 0

        # ---- 第二步：批量向量化 ----
        texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed_documents(texts)
        logger.info("向量化完成: %d 条", len(vectors))

        # ---- 第三步：写入 Chroma ----
        # TODO(P1-4): PersistentClient 每次调用都重新创建，后续 retriever 模块也会
        # 创建自己的 client，两个 client 同时持有同一 SQLite 目录可能锁冲突。
        # 后续应将 client 做成单例或在 Settings 中复用。
        client = PersistentClient(path=str(settings.kb_vector_path))
        if reset:
            existing_names = {col.name for col in client.list_collections()}
            if KB_COLLECTION_NAME in existing_names:
                client.delete_collection(KB_COLLECTION_NAME)
                logger.info("已删除旧 collection: %s", KB_COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=KB_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # P0-1 修复：增量入库前，按 source 清理该来源下所有旧 Chunk。
        # 场景：文件修改后切块数/内容变化，旧 ID 残留会导致 RAG 召回已删除的脏数据。
        # reset=True 时 collection 是全新的，get 返回空，此循环无副作用。
        sources = sorted({str(chunk.metadata["source"]) for chunk in chunks})
        for src in sources:
            existing = collection.get(where={"source": src})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.info("清理旧 Chunk: source=%s, 删除 %d 条", src, len(existing["ids"]))

        ids = [
            _chunk_id(
                str(chunk.metadata["source"]),
                int(chunk.metadata["chunk_index"]),
            )
            for chunk in chunks
        ]
        # P0-2 修复：bool metadata 序列化为 int，防御性写入 Chroma
        metadatas = [_serialize_metadata(chunk.metadata) for chunk in chunks]
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )
        logger.info("写入完成: collection 现有 %d 个 Chunk", collection.count())
        return len(chunks)
    finally:
        if own_embedder:
            embedder.close()


def ingest_markdown_file(
    path: str | Path,
    *,
    settings: Settings | None = None,
    chunker: MarkdownChunker | None = None,
    embedder: DashScopeEmbedder | None = None,
) -> int:
    """入库单个 Markdown 文件（增量语义），返回该文件生成的 Chunk 数。

    为什么不暴露 reset 参数？
        单文件模式的天然语义是「更新这一个 source」，P0-1 已按 source 清理旧 Chunk。
        如果允许 reset=True，会把整个 KB_COLLECTION_NAME collection 全删掉再只写这一
        个文件，其它文件的数据全部丢失——属于灾难级误操作，因此强制 reset=False。
        需要重建全库时，请走 ingest_markdown_files(..., reset=True) 或 CLI --kb-path --reset。
    """
    return ingest_markdown_files(
        [Path(path)],
        settings=settings,
        chunker=chunker,
        embedder=embedder,
        reset=False,  # P0-3 修复：单文件模式强制关闭 reset，避免误清空全库
    )
