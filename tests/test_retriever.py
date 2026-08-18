"""retriever.py baseline 测试：边界、空 collection、正常召回排序。

策略：
    - Chroma 用真 PersistentClient 指向 tmp_path（不用 mock collection，行为最真实）；
    - Embedder 用 FakeEmbedder（返回固定维度的伪随机向量，可预测 query 与 doc 的距离），
      避免触达真实 LLM/Embedding API；
    - Settings 注入自定义 kb_vector_path（即 tmp_path），绕过全局 data/vectorstore。
"""

from __future__ import annotations

from typing import Any

import pytest
from chromadb import PersistentClient

from interview_agent.config import KB_COLLECTION_NAME, Settings
from interview_agent.knowledge.embedder import DashScopeEmbedder
from interview_agent.knowledge.retriever import (
    _deserialize_metadata,
    _similarity_from_distance,
    retrieve,
)

# ---------------------------------------------------------------------------
# 假 Embedder：不触达网络，返回可预测的伪向量
# ---------------------------------------------------------------------------


class FakeEmbedder(DashScopeEmbedder):
    """返回定长伪向量的 embedder，仅用于本地测试。

    向量化策略（关键：保证不同 query/doc 之间的"相对距离"是确定的）：
        - 对每个文本取基于内容的种子，生成固定维度（16 维）的向量；
        - 相同文本永远返回相同向量；
        - 不同文本大概率不同，距离可被余弦空间正确排序。
    """

    DIM = 16

    def __init__(self) -> None:
        # 跳过父类 __init__（它会校验 API Key）—— 本类完全不触达网络
        self.settings = None  # type: ignore[assignment]
        self.batch_size = 1
        self._client = None  # type: ignore[assignment]

    def embed_documents(self, texts: list[str], **_: Any) -> list[list[float]]:
        return [self._pseudo_vec(t) for t in texts]

    def embed_query(self, text: str, **_: Any) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty query")
        return self._pseudo_vec(text)

    def close(self) -> None:
        pass

    @classmethod
    def _pseudo_vec(cls, text: str) -> list[float]:
        """生成一个 16 维伪向量：用文本内容的 hash 做种子 + LCG。"""
        h = 0
        for ch in text:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        vec = []
        x = h or 1
        for _ in range(cls.DIM):
            x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
            # 归一化到 [-1, 1]
            vec.append((x / 0xFFFFFFFF) * 2.0 - 1.0)
        return vec


# ---------------------------------------------------------------------------
# 通用 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_settings(tmp_path) -> Settings:
    """返回 kb_vector_path 指向 pytest 临时目录的 Settings，不会污染真实库。"""
    # Settings 的 after_validator 会把相对路径拼到 PROJECT_ROOT，所以用绝对路径 tmp_path
    vector_path = tmp_path / "vectorstore"
    vector_path.mkdir(parents=True, exist_ok=True)
    return Settings(
        kb_vector_path=vector_path.resolve(),
        retrieval_top_k=4,
        # 其它字段走默认 + 环境变量里的假 DASHSCOPE_API_KEY（conftest 已注入）
    )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def _seed_kb(
    settings: Settings,
    docs: list[tuple[str, dict]],
    embedder: FakeEmbedder,
) -> None:
    """往临时 Chroma 里直接灌 data（不走 ingest.py，保证单测独立）。"""
    client = PersistentClient(path=str(settings.kb_vector_path))
    col = client.get_or_create_collection(
        name=KB_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    texts = [d for d, _ in docs]
    metas = [m for _, m in docs]
    vectors = embedder.embed_documents(texts)
    col.upsert(
        ids=[f"doc-{i}" for i in range(len(docs))],
        documents=texts,
        metadatas=metas,
        embeddings=vectors,
    )


# ---------------------------------------------------------------------------
# 辅助函数单测：_similarity_from_distance / _deserialize_metadata
# ---------------------------------------------------------------------------


def test_similarity_from_distance_clamp():
    """distance∈[0,2] → similarity∈[1,0]；超出范围夹取。"""
    assert _similarity_from_distance(0.0) == pytest.approx(1.0)
    assert _similarity_from_distance(1.0) == pytest.approx(0.0)
    assert _similarity_from_distance(0.5) == pytest.approx(0.5)
    # 越界夹取
    assert _similarity_from_distance(-0.1) == pytest.approx(1.0)
    assert _similarity_from_distance(2.0) == pytest.approx(0.0)


def test_deserialize_metadata_int_bool_and_defaults():
    """写入侧是 int 0/1；读回时统一成 bool；缺字段填默认值。"""
    raw = {
        "source": "/path/to/a.md",
        "doc_title": "标题",
        "heading_path": "## 一 / ### 1.1",
        "chunk_index": 3,
        "heading_level": 2,
        "is_code_block": 1,
        "is_mermaid": 0,
        # is_table 缺失
    }
    m = _deserialize_metadata(raw)
    assert m["source"] == "/path/to/a.md"
    assert m["doc_title"] == "标题"
    assert m["heading_path"] == "## 一 / ### 1.1"
    assert m["chunk_index"] == 3
    assert m["heading_level"] == 2
    assert m["is_code_block"] is True
    assert m["is_mermaid"] is False
    # 缺字段默认：
    assert m["is_table"] is False


def test_deserialize_metadata_none_is_safe():
    m = _deserialize_metadata(None)
    assert m["source"] == ""
    assert m["doc_title"] == ""
    assert m["chunk_index"] == 0
    assert m["is_code_block"] is False


# ---------------------------------------------------------------------------
# retrieve() 主 API 的三组 baseline 测试
# ---------------------------------------------------------------------------


class TestRetrieveBoundaries:
    """P0 边界：非法参数必抛 ValueError，避免脏数据继续往下流。"""

    def test_rejects_empty_query(self, tmp_settings, fake_embedder):
        with pytest.raises(ValueError, match="query must not be empty"):
            retrieve("   ", settings=tmp_settings, embedder=fake_embedder)

    def test_rejects_non_string_query(self, tmp_settings, fake_embedder):
        with pytest.raises(ValueError, match="query must not be empty"):
            retrieve(None, settings=tmp_settings, embedder=fake_embedder)  # type: ignore[arg-type]

    def test_rejects_top_k_zero(self, tmp_settings, fake_embedder):
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            retrieve("q", top_k=0, settings=tmp_settings, embedder=fake_embedder)

    def test_rejects_top_k_negative(self, tmp_settings, fake_embedder):
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            retrieve("q", top_k=-1, settings=tmp_settings, embedder=fake_embedder)

    def test_rejects_min_score_out_of_range(self, tmp_settings, fake_embedder):
        with pytest.raises(ValueError, match="min_score must be within"):
            retrieve("q", min_score=1.1, settings=tmp_settings, embedder=fake_embedder)
        with pytest.raises(ValueError, match="min_score must be within"):
            retrieve("q", min_score=-0.1, settings=tmp_settings, embedder=fake_embedder)


class TestRetrieveCollectionMissing:
    """P0 空 collection：返回 [] + warning，不抛异常。"""

    def test_missing_collection_returns_empty(self, tmp_settings, fake_embedder, caplog):
        import logging

        # 空临时目录，从未建 collection
        with caplog.at_level(logging.WARNING, logger="interview_agent.knowledge.retriever"):
            res = retrieve("任何问题", settings=tmp_settings, embedder=fake_embedder)
        assert res == []
        assert any("collection 不存在" in r.getMessage() for r in caplog.records)
        assert KB_COLLECTION_NAME in caplog.text

    def test_default_top_k_follows_settings_retrieval_top_k(self, tmp_settings, fake_embedder):
        """不传 top_k 时，effective_top_k 使用 settings.retrieval_top_k。

        间接验证：如果 settings.retrieval_top_k 被读了，那么 collection 缺失时
        日志不会报 top_k 相关错误（即流程顺利走到了 collection 检查那一步）。
        """
        s = Settings(kb_vector_path=tmp_settings.kb_vector_path, retrieval_top_k=7)
        assert retrieve("空", settings=s, embedder=fake_embedder) == []
        # 没抛 ValueError(top_k) 即代表默认值生效路径走对了


class TestRetrieveNormal:
    """P0 正常召回：灌 5 条文档，验证 top_k、分数排序、min_score 过滤、metadata 透传。"""

    DOCS: list[tuple[str, dict]] = [
        # 0
        ("RAG 检索增强生成：从向量库召回相关段落与 prompt 拼接后喂给 LLM。",
         {"source": "rag.md", "chunk_index": 0, "is_code_block": 0}),
        # 1 —— 含 Python 代码说明（写入侧按 int 存 bool）
        ("langchain_chroma.from_documents：把 Documents 批量灌进 Chroma collection。",
         {"source": "langchain.md", "chunk_index": 1, "is_code_block": 1}),
        # 2
        (
            "余弦相似度：向量内积除以模长乘积，值越大越相似；"
            "Chroma 默认返回 cosine distance（=1-相似度）。",
            {"source": "vector.md", "chunk_index": 2, "is_code_block": 0},
        ),
        # 3 —— 故意不相关（完全没提向量/RAG）
        ("简历解析：从 PDF 或 Word 提取姓名、学历、工作经历，生成结构化 ParsedResume。",
         {"source": "resume.md", "chunk_index": 3, "is_code_block": 0}),
        # 4 —— 直接出现 query 关键词，预期最相关
        (
            "Chroma 向量库检索的 query 向量化流程：调用 embed_query "
            "→ collection.query(n_results=top_k)。",
            {"source": "vector.md", "chunk_index": 4, "is_code_block": 0},
        ),
    ]

    @pytest.fixture
    def seeded(self, tmp_settings, fake_embedder):
        _seed_kb(tmp_settings, self.DOCS, fake_embedder)
        return tmp_settings, fake_embedder

    def test_returns_top_k_ordered_by_score(self, seeded):
        """query='Chroma 向量库检索'，期望 doc 4 最相关，doc 0/2/1 其次，doc 3 最不相关。"""
        settings, embd = seeded
        # top_k=3 只取前 3
        chunks = retrieve("Chroma 向量库检索", top_k=3, settings=settings, embedder=embd)
        assert len(chunks) == 3
        # 严格降序（允许平局）
        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_collection_size_is_safe(self, seeded):
        """KB 里只有 5 条，要 top_k=100 应全部返回，不炸。"""
        settings, embd = seeded
        chunks = retrieve("任何 query", top_k=100, settings=settings, embedder=embd)
        assert len(chunks) == len(self.DOCS)

    def test_min_score_filters_low_relevance(self, seeded):
        """min_score=1.0 时几乎不可能命中，返回空；min_score=0 时全部通过。"""
        settings, embd = seeded
        assert retrieve("向量", min_score=1.0, settings=settings, embedder=embd) == []
        # top_k=5 + min_score=0 应拿满 5 条
        chunks = retrieve("向量", top_k=5, min_score=0.0, settings=settings, embedder=embd)
        assert len(chunks) == 5

    def test_source_and_score_fields_populated(self, seeded):
        """每个返回的 KnowledgeChunk 都带正确的 source 类型与合法 score 范围。"""
        settings, embd = seeded
        chunks = retrieve("RAG 检索 向量", top_k=4, settings=settings, embedder=embd)
        for c in chunks:
            assert isinstance(c.content, str) and c.content
            assert isinstance(c.source, str)
            assert 0.0 <= c.score <= 1.0

    def test_default_top_k_respects_settings(self, seeded, monkeypatch):
        """显式不传 top_k 时，settings.retrieval_top_k=2 → 只返回 2 条。"""
        settings, embd = seeded
        small_settings = Settings(
            kb_vector_path=settings.kb_vector_path,
            retrieval_top_k=2,
        )
        chunks = retrieve("任何 query", settings=small_settings, embedder=embd)
        assert len(chunks) == 2
