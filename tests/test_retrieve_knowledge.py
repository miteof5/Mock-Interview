"""retrieve_knowledge 节点单测：mock retriever.retrieve，不触达向量库。

节点是纯逻辑：取 outline.topics[index] → 拼 query → 调 retrieve → 返回 results。
覆盖正常路径、query 构造契约、index 越界、outline 缺失。
"""

from __future__ import annotations

import pytest

from interview_agent.knowledge.retriever import retrieve as _real_retrieve  # noqa: F401
from interview_agent.models import InterviewOutline, InterviewTopic, KnowledgeChunk
from interview_agent.nodes.retrieve_knowledge import _build_query, retrieve_knowledge


def _outline(n: int = 3) -> InterviewOutline:
    """构造 n 个主题的提纲，id 形如 t0/t1/...，title/focus 用于拼 query。"""
    return InterviewOutline(
        topics=[
            InterviewTopic(
                id=f"t{i}",
                title=f"主题{i}",
                focus=f"考点{i}",
            )
            for i in range(n)
        ]
    )


@pytest.fixture
def fake_retriever(monkeypatch):
    """注入 stub retrieve，记录调用参数并返回固定 KnowledgeChunk 列表。

    节点内部是 `from ...retriever import retrieve` 后直接调用，
    所以要 patch 的是节点命名空间里的 retrieve 引用。
    """
    calls = {"query": []}

    def _stub_retrieve(query: str, *args, **kwargs):
        calls["query"].append(query)
        return [
            KnowledgeChunk(
                content=f"知识片段-{query}",
                source="doc.md",
                score=0.9,
            ),
            KnowledgeChunk(
                content=f"知识片段2-{query}",
                source="",
                score=0.5,
            ),
        ]

    monkeypatch.setattr(
        "interview_agent.nodes.retrieve_knowledge.retrieve",
        _stub_retrieve,
    )
    return calls


# ====================== _build_query 单元测试 ======================

def test_build_query_combines_title_and_focus():
    """_build_query 把 title 和 focus 用空格拼接。"""
    topic = InterviewTopic(id="t0", title="分布式系统", focus="一致性协议")
    assert _build_query(topic) == "分布式系统 一致性协议"


def test_build_query_strips_whitespace():
    """_build_query 末尾 .strip()，空 focus 不留多余空格。"""
    topic = InterviewTopic(id="t0", title="主题", focus="")
    # topic.focus 为空串时，f"{title} {focus}" 末尾会有空格，strip 掉
    assert _build_query(topic) == "主题"


# ====================== 正常路径 + 调用契约 ======================

def test_retrieve_knowledge_happy_path(fake_retriever):
    """正常路径：返回检索结果，写入 retrieval_results。"""
    state = {"outline": _outline(3), "current_topic_index": 0}
    result = retrieve_knowledge(state)
    assert "retrieval_results" in result
    chunks = result["retrieval_results"]
    assert len(chunks) == 2
    assert isinstance(chunks[0], KnowledgeChunk)
    assert chunks[0].score == 0.9


def test_retrieve_knowledge_passes_topic_query_to_retrieve(fake_retriever):
    """调用契约：retrieve 收到的 query = '{title} {focus}'。"""
    state = {"outline": _outline(3), "current_topic_index": 0}
    retrieve_knowledge(state)
    # topics[0].title="主题0", focus="考点0" → "主题0 考点0"
    assert fake_retriever["query"] == ["主题0 考点0"]


def test_retrieve_knowledge_uses_current_topic_index(fake_retriever):
    """index=1 时应取第二个主题拼 query。"""
    state = {"outline": _outline(3), "current_topic_index": 1}
    retrieve_knowledge(state)
    # topics[1].title="主题1", focus="考点1"
    assert fake_retriever["query"] == ["主题1 考点1"]


def test_retrieve_knowledge_invokes_retrieve_once(fake_retriever):
    """每次只调用 retrieve 一次。"""
    state = {"outline": _outline(3), "current_topic_index": 0}
    retrieve_knowledge(state)
    assert len(fake_retriever["query"]) == 1


def test_retrieve_knowledge_default_index_zero(fake_retriever):
    """state 缺少 current_topic_index 时默认取 0。"""
    state = {"outline": _outline(3)}
    retrieve_knowledge(state)
    assert fake_retriever["query"] == ["主题0 考点0"]


# ====================== 边界：outline 缺失 / 空 ======================

def test_retrieve_knowledge_rejects_none_outline(fake_retriever):
    """outline=None：应抛 ValueError，不应触达 retrieve。"""
    state = {"outline": None, "current_topic_index": 0}
    with pytest.raises(ValueError, match="outline 为空"):
        retrieve_knowledge(state)
    assert fake_retriever["query"] == []


def test_retrieve_knowledge_rejects_empty_topics(fake_retriever):
    """outline.topics=[]：应抛 ValueError。"""
    state = {
        "outline": InterviewOutline(topics=[]),
        "current_topic_index": 0,
    }
    with pytest.raises(ValueError, match="outline 为空"):
        retrieve_knowledge(state)
    assert fake_retriever["query"] == []


# ====================== 边界：index 越界 ======================

def test_retrieve_knowledge_rejects_negative_index(fake_retriever):
    """index=-1：应抛 ValueError。"""
    state = {"outline": _outline(3), "current_topic_index": -1}
    with pytest.raises(ValueError, match="current_topic_index 越界"):
        retrieve_knowledge(state)
    assert fake_retriever["query"] == []


def test_retrieve_knowledge_rejects_index_overflow(fake_retriever):
    """index >= len(topics)：应抛 ValueError。"""
    state = {"outline": _outline(3), "current_topic_index": 3}
    with pytest.raises(ValueError, match="current_topic_index 越界"):
        retrieve_knowledge(state)
    assert fake_retriever["query"] == []


def test_retrieve_knowledge_rejects_index_at_boundary(fake_retriever):
    """index == len(topics)（恰好越界）：应抛 ValueError。"""
    state = {"outline": _outline(2), "current_topic_index": 2}
    with pytest.raises(ValueError, match="current_topic_index 越界"):
        retrieve_knowledge(state)
