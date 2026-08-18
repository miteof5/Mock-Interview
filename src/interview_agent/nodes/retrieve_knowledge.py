"""retrieve_knowledge 节点：按当前主题检索知识库。"""

from __future__ import annotations

from interview_agent.knowledge.retriever import retrieve
from interview_agent.models import InterviewTopic
from interview_agent.state import InterviewState


def _build_query(topic: InterviewTopic) -> str:
    """用主题标题和考查重点拼接检索 query。"""
    return f"{topic.title} {topic.focus}".strip()


def retrieve_knowledge(state: InterviewState) -> dict:
    """检索当前主题相关 Chunk，结果写入 retrieval_results。"""
    outline = state.get("outline")
    if outline is None or not outline.topics:
        raise ValueError("outline 为空，无法定位当前主题")

    index = state.get("current_topic_index", 0)
    if index < 0 or index >= len(outline.topics):
        raise ValueError(f"current_topic_index 越界: {index}")

    topic = outline.topics[index]
    results = retrieve(_build_query(topic))
    return {"retrieval_results": results}
