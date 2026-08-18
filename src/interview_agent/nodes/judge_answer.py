"""judge_answer 节点：调用 LLM 对当前回答评分。"""

from __future__ import annotations

from interview_agent.evaluation.rubric import normalize_score_items
from interview_agent.llm.client import DashScopeClient
from interview_agent.models import InterviewTopic
from interview_agent.state import InterviewState


def _topic_text(topic: InterviewTopic) -> str:
    """与 ask_question 节点保持一致的主题描述格式。"""
    return f"{topic.title}：{topic.focus}"


def judge_answer(state: InterviewState) -> dict:
    """对当前问题与回答调用 llm.judge，只返回本轮增量结果。"""
    outline = state.get("outline")
    if outline is None or not outline.topics:
        raise ValueError("outline 为空，无法定位当前主题")
    index = state.get("current_topic_index", 0)
    if index < 0 or index >= len(outline.topics):
        raise ValueError(f"current_topic_index 越界: {index}")
    topic = outline.topics[index]

    question = state.get("current_question")
    if question is None:
        raise ValueError("缺少当前问题")
    answer = state.get("last_answer") or ""
    if not answer.strip():
        raise ValueError("缺少候选人回答")

    judgment = DashScopeClient().judge(
        topic=_topic_text(topic),
        question=question.content,
        answer=answer,
        retrieval_results=state.get("retrieval_results"),
    )
    # llm.judge 内部已对齐一次，这里按节点层契约再兜底一次
    judgment.scores = normalize_score_items(
        judgment.scores,
        default_score=judgment.overall_score,
    )
    return {
        "last_judgment": judgment,
        "judgments": [judgment],
    }
