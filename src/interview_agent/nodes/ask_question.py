"""ask_question 节点：按 last_decision.action 分支出首问或追问。"""

from __future__ import annotations

from interview_agent.llm.client import DashScopeClient
from interview_agent.models import InterviewMessage, InterviewQuestion, InterviewTopic
from interview_agent.state import InterviewState


def _topic_text(topic: InterviewTopic) -> str:
    """把主题信息拼成传给 LLM 的描述（新版含难度+候选人依据，供出题端动态微调）。"""
    lines = [
        f"主题标题：{topic.title}",
        f"考查重点：{topic.focus}",
        f"难度档位：{topic.difficulty}",
        f"候选人经历依据：{topic.candidate_basis}",
        f"是否必问（JD核心要求）：{'是' if topic.must_ask else '否'}",
    ]
    return "\n".join(lines)


def _format_history(history: list[InterviewMessage]) -> str:
    """把对话历史拼成给 LLM 看的纯文本，空历史给显式占位。"""
    if not history:
        return "(暂无历史对话)"
    lines = []
    for msg in history:
        speaker = "面试官" if msg.role == "interviewer" else "候选人"
        lines.append(f"{speaker}: {msg.content}")
    return "\n".join(lines)


def ask_question(state: InterviewState) -> dict:
    """根据上一轮决策出题，并更新对话历史与计数。"""
    outline = state.get("outline")
    if outline is None or not outline.topics:
        raise ValueError("outline 为空，无法出题")
    index = state.get("current_topic_index", 0)
    if index < 0 or index >= len(outline.topics):
        raise ValueError(f"current_topic_index 越界: {index}")
    topic = outline.topics[index]

    decision = state.get("last_decision")
    action = decision.action if decision is not None else None
    client = DashScopeClient()

    if action == "follow_up":
        question = _ask_follow_up(client, state, topic)
        follow_up_count = state.get("follow_up_count", 0) + 1
    elif action is None or action == "next_topic":
        question = _ask_first(client, state, topic)
        follow_up_count = 0
    else:
        raise ValueError(f"未知 last_decision.action: {action!r}")

    return {
        "current_question": question,
        # 始终把 current_topic_id 与 current_topic_index 对齐：
        # 首问分支（action=None/next_topic）此前依赖 decide_next 才首次写入，
        # 这里统一回填避免「id 仍为 None 而 index 已推进」的语义不一致。
        "current_topic_id": topic.id,
        "history": [InterviewMessage(role="interviewer", content=question.content)],
        "question_count": state.get("question_count", 0) + 1,
        "follow_up_count": follow_up_count,
    }


def _ask_first(
    client: DashScopeClient,
    state: InterviewState,
    topic: InterviewTopic,
) -> InterviewQuestion:
    """新主题首问：结合结构化画像、检索知识和历史对话出题。

    传结构化 ParsedJD/ParsedResume（都允许 None，解析失败时走兜底），不再传原始文本；
    全局 difficulty 由 state 读取并下发给客户端，作为出题难度基准与硬上限。
    """
    return client.ask_question(
        state.get("jd"),       # jd_profile: ParsedJD | None
        state.get("resume"),   # resume_profile: ParsedResume | None
        topic=_topic_text(topic),
        knowledge=state.get("retrieval_results"),
        history=_format_history(state.get("history") or []),
        difficulty=state.get("difficulty", "中等"),
    )


def _ask_follow_up(
    client: DashScopeClient,
    state: InterviewState,
    topic: InterviewTopic,
) -> InterviewQuestion:
    """追问：必须基于上一轮问题和候选人回答。"""
    previous = state.get("current_question")
    if previous is None:
        raise ValueError("follow_up 分支缺少上一轮问题")
    answer = state.get("last_answer") or ""
    if not answer.strip():
        raise ValueError("follow_up 分支缺少候选人回答")
    return client.follow_up(
        topic=_topic_text(topic),
        question=previous.content,
        answer=answer,
        difficulty=state.get("difficulty", "中等"),
    )
