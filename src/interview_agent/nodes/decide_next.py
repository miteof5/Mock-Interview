"""decide_next 节点：按优先级决定追问、切题或结束。"""

from __future__ import annotations

from interview_agent.models import DecisionResult, NextAction
from interview_agent.state import InterviewState


def _action_from_suggestion(suggestion: str) -> NextAction:
    """从 last_judgment.follow_up_suggestion 中解析三选一动作。"""
    text = (suggestion or "").strip()
    if not text:
        return "follow_up"

    upper = text.upper()
    if upper.startswith("FOLLOW_UP"):
        return "follow_up"
    if upper.startswith("NEXT_TOPIC"):
        return "next_topic"
    if upper.startswith("END"):
        return "end"

    # 中文启发式兜底：模型没按结构化标记输出时仍能识别常见意图
    if any(keyword in text for keyword in ("结束", "停止", "终止")):
        return "end"
    if any(keyword in text for keyword in ("下一", "切题", "换题", "切换")):
        return "next_topic"
    return "follow_up"


def _decide(action: NextAction, reason: str, state: InterviewState) -> dict:
    """生成决策结果，并同步更新主题游标。"""
    update: dict = {
        "last_decision": DecisionResult(action=action, reason=reason),
    }
    if action == "end":
        update["stage"] = "finished"
        return update

    outline = state.get("outline")
    if outline is None or not outline.topics:
        raise ValueError("outline 为空，无法定位下一个主题")

    if action == "next_topic":
        index = state.get("current_topic_index", 0) + 1
        if index >= len(outline.topics):
            # 已到最后一个主题，没有下一个可切，强制结束
            update["stage"] = "finished"
            update["last_decision"] = DecisionResult(
                action="end",
                reason="没有更多主题，结束面试",
            )
            return update
        update["current_topic_index"] = index
        update["current_topic_id"] = outline.topics[index].id
        return update

    # follow_up：停留在当前主题，并确保 current_topic_id 与索引一致
    index = state.get("current_topic_index", 0)
    if index < 0 or index >= len(outline.topics):
        raise ValueError(f"current_topic_index 越界: {index}")
    update["current_topic_id"] = outline.topics[index].id
    return update


def decide_next(state: InterviewState) -> dict:
    """按固定优先级做边界决策：

    1. question_count 达到 max_questions -> end；
    2. follow_up_count 达到 max_follow_ups -> next_topic；
    3. 其余按 last_judgment.follow_up_suggestion 三选一。
    """
    question_count = state.get("question_count", 0)
    max_questions = state.get("max_questions", 0)
    if question_count >= max_questions:
        return _decide(
            "end",
            f"question_count={question_count} 达到上限 max_questions={max_questions}",
            state,
        )

    follow_up_count = state.get("follow_up_count", 0)
    max_follow_ups = state.get("max_follow_ups", 0)
    if follow_up_count >= max_follow_ups:
        return _decide(
            "next_topic",
            f"follow_up_count={follow_up_count} 达到上限 max_follow_ups={max_follow_ups}",
            state,
        )

    judgment = state.get("last_judgment")
    if judgment is None:
        raise ValueError("缺少 last_judgment，无法决定下一步")
    action = _action_from_suggestion(judgment.follow_up_suggestion)
    reason = f"last_judgment 建议：{judgment.follow_up_suggestion or '（空）'}"
    return _decide(action, reason, state)
