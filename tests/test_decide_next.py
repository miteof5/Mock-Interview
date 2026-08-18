"""decide_next 节点单测：覆盖优先级边界 + 三选一解析 + 越界兜底。

decide_next 是纯逻辑节点（不触达 LLM），所以测试不需要 mock 网络；
只需构造 InterviewState 子集即可。
"""

from __future__ import annotations

import pytest

from interview_agent.models import (
    AnswerJudgment,
    InterviewOutline,
    InterviewTopic,
)
from interview_agent.nodes.decide_next import decide_next


def _outline(n: int = 3) -> InterviewOutline:
    """构造 n 个主题的提纲，id 形如 t0/t1/...，便于断言。"""
    return InterviewOutline(
        topics=[
            InterviewTopic(id=f"t{i}", title=f"主题{i}", focus=f"focus{i}")
            for i in range(n)
        ]
    )


def _base_state(**overrides) -> dict:
    """构造一个"未达任何边界、last_judgment 建议 FOLLOW_UP"的基准 state。

    所有优先级测试都从这里出发，按需覆盖字段。
    """
    return {
        "outline": _outline(3),
        "current_topic_index": 0,
        "current_topic_id": "t0",
        "question_count": 1,
        "max_questions": 15,
        "follow_up_count": 0,
        "max_follow_ups": 2,
        "last_judgment": AnswerJudgment(
            topic_id="t0",
            question="q",
            answer="a",
            overall_score=7,
            follow_up_suggestion="FOLLOW_UP",
        ),
        **overrides,
    }


# ====================== 优先级 1：question_count >= max_questions ======================

def test_question_count_equals_max_returns_end():
    """question_count == max_questions：优先级 1 触发，立即 end。"""
    state = _base_state(question_count=15, max_questions=15)
    result = decide_next(state)
    assert result["last_decision"].action == "end"
    assert result["stage"] == "finished"
    assert "question_count=15" in result["last_decision"].reason


def test_question_count_exceeds_max_returns_end():
    """question_count > max_questions：也应 end（边界容错）。"""
    state = _base_state(question_count=16, max_questions=15)
    result = decide_next(state)
    assert result["last_decision"].action == "end"


def test_question_count_priority_over_follow_up_max():
    """优先级测试：question_count 和 follow_up_count 都达上限时，以 end 优先。"""
    state = _base_state(
        question_count=15, max_questions=15,
        follow_up_count=2, max_follow_ups=2,
    )
    result = decide_next(state)
    assert result["last_decision"].action == "end"


# ====================== 优先级 2：follow_up_count >= max_follow_ups ======================

def test_follow_up_count_equals_max_returns_next_topic():
    """follow_up_count == max_follow_ups：优先级 2 触发，切下一主题。"""
    state = _base_state(follow_up_count=2, max_follow_ups=2)
    result = decide_next(state)
    assert result["last_decision"].action == "next_topic"
    assert result["current_topic_index"] == 1
    assert result["current_topic_id"] == "t1"
    assert "follow_up_count=2" in result["last_decision"].reason


def test_follow_up_count_exceeds_max_returns_next_topic():
    """follow_up_count > max_follow_ups：也应切题（边界容错）。"""
    state = _base_state(follow_up_count=3, max_follow_ups=2)
    result = decide_next(state)
    assert result["last_decision"].action == "next_topic"


def test_follow_up_count_max_at_last_topic_returns_end():
    """follow_up_count 达上限但已是最后一个主题：next_topic 自动改 end。"""
    state = _base_state(
        current_topic_index=2,
        current_topic_id="t2",
        follow_up_count=2,
        max_follow_ups=2,
    )
    result = decide_next(state)
    # 由于 next_topic 越界，_decide 把 action 改成 end
    assert result["last_decision"].action == "end"
    assert result["stage"] == "finished"
    assert "没有更多主题" in result["last_decision"].reason


# ====================== 优先级 3：follow_up_suggestion 三选一 ======================

def test_suggestion_follow_up_keeps_topic():
    """建议 FOLLOW_UP：停留在当前主题，刷新 current_topic_id 与 index 对齐。"""
    state = _base_state(current_topic_index=1, current_topic_id="t1")
    result = decide_next(state)
    assert result["last_decision"].action == "follow_up"
    # follow_up 分支不改 current_topic_index（不在返回 dict 里），但会刷新 id
    assert result["current_topic_id"] == "t1"


def test_suggestion_next_topic_advances_index():
    """建议 NEXT_TOPIC：索引+1，更新 current_topic_id。"""
    state = _base_state(
        last_judgment=AnswerJudgment(
            topic_id="t0", question="q", answer="a",
            overall_score=8, follow_up_suggestion="NEXT_TOPIC",
        )
    )
    result = decide_next(state)
    assert result["last_decision"].action == "next_topic"
    assert result["current_topic_index"] == 1
    assert result["current_topic_id"] == "t1"


def test_suggestion_end_finishes_interview():
    """建议 END：直接结束。"""
    state = _base_state(
        last_judgment=AnswerJudgment(
            topic_id="t0", question="q", answer="a",
            overall_score=9, follow_up_suggestion="END",
        )
    )
    result = decide_next(state)
    assert result["last_decision"].action == "end"
    assert result["stage"] == "finished"


def test_suggestion_empty_defaults_to_follow_up():
    """建议为空串：兜底为 follow_up。"""
    state = _base_state(
        last_judgment=AnswerJudgment(
            topic_id="t0", question="q", answer="a",
            overall_score=7, follow_up_suggestion="",
        )
    )
    result = decide_next(state)
    assert result["last_decision"].action == "follow_up"


def test_suggestion_chinese_end_keyword():
    """中文兜底：建议包含「结束」→ end。"""
    state = _base_state(
        last_judgment=AnswerJudgment(
            topic_id="t0", question="q", answer="a",
            overall_score=7, follow_up_suggestion="候选人表现优秀，建议结束面试",
        )
    )
    result = decide_next(state)
    assert result["last_decision"].action == "end"


def test_suggestion_chinese_next_topic_keyword():
    """中文兜底：建议包含「换题」→ next_topic。"""
    state = _base_state(
        last_judgment=AnswerJudgment(
            topic_id="t0", question="q", answer="a",
            overall_score=7, follow_up_suggestion="此题已充分考察，建议换题",
        )
    )
    result = decide_next(state)
    assert result["last_decision"].action == "next_topic"


def test_suggestion_unknown_text_defaults_to_follow_up():
    """建议文本无法识别任何关键词：兜底为 follow_up。"""
    state = _base_state(
        last_judgment=AnswerJudgment(
            topic_id="t0", question="q", answer="a",
            overall_score=7, follow_up_suggestion="随便一段不相关的话",
        )
    )
    result = decide_next(state)
    assert result["last_decision"].action == "follow_up"


# ====================== 异常分支 ======================

def test_missing_last_judgment_raises():
    """优先级都未触发但 last_judgment 缺失：应抛 ValueError。"""
    state = _base_state(last_judgment=None)
    with pytest.raises(ValueError, match="缺少 last_judgment"):
        decide_next(state)


def test_next_topic_without_outline_raises():
    """建议 NEXT_TOPIC 但 outline 缺失：应抛 ValueError。"""
    state = _base_state(
        outline=None,
        last_judgment=AnswerJudgment(
            topic_id="t0", question="q", answer="a",
            overall_score=7, follow_up_suggestion="NEXT_TOPIC",
        ),
    )
    with pytest.raises(ValueError, match="outline 为空"):
        decide_next(state)


def test_follow_up_without_outline_raises():
    """建议 FOLLOW_UP 但 outline 缺失：应抛 ValueError（follow_up 分支也读 outline）。"""
    state = _base_state(
        outline=None,
        last_judgment=AnswerJudgment(
            topic_id="t0", question="q", answer="a",
            overall_score=7, follow_up_suggestion="FOLLOW_UP",
        ),
    )
    with pytest.raises(ValueError, match="outline 为空"):
        decide_next(state)
