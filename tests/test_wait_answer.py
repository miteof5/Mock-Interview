"""wait_answer 节点单测：mock langgraph.types.interrupt，不真正暂停。

节点逻辑：从 state 取 current_question → interrupt 暂停暴露问题 → resume 后校验 answer →
写 last_answer + 增量 candidate history。

mock 策略：替换节点命名空间里的 interrupt 引用为 stub，直接返回预设 answer。
"""

from __future__ import annotations

import pytest

from interview_agent.models import InterviewQuestion
from interview_agent.nodes.wait_answer import wait_answer


def _question(content: str = "请讲讲你的分布式项目") -> InterviewQuestion:
    """构造当前问题。"""
    return InterviewQuestion(
        topic_id="t0",
        content=content,
        question_type="open",
    )


@pytest.fixture
def mock_interrupt(monkeypatch):
    """让 interrupt 直接返回预设 answer，记录入参（不真正暂停）。

    用法：
        mock_interrupt["return_value"] = "我的回答"  # 设置 resume 返回值
        result = wait_answer(state)
        mock_interrupt["calls"][0]  # 读取 interrupt 入参（含 question）
    """
    state = {"return_value": "我的回答", "calls": []}

    def _stub_interrupt(payload):
        state["calls"].append(payload)
        return state["return_value"]

    monkeypatch.setattr(
        "interview_agent.nodes.wait_answer.interrupt",
        _stub_interrupt,
    )
    return state


# ====================== 正常路径 + 调用契约 ======================

def test_happy_path_writes_last_answer(mock_interrupt):
    """正常路径：resume 返回的回答写入 last_answer。"""
    state = {"current_question": _question()}
    result = wait_answer(state)
    assert result["last_answer"] == "我的回答"


def test_happy_path_returns_single_candidate_history(mock_interrupt):
    """history 返回增量单条，role=candidate，content=回答。"""
    state = {"current_question": _question()}
    result = wait_answer(state)
    assert isinstance(result["history"], list)
    assert len(result["history"]) == 1
    assert result["history"][0].role == "candidate"
    assert result["history"][0].content == "我的回答"


def test_answer_is_stripped(mock_interrupt):
    """回答前后空白被 strip()。"""
    mock_interrupt["return_value"] = "  我用 Redis 做了分布式锁  "
    state = {"current_question": _question()}
    result = wait_answer(state)
    assert result["last_answer"] == "我用 Redis 做了分布式锁"
    # history 里的 content 也应该是 strip 后的
    assert result["history"][0].content == "我用 Redis 做了分布式锁"


def test_interrupt_receives_current_question_content(mock_interrupt):
    """interrupt 入参含 current_question.content，供调用方展示。"""
    state = {"current_question": _question("请讲讲你的项目")}
    wait_answer(state)
    payload = mock_interrupt["calls"][0]
    assert payload["question"] == "请讲讲你的项目"


def test_interrupt_called_once(mock_interrupt):
    """每次只调用 interrupt 一次。"""
    state = {"current_question": _question()}
    wait_answer(state)
    assert len(mock_interrupt["calls"]) == 1


# ====================== 边界：current_question 缺失 ======================

def test_missing_question_passes_empty_string_to_interrupt(mock_interrupt):
    """current_question=None：interrupt 入参 question 为空串（不抛错，留给调用方兜底）。"""
    state = {"current_question": None}
    wait_answer(state)
    payload = mock_interrupt["calls"][0]
    assert payload["question"] == ""


def test_missing_question_key_passes_empty_string_to_interrupt(mock_interrupt):
    """state 缺少 current_question 键：state.get 返回 None → 入参 question 为空串。"""
    state = {}
    wait_answer(state)
    payload = mock_interrupt["calls"][0]
    assert payload["question"] == ""


# ====================== 边界：answer 校验 ======================

def test_empty_answer_raises(mock_interrupt):
    """resume 返回空串：应抛 ValueError。"""
    mock_interrupt["return_value"] = ""
    state = {"current_question": _question()}
    with pytest.raises(ValueError, match="候选人回答不能为空"):
        wait_answer(state)


def test_whitespace_only_answer_raises(mock_interrupt):
    """resume 返回纯空白：应抛 ValueError。"""
    mock_interrupt["return_value"] = "   "
    state = {"current_question": _question()}
    with pytest.raises(ValueError, match="候选人回答不能为空"):
        wait_answer(state)


def test_non_string_answer_raises(mock_interrupt):
    """resume 返回非字符串（如 dict）：应抛 ValueError。"""
    mock_interrupt["return_value"] = {"unexpected": "dict"}
    state = {"current_question": _question()}
    with pytest.raises(ValueError, match="候选人回答不能为空"):
        wait_answer(state)


def test_none_answer_raises(mock_interrupt):
    """resume 返回 None：应抛 ValueError。"""
    mock_interrupt["return_value"] = None
    state = {"current_question": _question()}
    with pytest.raises(ValueError, match="候选人回答不能为空"):
        wait_answer(state)
