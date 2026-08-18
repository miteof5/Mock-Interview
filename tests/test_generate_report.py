"""generate_report 节点单测：mock DashScopeClient.generate_report，不触达网络。

节点逻辑最简单：取 session_id + judgments → 调 LLM → 返回 {report, stage="finished"}。
兜底（session_id 覆盖 / rounds 填充 / 维度对齐）由 client 层负责，节点层不感知，
所以本测试聚焦在调用契约 + 边界。
"""

from __future__ import annotations

import pytest

from interview_agent.models import (
    AnswerJudgment,
    InterviewReport,
    ScoreItem,
)
from interview_agent.nodes.generate_report import generate_report


def _judgments(n: int = 2) -> list[AnswerJudgment]:
    """构造 n 条评判历史。"""
    return [
        AnswerJudgment(
            topic_id=f"t{i}",
            question=f"问题{i}",
            answer=f"回答{i}",
            overall_score=7.0 + i * 0.5,
            scores=[
                ScoreItem(dimension="专业准确性", score=8.0, evidence="ok"),
            ],
            follow_up_suggestion="FOLLOW_UP",
            summary=f"小结{i}",
        )
        for i in range(n)
    ]


@pytest.fixture
def fake_client(monkeypatch):
    """注入 stub DashScopeClient，记录 generate_report 调用参数并返回固定 report。"""
    calls = {"generate_report": []}

    class _Stub:
        def generate_report(self, session_id, judgments, *, model=None, temperature=0.2):
            calls["generate_report"].append({
                "session_id": session_id,
                "judgments": judgments,
            })
            return InterviewReport(
                session_id=session_id,
                overall_score=8.0,
                dimension_scores=[
                    ScoreItem(dimension="专业准确性", score=8.0, evidence="ok"),
                ],
                summary="总体表现不错",
            )

    monkeypatch.setattr(
        "interview_agent.nodes.generate_report.DashScopeClient",
        lambda *args, **kwargs: _Stub(),
    )
    return calls


# ====================== 正常路径 + 调用契约 ======================

def test_generate_report_happy_path(fake_client):
    """正常路径：返回 report + stage=finished。"""
    state = {
        "session_id": "sess-123",
        "judgments": _judgments(2),
    }
    result = generate_report(state)
    assert result["stage"] == "finished"
    assert isinstance(result["report"], InterviewReport)
    assert result["report"].session_id == "sess-123"
    assert result["report"].overall_score == 8.0


def test_generate_report_passes_session_id_to_client(fake_client):
    """调用契约：传给 LLM 的 session_id 必须与 state 一致。"""
    state = {
        "session_id": "my-session-id",
        "judgments": _judgments(1),
    }
    generate_report(state)
    assert fake_client["generate_report"][0]["session_id"] == "my-session-id"


def test_generate_report_passes_judgments_to_client(fake_client):
    """调用契约：judgments 列表透传给 LLM。"""
    judgments = _judgments(3)
    state = {
        "session_id": "sess",
        "judgments": judgments,
    }
    generate_report(state)
    # 透传同一对象引用（节点不做拷贝/加工）
    assert fake_client["generate_report"][0]["judgments"] is judgments


def test_generate_report_invokes_once(fake_client):
    """调用契约：每次只调用 generate_report 一次。"""
    state = {
        "session_id": "sess",
        "judgments": _judgments(1),
    }
    generate_report(state)
    assert len(fake_client["generate_report"]) == 1


# ====================== 边界：session_id ======================

def test_generate_report_missing_session_id_raises(fake_client):
    """session_id=None：应抛 ValueError，不应触达 LLM。"""
    state = {
        "session_id": None,
        "judgments": _judgments(1),
    }
    with pytest.raises(ValueError, match="session_id 不能为空"):
        generate_report(state)
    assert fake_client["generate_report"] == []


def test_generate_report_empty_session_id_raises(fake_client):
    """session_id 为空串：应抛 ValueError。"""
    state = {
        "session_id": "",
        "judgments": _judgments(1),
    }
    with pytest.raises(ValueError, match="session_id 不能为空"):
        generate_report(state)


def test_generate_report_missing_session_id_key_raises(fake_client):
    """state 缺少 session_id 键：state.get 返回 None → 抛错。"""
    state = {"judgments": _judgments(1)}
    with pytest.raises(ValueError, match="session_id 不能为空"):
        generate_report(state)


# ====================== 边界：judgments ======================

def test_generate_report_missing_judgments_raises(fake_client):
    """judgments=None：应抛 ValueError，不应触达 LLM。"""
    state = {
        "session_id": "sess",
        "judgments": None,
    }
    with pytest.raises(ValueError, match="judgments 为空"):
        generate_report(state)
    assert fake_client["generate_report"] == []


def test_generate_report_empty_judgments_raises(fake_client):
    """judgments=[]：应抛 ValueError。"""
    state = {
        "session_id": "sess",
        "judgments": [],
    }
    with pytest.raises(ValueError, match="judgments 为空"):
        generate_report(state)


def test_generate_report_missing_judgments_key_raises(fake_client):
    """state 缺少 judgments 键：state.get 返回 None → 退化为 [] → 抛错。"""
    state = {"session_id": "sess"}
    with pytest.raises(ValueError, match="judgments 为空"):
        generate_report(state)
