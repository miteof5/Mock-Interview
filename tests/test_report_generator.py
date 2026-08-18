"""report_generator 薄封装单测：在 tmp_path 下构造会话目录 + raw_history.json，
注入 stub DashScopeClient，不触达网络。

被测函数链：load_session → 校验 judgments 非空 → AnswerJudgment.model_validate 重建 →
client.generate_report → 返回 InterviewReport。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interview_agent.evaluation.report_generator import generate_report_from_session
from interview_agent.models import AnswerJudgment, InterviewReport, ScoreItem


def _judgments_dicts() -> list[dict]:
    """构造 2 条 judgment 的纯 dict 形式（模拟 raw_history.json 里的内容）。"""
    return [
        {
            "topic_id": "t0",
            "question": "问题0",
            "answer": "回答0",
            "overall_score": 7.0,
            "scores": [
                {"dimension": "专业准确性", "score": 8.0, "evidence": "ok"},
            ],
            "follow_up_suggestion": "FOLLOW_UP",
            "summary": "小结0",
        },
        {
            "topic_id": "t1",
            "question": "问题1",
            "answer": "回答1",
            "overall_score": 8.0,
            "scores": [
                {"dimension": "专业准确性", "score": 9.0, "evidence": "good"},
            ],
            "follow_up_suggestion": "NEXT_TOPIC",
            "summary": "小结1",
        },
    ]


def _write_session(
    sessions_root: Path,
    session_id: str,
    judgments: list[dict] | None,
    *,
    include_judgments_key: bool = True,
) -> Path:
    """在 sessions_root 下构造会话目录 + raw_history.json，返回会话目录。

    include_judgments_key=False 时写入的 state 不含 judgments 键（用于测试缺失键兜底）。
    """
    session_dir = sessions_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    state: dict = {"session_id": session_id, "stage": "finished"}
    if include_judgments_key:
        state["judgments"] = judgments
    (session_dir / "raw_history.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return session_dir


class _StubClient:
    """记录 generate_report 调用参数并返回固定 InterviewReport。"""

    def __init__(self) -> None:
        self.calls = {"generate_report": []}

    def generate_report(self, session_id, judgments, *, model=None, temperature=0.2):
        self.calls["generate_report"].append({
            "session_id": session_id,
            "judgments": judgments,
        })
        return InterviewReport(
            session_id=session_id,
            overall_score=8.0,
            dimension_scores=[
                ScoreItem(dimension="专业准确性", score=8.0, evidence="ok"),
            ],
            summary="总体不错",
        )


@pytest.fixture
def stub_client():
    """返回 stub client 实例，测试通过 client= 参数注入。"""
    return _StubClient()


# ====================== 正常路径 + 调用契约 ======================

def test_happy_path_returns_report(tmp_path, stub_client):
    """正常路径：返回 InterviewReport。"""
    _write_session(tmp_path, "sess-1", _judgments_dicts())
    result = generate_report_from_session(
        "sess-1", sessions_root=tmp_path, client=stub_client
    )
    assert isinstance(result, InterviewReport)
    assert result.overall_score == 8.0
    assert result.summary == "总体不错"


def test_passes_session_id_to_client(tmp_path, stub_client):
    """调用契约：传给 client 的 session_id 与入参一致。"""
    _write_session(tmp_path, "my-session", _judgments_dicts())
    generate_report_from_session(
        "my-session", sessions_root=tmp_path, client=stub_client
    )
    assert stub_client.calls["generate_report"][0]["session_id"] == "my-session"


def test_rebuilds_judgments_as_pydantic_objects(tmp_path, stub_client):
    """调用契约：raw_history.json 里是 dict，传给 client 前应重建为 AnswerJudgment 实例。"""
    _write_session(tmp_path, "sess-1", _judgments_dicts())
    generate_report_from_session(
        "sess-1", sessions_root=tmp_path, client=stub_client
    )
    judgments = stub_client.calls["generate_report"][0]["judgments"]
    assert len(judgments) == 2
    # 关键：每个元素都是 AnswerJudgment 实例，不是 dict
    assert all(isinstance(j, AnswerJudgment) for j in judgments)
    assert judgments[0].topic_id == "t0"
    assert judgments[0].question == "问题0"
    assert judgments[1].overall_score == 8.0


def test_passes_full_judgments_list(tmp_path, stub_client):
    """调用契约：judgments 列表完整透传，数量与文件一致。"""
    _write_session(tmp_path, "sess-1", _judgments_dicts())
    generate_report_from_session(
        "sess-1", sessions_root=tmp_path, client=stub_client
    )
    judgments = stub_client.calls["generate_report"][0]["judgments"]
    assert len(judgments) == 2


def test_invokes_generate_report_once(tmp_path, stub_client):
    """调用契约：只调用 client.generate_report 一次。"""
    _write_session(tmp_path, "sess-1", _judgments_dicts())
    generate_report_from_session(
        "sess-1", sessions_root=tmp_path, client=stub_client
    )
    assert len(stub_client.calls["generate_report"]) == 1


def test_uses_injected_client_not_real_one(tmp_path, stub_client, monkeypatch):
    """依赖注入：传入 stub client 时不应实例化真实 DashScopeClient。

    通过把 DashScopeClient 替换为会抛错的占位类来验证——如果被实例化就会炸。
    """
    _write_session(tmp_path, "sess-1", _judgments_dicts())

    def _fail_if_instantiated(*args, **kwargs):
        raise AssertionError("不应实例化真实 DashScopeClient")

    monkeypatch.setattr(
        "interview_agent.evaluation.report_generator.DashScopeClient",
        _fail_if_instantiated,
    )
    # 传入 stub_client，应该完全不触碰真实 client
    result = generate_report_from_session(
        "sess-1", sessions_root=tmp_path, client=stub_client
    )
    assert isinstance(result, InterviewReport)


# ====================== 边界：会话不存在 ======================

def test_missing_session_dir_raises_file_not_found(tmp_path, stub_client):
    """会话目录不存在：load_session 抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        generate_report_from_session(
            "nonexistent-session", sessions_root=tmp_path, client=stub_client
        )
    # 不应触达 LLM
    assert stub_client.calls["generate_report"] == []


# ====================== 边界：judgments 空 / 缺失 ======================

def test_empty_judgments_raises(tmp_path, stub_client):
    """judgments 为空列表：应抛 ValueError，不应触达 LLM。"""
    _write_session(tmp_path, "sess-1", [])
    with pytest.raises(ValueError, match="没有 judgments"):
        generate_report_from_session(
            "sess-1", sessions_root=tmp_path, client=stub_client
        )
    assert stub_client.calls["generate_report"] == []


def test_missing_judgments_key_raises(tmp_path, stub_client):
    """state 缺少 judgments 键：state.get 返回 None → 退化为 [] → 抛 ValueError。"""
    _write_session(tmp_path, "sess-1", None, include_judgments_key=False)
    with pytest.raises(ValueError, match="没有 judgments"):
        generate_report_from_session(
            "sess-1", sessions_root=tmp_path, client=stub_client
        )
    assert stub_client.calls["generate_report"] == []


# ====================== 边界：session_id 合法性（透传 load_session 的校验）======================

def test_illegal_session_id_raises(tmp_path, stub_client):
    """非法 session_id（含路径穿越字符）：load_session 抛 ValueError。"""
    with pytest.raises(ValueError, match="非法的 session_id"):
        generate_report_from_session(
            "../escape", sessions_root=tmp_path, client=stub_client
        )
    assert stub_client.calls["generate_report"] == []
