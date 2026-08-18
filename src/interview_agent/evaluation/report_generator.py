"""报告生成薄封装：从已保存会话读取 judgments 并调用 LLM 生成报告。"""

from __future__ import annotations

from pathlib import Path

from interview_agent.llm.client import DashScopeClient
from interview_agent.models import AnswerJudgment, InterviewReport
from interview_agent.storage.session_store import load_session


def generate_report_from_session(
    session_id: str,
    *,
    sessions_root: Path | None = None,
    client: DashScopeClient | None = None,
) -> InterviewReport:
    """加载会话中的 judgments，调用 llm.generate_report 生成最终报告。"""
    state = load_session(session_id, sessions_root=sessions_root)
    raw_judgments = state.get("judgments") or []
    if not raw_judgments:
        raise ValueError(f"会话 {session_id} 没有 judgments，无法生成报告")

    judgments = [AnswerJudgment.model_validate(item) for item in raw_judgments]
    llm = client or DashScopeClient()
    return llm.generate_report(session_id, judgments)


__all__ = ["generate_report_from_session"]
