"""generate_report 节点：调用 LLM 生成最终评估报告。"""

from __future__ import annotations

from interview_agent.llm.client import DashScopeClient
from interview_agent.state import InterviewState


def generate_report(state: InterviewState) -> dict:
    """汇总各轮评判生成报告，并把阶段置为 finished。"""
    session_id = state.get("session_id")
    if not session_id:
        raise ValueError("session_id 不能为空")
    judgments = state.get("judgments") or []
    if not judgments:
        raise ValueError("judgments 为空，无法生成报告")

    report = DashScopeClient().generate_report(session_id, judgments)
    return {
        "report": report,
        "stage": "finished",
    }
