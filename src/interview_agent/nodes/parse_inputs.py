"""parse_inputs 节点：调用 LLM 解析简历/JD 为结构化对象。"""

from __future__ import annotations

from interview_agent.llm.client import DashScopeClient
from interview_agent.state import InterviewState


def parse_inputs(state: InterviewState) -> dict:
    """调用 llm.parse_resume / parse_jd，并把阶段推进到 planning。"""
    resume_text = state.get("resume_text") or ""
    jd_text = state.get("jd_text") or ""
    if not resume_text.strip() or not jd_text.strip():
        raise ValueError("resume_text 和 jd_text 不能为空")

    client = DashScopeClient()
    resume = client.parse_resume(resume_text)
    jd = client.parse_jd(jd_text)

    return {
        "resume": resume,
        "jd": jd,
        "stage": "planning",
    }
