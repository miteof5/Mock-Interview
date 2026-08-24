"""parse_inputs 节点：调用 LLM 解析简历/JD 为结构化对象，带内容 hash 缓存。"""

from __future__ import annotations

from interview_agent.llm.client import DashScopeClient
from interview_agent.models import ParsedJD, ParsedResume
from interview_agent.state import InterviewState
from interview_agent.storage import parse_cache


def parse_inputs(state: InterviewState) -> dict:
    """调用 llm.parse_resume / parse_jd，并把阶段推进到 planning。

    缓存策略：对 resume_text / jd_text 算 SHA256，命中缓存则跳过 LLM 直接复用上次解析结果。
    内容一字节不同则 hash 不同，自动重新解析，无需手动失效。
    """
    resume_text = state.get("resume_text") or ""
    jd_text = state.get("jd_text") or ""
    if not resume_text.strip() or not jd_text.strip():
        raise ValueError("resume_text 和 jd_text 不能为空")

    resume_key = "resume_" + parse_cache.content_hash(resume_text)
    jd_key = "jd_" + parse_cache.content_hash(jd_text)

    # 简历：命中缓存则从 dict 重建，否则调 LLM 解析并存缓存
    resume_cached = parse_cache.get(resume_key)
    if resume_cached is not None:
        resume = ParsedResume(**resume_cached)
    else:
        resume = DashScopeClient().parse_resume(resume_text)
        parse_cache.set(resume_key, resume.model_dump(mode="json"))

    # JD：同上
    jd_cached = parse_cache.get(jd_key)
    if jd_cached is not None:
        jd = ParsedJD(**jd_cached)
    else:
        jd = DashScopeClient().parse_jd(jd_text)
        parse_cache.set(jd_key, jd.model_dump(mode="json"))

    return {
        "resume": resume,
        "jd": jd,
        "stage": "planning",
    }
