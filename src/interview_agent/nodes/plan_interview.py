"""plan_interview 节点：根据结构化 JD + 候选人画像调用 LLM 生成面试提纲。"""

from __future__ import annotations

from interview_agent.llm.client import DashScopeClient
from interview_agent.state import InterviewState


def plan_interview(state: InterviewState) -> dict:
    """调用 llm.plan_interview 产出 outline，并把阶段推进到 asking。

    新版链路：parse_inputs 已经把简历/JD 解析为结构化对象存进 state.resume/jd，
    plan_interview 直接拿结构化画像调用 LLM，不再二次传递原始文本（符合 prompts 原则 6）。
    """
    jd = state.get("jd")
    if jd is None:
        raise ValueError("jd（结构化 JD）为空，无法进行面试规划")

    resume = state.get("resume")  # 允许 None（简历解析失败时也能按 JD 裸规划）
    outline = DashScopeClient().plan_interview(jd, resume=resume)
    return {
        "outline": outline,
        "stage": "asking",
    }
