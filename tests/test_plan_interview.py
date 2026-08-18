"""plan_interview 节点单测：mock DashScopeClient.plan_interview，不触达网络。

覆盖正常路径、调用契约、空值边界。
"""

from __future__ import annotations

import pytest

from interview_agent.models import (
    InterviewOutline,
    InterviewTopic,
    ParsedJD,
    ParsedResume,
)
from interview_agent.nodes.plan_interview import plan_interview


def _stub_outline() -> InterviewOutline:
    """返回固定的 InterviewOutline，供 stub client 返回。"""
    return InterviewOutline(
        topics=[
            InterviewTopic(id="t0", title="项目经历", focus="架构设计"),
            InterviewTopic(id="t1", title="分布式系统", focus="一致性协议"),
        ]
    )


def _stub_jd() -> ParsedJD:
    """返回一个最小可用的结构化 JD，模拟 parse_inputs 节点的产出。"""
    return ParsedJD(
        raw_text="需要招一个后端工程师，3 年经验",
        title="后端工程师",
        must_have=["Python", "SQL"],
        nice_to_have=["Redis"],
    )


def _stub_resume() -> ParsedResume:
    """返回一个最小可用的结构化简历，模拟 parse_inputs 节点的产出。"""
    return ParsedResume(
        raw_text="张三，3 年后端经验",
        name="张三",
        skills=["Python"],
    )


@pytest.fixture
def fake_client(monkeypatch):
    """注入 stub DashScopeClient，记录 plan_interview 调用参数（结构化 jd/resume）。"""
    calls = {"jd": [], "resume": []}

    class _Stub:
        def plan_interview(self, jd, resume=None) -> InterviewOutline:
            calls["jd"].append(jd)
            calls["resume"].append(resume)
            return _stub_outline()

    monkeypatch.setattr(
        "interview_agent.nodes.plan_interview.DashScopeClient",
        lambda *args, **kwargs: _Stub(),
    )
    return calls


# ====================== 正常路径 + 调用契约 ======================

def test_plan_interview_happy_path(fake_client):
    """正常路径：返回 outline，stage 推进到 asking。"""
    state = {"jd": _stub_jd(), "resume": _stub_resume()}
    result = plan_interview(state)
    assert result["stage"] == "asking"
    outline = result["outline"]
    assert isinstance(outline, InterviewOutline)
    assert len(outline.topics) == 2
    assert outline.topics[0].id == "t0"
    assert outline.topics[1].title == "分布式系统"


def test_plan_interview_passes_structured_objects_to_client(fake_client):
    """调用契约：client.plan_interview 收到的是结构化 ParsedJD / ParsedResume 对象（非 raw str）。"""
    jd = _stub_jd()
    resume = _stub_resume()
    state = {"jd": jd, "resume": resume}
    plan_interview(state)
    assert fake_client["jd"][0] is jd           # 传同一个 ParsedJD 对象引用
    assert fake_client["resume"][0] is resume   # 传同一个 ParsedResume 对象引用


def test_plan_interview_resume_optional(fake_client):
    """简历解析失败时 resume=None，仍可正常规划（参数为可选）。"""
    jd = _stub_jd()
    state = {"jd": jd}
    plan_interview(state)
    assert fake_client["jd"][0] is jd
    assert fake_client["resume"][0] is None     # resume 缺失时显式传 None，不抛错


def test_plan_interview_invokes_once(fake_client):
    """调用契约：每次只调用 plan_interview 一次。"""
    state = {"jd": _stub_jd()}
    plan_interview(state)
    assert len(fake_client["jd"]) == 1


# ====================== 空值边界 ======================

def test_plan_interview_rejects_missing_jd(fake_client):
    """state 缺少 jd（结构化 JD 为 None）：应抛 ValueError，且不应触达 LLM。"""
    state = {"resume": _stub_resume()}   # 有简历但没有 jd
    with pytest.raises(ValueError, match="jd（结构化 JD）为空"):
        plan_interview(state)
    assert fake_client["jd"] == []


def test_plan_interview_rejects_explicit_none_jd(fake_client):
    """state 里 jd 显式为 None（parse_inputs 失败）：同样抛错。"""
    state = {"jd": None}
    with pytest.raises(ValueError, match="jd（结构化 JD）为空"):
        plan_interview(state)
    assert fake_client["jd"] == []
