"""parse_inputs 节点单测：mock DashScopeClient 的两个语义化方法，
不触达网络；覆盖正常路径、调用契约、空值边界。
"""

from __future__ import annotations

import pytest

from interview_agent.models import ParsedJD, ParsedResume
from interview_agent.nodes.parse_inputs import parse_inputs


@pytest.fixture
def fake_client(monkeypatch):
    """注入 stub DashScopeClient，记录调用参数并返回固定结构化对象。

    parse_inputs 节点内部是 `client = DashScopeClient()`，
    所以这里把节点命名空间里的 DashScopeClient 替换为一个工厂，
    调用后返回 _Stub 实例。返回 calls 字典供测试断言调用参数。
    """
    calls = {"resume": [], "jd": []}

    class _Stub:
        def parse_resume(self, resume_text: str) -> ParsedResume:
            calls["resume"].append(resume_text)
            return ParsedResume(
                raw_text=resume_text,
                name="张三",
                target_position="后端工程师",
                years_of_experience=3.0,
                skills=["Python", "FastAPI", "PostgreSQL"],
            )

        def parse_jd(self, jd_text: str) -> ParsedJD:
            calls["jd"].append(jd_text)
            return ParsedJD(
                raw_text=jd_text,
                title="后端工程师",
                company="ACME",
                responsibilities=["设计服务架构", "维护 API"],
                requirements=["3 年经验", "Python 熟练"],
                must_have=["Python", "分布式系统"],
                nice_to_have=["K8s"],
                keywords=["Python", "后端", "分布式"],
            )

    monkeypatch.setattr(
        "interview_agent.nodes.parse_inputs.DashScopeClient",
        lambda *args, **kwargs: _Stub(),
    )
    return calls


# ====================== 正常路径 + 调用契约 ======================

def test_parse_inputs_happy_path(fake_client):
    """正常路径：返回结构化 resume/jd，stage 推进到 planning。"""
    state = {"resume_text": "my resume", "jd_text": "my jd"}
    result = parse_inputs(state)
    assert result["stage"] == "planning"
    assert isinstance(result["resume"], ParsedResume)
    assert isinstance(result["jd"], ParsedJD)
    assert result["resume"].name == "张三"
    assert result["jd"].title == "后端工程师"


def test_parse_inputs_passes_raw_text_to_resume_parser(fake_client):
    """调用契约：client.parse_resume 收到的必须是 state 里的原始 resume_text。"""
    state = {"resume_text": "原始简历文本", "jd_text": "any"}
    parse_inputs(state)
    assert fake_client["resume"] == ["原始简历文本"]


def test_parse_inputs_passes_raw_text_to_jd_parser(fake_client):
    """调用契约：client.parse_jd 收到的必须是 state 里的原始 jd_text。"""
    state = {"resume_text": "any", "jd_text": "原始 JD 文本"}
    parse_inputs(state)
    assert fake_client["jd"] == ["原始 JD 文本"]


def test_parse_inputs_invokes_each_parser_once(fake_client):
    """调用契约：每次只调用 parse_resume / parse_jd 各一次，不重复。"""
    state = {"resume_text": "r", "jd_text": "j"}
    parse_inputs(state)
    assert len(fake_client["resume"]) == 1
    assert len(fake_client["jd"]) == 1


# ====================== 空值边界 ======================

def test_parse_inputs_rejects_empty_resume(fake_client):
    """空简历：应抛 ValueError，且不应触达 LLM。"""
    state = {"resume_text": "   ", "jd_text": "my jd"}
    with pytest.raises(ValueError, match="resume_text 和 jd_text 不能为空"):
        parse_inputs(state)
    # 关键：边界失败时不应调用 LLM
    assert fake_client["resume"] == []
    assert fake_client["jd"] == []


def test_parse_inputs_rejects_empty_jd(fake_client):
    """空 JD：应抛 ValueError，且不应触达 LLM。"""
    state = {"resume_text": "my resume", "jd_text": ""}
    with pytest.raises(ValueError, match="resume_text 和 jd_text 不能为空"):
        parse_inputs(state)
    assert fake_client["resume"] == []
    assert fake_client["jd"] == []


def test_parse_inputs_rejects_both_empty(fake_client):
    """两者都空：应抛 ValueError。"""
    state = {"resume_text": "", "jd_text": ""}
    with pytest.raises(ValueError, match="resume_text 和 jd_text 不能为空"):
        parse_inputs(state)


def test_parse_inputs_rejects_missing_keys(fake_client):
    """state 中缺少 resume_text / jd_text 键：state.get 返回 None → 退化为空串 → 抛错。"""
    state = {}
    with pytest.raises(ValueError, match="resume_text 和 jd_text 不能为空"):
        parse_inputs(state)
