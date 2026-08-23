"""graph 集成测试：真实节点接线 + mock LLM/retriever/interrupt。

测试聚焦在 graph 接线是否正确（节点顺序、条件路由、循环回边），
不重复测单个节点行为（已由各节点单测覆盖）。

mock 策略：
- DashScopeClient：所有语义化方法返回固定 Pydantic 对象；judge 的
  follow_up_suggestion 通过队列控制，引导流程走向 END/FOLLOW_UP/NEXT_TOPIC。
- retriever.retrieve：返回固定 KnowledgeChunk 列表。
- interrupt：直接返回固定 answer，不真正暂停，让 invoke 一次跑完。
"""

from __future__ import annotations

from typing import Any

import pytest

from interview_agent.graph import build_app
from interview_agent.models import (
    AnswerJudgment,
    InterviewOutline,
    InterviewQuestion,
    InterviewReport,
    InterviewTopic,
    KnowledgeChunk,
    ParsedJD,
    ParsedResume,
    ScoreItem,
)
from interview_agent.state import build_initial_state


# ---------------------------------------------------------------------------
# 可配置 FakeClient：通过 judge_suggestions 队列控制每轮 judge 的建议
# ---------------------------------------------------------------------------
class FakeClient:
    """记录所有调用 + 通过 judge_suggestions 队列控制流程走向。

    用法：
        client = FakeClient(judge_suggestions=["FOLLOW_UP", "END"])
        # 第一轮 judge 返回 "FOLLOW_UP"，第二轮返回 "END"
    """

    def __init__(self, judge_suggestions: list[str] | None = None) -> None:
        self.judge_suggestions = list(judge_suggestions or ["END"])
        self.calls: dict[str, list[dict]] = {
            "parse_resume": [],
            "parse_jd": [],
            "plan_interview": [],
            "ask_question": [],
            "follow_up": [],
            "judge": [],
            "generate_report": [],
        }

    def parse_resume(self, resume_text: str) -> ParsedResume:
        self.calls["parse_resume"].append({"resume_text": resume_text})
        return ParsedResume(
            raw_text=resume_text,
            name="张三",
            target_position="后端工程师",
            years_of_experience=3.0,
            skills=["Python"],
        )

    def parse_jd(self, jd_text: str) -> ParsedJD:
        self.calls["parse_jd"].append({"jd_text": jd_text})
        return ParsedJD(
            raw_text=jd_text,
            title="后端工程师",
            company="ACME",
            keywords=["Python", "分布式"],
        )

    def plan_interview(self, jd, resume=None, difficulty="中等") -> InterviewOutline:
        # 新版 client：参数是结构化 ParsedJD + ParsedResume|None；桩实现只用 jd.raw_text 便于断言
        jd_text = getattr(jd, "raw_text", str(jd))
        self.calls["plan_interview"].append({"jd_text": jd_text, "has_resume": resume is not None})
        return InterviewOutline(
            topics=[
                InterviewTopic(id="t0", title="项目经历", focus="架构设计"),
                InterviewTopic(id="t1", title="分布式系统", focus="一致性协议"),
            ]
        )

    def ask_question(self, jd_profile, resume_profile, *, topic, knowledge, history, difficulty):
        # 新版 client：前两个位置参数是结构化画像 ParsedJD|None / ParsedResume|None
        self.calls["ask_question"].append({"topic": topic})
        return InterviewQuestion(
            topic_id="t0",
            content=f"首问：{topic}",
            question_type="open",
        )

    def follow_up(self, *, topic, question, answer, difficulty):
        self.calls["follow_up"].append({
            "topic": topic,
            "question": question,
            "answer": answer,
        })
        return InterviewQuestion(
            topic_id="t0",
            content=f"追问：{question[:10]}",
            question_type="follow_up",
        )

    def judge(self, *, topic, question, answer, retrieval_results):
        # 从队列头部取一个 suggestion，队列空时默认 "END"
        suggestion = (
            self.judge_suggestions.pop(0)
            if self.judge_suggestions
            else "END"
        )
        self.calls["judge"].append({
            "topic": topic,
            "question": question,
            "answer": answer,
            "suggestion_used": suggestion,
        })
        return AnswerJudgment(
            topic_id="t0",
            question=question,
            answer=answer,
            overall_score=6.0,
            scores=[
                ScoreItem(dimension="专业准确性", score=8.0, evidence="ok"),
            ],
            follow_up_suggestion=suggestion,
            summary="回答不错",
        )

    def generate_report(self, session_id, judgments, *, model=None, temperature=0.2):
        self.calls["generate_report"].append({
            "session_id": session_id,
            "judgments_count": len(judgments),
        })
        return InterviewReport(
            session_id=session_id,
            overall_score=7.5,
            dimension_scores=[
                ScoreItem(dimension="专业准确性", score=8.0, evidence="ok"),
            ],
            summary="总体表现不错",
        )


@pytest.fixture
def mock_dependencies(monkeypatch):
    """统一注入 FakeClient + stub retrieve + stub interrupt。

    返回一个工厂函数，测试可传入 judge_suggestions 创建 FakeClient：
        client = mock_dependencies(judge_suggestions=["FOLLOW_UP", "END"])
    """
    holder: dict[str, Any] = {"client": None}

    def _factory(judge_suggestions: list[str] | None = None) -> FakeClient:
        client = FakeClient(judge_suggestions=judge_suggestions)
        holder["client"] = client
        # 替换所有节点命名空间里的 DashScopeClient 引用
        for module_name in (
            "parse_inputs",
            "plan_interview",
            "ask_question",
            "judge_answer",
            "generate_report",
        ):
            monkeypatch.setattr(
                f"interview_agent.nodes.{module_name}.DashScopeClient",
                lambda *args, **kwargs: client,
            )
        return client

    # stub retrieve：返回固定 KnowledgeChunk
    def _stub_retrieve(query: str, *args, **kwargs):
        return [KnowledgeChunk(content=f"知识-{query}", source="doc.md", score=0.9)]

    monkeypatch.setattr(
        "interview_agent.nodes.retrieve_knowledge.retrieve",
        _stub_retrieve,
    )

    # stub interrupt：直接返回固定 answer，不真正暂停
    def _stub_interrupt(payload):
        return "我用 Redis 做了分布式锁"

    monkeypatch.setattr(
        "interview_agent.nodes.wait_answer.interrupt",
        _stub_interrupt,
    )

    return _factory


def _initial_state(session_id: str = "integration-test") -> dict:
    """构造集成测试初始 state。"""
    return build_initial_state({
        "session_id": session_id,
        "resume_text": "fake resume",
        "jd_text": "fake jd",
    })


# ====================== 主链路：单轮即结束 ======================

def test_single_round_end_to_end(mock_dependencies):
    """单轮主链路：parse→plan→retrieve→ask→wait→judge(END)→decide→report→END。

    验证 graph 接线正确：节点顺序、条件路由（END 走 report）、最终 stage=finished。
    """
    client = mock_dependencies(judge_suggestions=["END"])
    app = build_app(mode="memory")
    result = app.invoke(
        _initial_state(),
        config={"configurable": {"thread_id": "single-round"}},
    )

    # 阶段：最终 finished
    assert result["stage"] == "finished"
    # report 存在且 session_id 正确
    assert result["report"] is not None
    assert result["report"].session_id == "integration-test"

    # 节点调用顺序验证：每个节点都被调用过
    assert len(client.calls["parse_resume"]) == 1
    assert len(client.calls["parse_jd"]) == 1
    assert len(client.calls["plan_interview"]) == 1
    assert len(client.calls["ask_question"]) == 1  # 首问一次
    assert len(client.calls["follow_up"]) == 0  # 单轮无追问
    assert len(client.calls["judge"]) == 1
    assert len(client.calls["generate_report"]) == 1

    # judge 收到的 judgments 数量 = 1（只评判了一轮）
    assert client.calls["generate_report"][0]["judgments_count"] == 1


def test_single_round_state_artifacts(mock_dependencies):
    """单轮结束后 state 关键字段：outline/judgments/history/question_count。"""
    mock_dependencies(judge_suggestions=["END"])
    app = build_app(mode="memory")
    result = app.invoke(
        _initial_state(),
        config={"configurable": {"thread_id": "artifacts"}},
    )

    # outline 存在且含 2 个主题
    assert result["outline"] is not None
    assert len(result["outline"].topics) == 2

    # judgments 累积了 1 条
    assert len(result["judgments"]) == 1
    assert result["judgments"][0].overall_score == 6.0

    # history 含 interviewer 首问 + candidate 回答 = 2 条
    assert len(result["history"]) == 2
    assert result["history"][0].role == "interviewer"
    assert result["history"][1].role == "candidate"

    # question_count = 1
    assert result["question_count"] == 1


# ====================== 多轮循环：FOLLOW_UP 后结束 ======================

def test_follow_up_loop_then_end(mock_dependencies):
    """两轮循环：第一轮 FOLLOW_UP（回到 retrieve→ask），第二轮 END。

    验证条件路由 follow_up → retrieve_knowledge 回边正确。
    """
    client = mock_dependencies(judge_suggestions=["FOLLOW_UP", "END"])
    app = build_app(mode="memory")
    result = app.invoke(
        _initial_state(),
        config={"configurable": {"thread_id": "follow-up-loop"}},
    )

    assert result["stage"] == "finished"

    # 首问 1 次 + 追问 1 次
    assert len(client.calls["ask_question"]) == 1
    assert len(client.calls["follow_up"]) == 1
    # judge 两次
    assert len(client.calls["judge"]) == 2
    # 最终报告含 2 条 judgments
    assert client.calls["generate_report"][0]["judgments_count"] == 2

    # question_count = 2（首问 + 追问）
    assert result["question_count"] == 2
    # follow_up_count = 1（第一轮 FOLLOW_UP 后自增到 1）
    assert result["follow_up_count"] == 1

    # history 累积 4 条：interviewer→candidate→interviewer→candidate
    assert len(result["history"]) == 4
    roles = [msg.role for msg in result["history"]]
    assert roles == ["interviewer", "candidate", "interviewer", "candidate"]


# ====================== 切题：NEXT_TOPIC 后结束 ======================

def test_next_topic_advances_index(mock_dependencies):
    """NEXT_TOPIC 切到第二个主题，然后再 END。

    验证条件路由 next_topic → retrieve_knowledge 回边 + current_topic_index 递增。
    """
    client = mock_dependencies(judge_suggestions=["NEXT_TOPIC", "END"])
    app = build_app(mode="memory")
    result = app.invoke(
        _initial_state(),
        config={"configurable": {"thread_id": "next-topic"}},
    )

    assert result["stage"] == "finished"

    # 两次首问（切题后是首问，不是追问）
    assert len(client.calls["ask_question"]) == 2
    assert len(client.calls["follow_up"]) == 0
    # judge 两次
    assert len(client.calls["judge"]) == 2

    # current_topic_index 推进到 1（第二个主题）
    assert result["current_topic_index"] == 1
    # 切题后 follow_up_count 重置为 0
    assert result["follow_up_count"] == 0


# ====================== 边界：max_questions 触发 end ======================

def test_max_questions_triggers_end(mock_dependencies):
    """max_questions=1 时，第一轮 judge 后 decide_next 直接 end（不等 judge 建议）。"""
    client = mock_dependencies(judge_suggestions=["FOLLOW_UP"])  # 建议追问但被边界否决
    app = build_app(mode="memory")
    state = _initial_state()
    state["max_questions"] = 1  # 只允许 1 题
    result = app.invoke(
        state,
        config={"configurable": {"thread_id": "max-q"}},
    )

    assert result["stage"] == "finished"
    # 只问了 1 题，没有追问（被 max_questions 否决）
    assert len(client.calls["ask_question"]) == 1
    assert len(client.calls["follow_up"]) == 0
    # decide_next 的 end 优先级高于 follow_up_suggestion
    assert result["last_decision"].action == "end"


# ====================== 边界：max_follow_ups 触发 next_topic ======================

def test_max_follow_ups_triggers_next_topic(mock_dependencies):
    """max_follow_ups=1 时，第一轮 FOLLOW_UP 后 decide_next 切题（第二轮首问）。

    注：max_follow_ups=1 意味着追问 1 次后强制切题。
    """
    client = mock_dependencies(judge_suggestions=["FOLLOW_UP", "END"])
    app = build_app(mode="memory")
    state = _initial_state()
    state["max_follow_ups"] = 1
    state["max_questions"] = 15  # 放宽，不触发 question 边界
    result = app.invoke(
        state,
        config={"configurable": {"thread_id": "max-fu"}},
    )

    assert result["stage"] == "finished"
    # 第一轮首问 + 第一轮 FOLLOW_UP 追问 → follow_up_count=1 触发 max → 切题
    # 第二轮首问（切题后重置）
    assert len(client.calls["ask_question"]) == 2
    assert len(client.calls["follow_up"]) == 1
    # current_topic_index 推进到 1
    assert result["current_topic_index"] == 1
