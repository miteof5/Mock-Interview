"""ask_question 节点单测：mock DashScopeClient 的 ask_question / follow_up 方法。

节点逻辑：据 last_decision.action 分三路（follow_up / 首问 / next_topic 首问），
更新 current_question、history（增量单条）、question_count、follow_up_count。
"""

from __future__ import annotations

import pytest

from interview_agent.models import (
    DecisionResult,
    InterviewMessage,
    InterviewOutline,
    InterviewQuestion,
    InterviewTopic,
    KnowledgeChunk,
    ParsedJD,
    ParsedResume,
)
from interview_agent.nodes.ask_question import (
    _format_history,
    _topic_text,
    ask_question,
)


def _stub_jd() -> ParsedJD:
    """最小可用结构化 JD：模拟 parse_inputs 的产出，给 state["jd"] 用。"""
    return ParsedJD(raw_text="my jd", title="后端工程师", must_have=["Python"])


def _stub_resume() -> ParsedResume:
    """最小可用结构化简历：模拟 parse_inputs 的产出，给 state["resume"] 用。"""
    return ParsedResume(raw_text="my resume", name="张三", skills=["Python"])


def _outline(n: int = 1) -> InterviewOutline:
    """构造 n 个主题的提纲（含默认 difficulty/candidate_basis，对齐新版模型字段）。"""
    return InterviewOutline(
        topics=[
            InterviewTopic(
                id=f"t{i}",
                title=f"分布式系统{i}",
                focus=f"一致性协议{i}",
            )
            for i in range(n)
        ]
    )


def _prev_question() -> InterviewQuestion:
    """上一轮问题，用于 follow_up 分支。"""
    return InterviewQuestion(
        topic_id="t0",
        content="请讲讲你的分布式项目",
        question_type="open",
    )


@pytest.fixture
def fake_client(monkeypatch):
    """注入 stub DashScopeClient，记录两个语义化方法的调用参数。"""
    calls = {"ask_question": [], "follow_up": []}

    class _Stub:
        # 新版签名：前两个位置参数是 ParsedJD|None / ParsedResume|None
        def ask_question(
            self, jd_profile, resume_profile, *, topic, knowledge, history, difficulty
        ):
            calls["ask_question"].append({
                "jd_profile": jd_profile,
                "resume_profile": resume_profile,
                "topic": topic,
                "knowledge": knowledge,
                "history": history,
            })
            return InterviewQuestion(
                topic_id="t0",
                content="首问：请介绍你的项目",
                question_type="open",
            )

        def follow_up(self, *, topic, question, answer, difficulty):
            calls["follow_up"].append({
                "topic": topic,
                "question": question,
                "answer": answer,
            })
            return InterviewQuestion(
                topic_id="t0",
                content="追问：你提到的 X 怎么做的",
                question_type="follow_up",
            )

    monkeypatch.setattr(
        "interview_agent.nodes.ask_question.DashScopeClient",
        lambda *args, **kwargs: _Stub(),
    )
    return calls


# ====================== 辅助函数：_topic_text / _format_history ======================

def test_topic_text_includes_all_new_fields():
    """新版 _topic_text 不只拼 title:focus，还包含难度档位、候选人依据、是否必问。"""
    topic = InterviewTopic(
        id="t0",
        title="分布式系统",
        focus="一致性",
        difficulty="中等",
        candidate_basis="简历提及，有项目经历",
        must_ask=True,
    )
    text = _topic_text(topic)
    assert "主题标题：分布式系统" in text
    assert "考查重点：一致性" in text
    assert "难度档位：中等" in text
    assert "候选人经历依据：简历提及，有项目经历" in text
    assert "是否必问（JD核心要求）：是" in text


def test_topic_text_must_ask_false_shows_no():
    """must_ask=False 时显示「否」。"""
    topic = InterviewTopic(id="t0", title="A", focus="B", must_ask=False)
    assert "是否必问（JD核心要求）：否" in _topic_text(topic)


def test_format_history_empty_returns_placeholder():
    """空历史返回固定占位符。"""
    assert _format_history([]) == "(暂无历史对话)"
    assert _format_history(None) == "(暂无历史对话)"


def test_format_history_with_messages():
    """有消息时拼成 '面试官: xxx\\n候选人: yyy' 格式。"""
    history = [
        InterviewMessage(role="interviewer", content="你好"),
        InterviewMessage(role="candidate", content="你好，我是张三"),
    ]
    text = _format_history(history)
    assert "面试官: 你好" in text
    assert "候选人: 你好，我是张三" in text


# ====================== 首问分支：action is None ======================

def test_first_question_when_action_none_calls_ask_question(fake_client):
    """action=None（首轮）：调 client.ask_question，不调 follow_up。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": None,
        "jd": _stub_jd(),
        "resume": _stub_resume(),
        "retrieval_results": [],
        "history": [],
        "question_count": 0,
        "follow_up_count": 0,
    }
    result = ask_question(state)
    assert len(fake_client["ask_question"]) == 1
    assert fake_client["follow_up"] == []
    assert result["current_question"].content == "首问：请介绍你的项目"
    assert result["current_question"].question_type == "open"


def test_first_question_resets_follow_up_count(fake_client):
    """首问分支强制重置 follow_up_count=0（即便之前是 2）。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": None,
        "jd": _stub_jd(),
        "resume": _stub_resume(),
        "question_count": 5,
        "follow_up_count": 2,  # 之前追问过 2 次
    }
    result = ask_question(state)
    assert result["follow_up_count"] == 0


def test_first_question_increments_question_count(fake_client):
    """首问分支 question_count +1。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": None,
        "jd": _stub_jd(),
        "resume": _stub_resume(),
        "question_count": 3,
    }
    result = ask_question(state)
    assert result["question_count"] == 4


# ====================== 首问分支：action == next_topic ======================

def test_next_topic_action_calls_ask_question_and_resets_count(fake_client):
    """action=next_topic：走首问路径，follow_up_count 重置为 0。"""
    state = {
        "outline": _outline(2),
        "current_topic_index": 1,  # 已切到第二个主题
        "last_decision": DecisionResult(action="next_topic", reason="切题"),
        "jd": _stub_jd(),
        "resume": _stub_resume(),
    }
    result = ask_question(state)
    assert len(fake_client["ask_question"]) == 1
    assert fake_client["follow_up"] == []
    assert result["follow_up_count"] == 0


def test_next_topic_uses_correct_topic_by_index(fake_client):
    """next_topic 后 index=1，取第二个主题，topic 文本包含 title/focus/difficulty/candidate_basis。"""
    state = {
        "outline": _outline(2),
        "current_topic_index": 1,
        "last_decision": DecisionResult(action="next_topic", reason="切题"),
        "jd": _stub_jd(),
        "resume": _stub_resume(),
    }
    ask_question(state)
    topic_text = fake_client["ask_question"][0]["topic"]
    # topics[1].title="分布式系统1" focus="一致性协议1"
    assert "主题标题：分布式系统1" in topic_text
    assert "考查重点：一致性协议1" in topic_text


# ====================== follow_up 分支 ======================

def test_follow_up_calls_client_follow_up(fake_client):
    """action=follow_up：调 client.follow_up，不调 ask_question。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": DecisionResult(action="follow_up", reason="继续追问"),
        "current_question": _prev_question(),
        "last_answer": "我用了 Redis 分布式锁",
        "question_count": 1,
        "follow_up_count": 0,
    }
    result = ask_question(state)
    assert len(fake_client["follow_up"]) == 1
    assert fake_client["ask_question"] == []
    assert result["current_question"].question_type == "follow_up"


def test_follow_up_increments_follow_up_count(fake_client):
    """follow_up 分支 follow_up_count +1。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": DecisionResult(action="follow_up", reason=""),
        "current_question": _prev_question(),
        "last_answer": "answer",
        "follow_up_count": 1,
    }
    result = ask_question(state)
    assert result["follow_up_count"] == 2


def test_follow_up_uses_previous_question_content(fake_client):
    """follow_up 把上一轮问题的 content 传给 client.follow_up。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": DecisionResult(action="follow_up", reason=""),
        "current_question": _prev_question(),
        "last_answer": "我的回答",
    }
    ask_question(state)
    assert fake_client["follow_up"][0]["question"] == "请讲讲你的分布式项目"


def test_follow_up_uses_last_answer(fake_client):
    """follow_up 把 last_answer 传给 client.follow_up。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": DecisionResult(action="follow_up", reason=""),
        "current_question": _prev_question(),
        "last_answer": "我用 Redis 做了分布式锁",
    }
    ask_question(state)
    assert fake_client["follow_up"][0]["answer"] == "我用 Redis 做了分布式锁"


def test_follow_up_missing_previous_raises(fake_client):
    """follow_up 分支缺少 current_question：应抛 ValueError，不触达 LLM。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": DecisionResult(action="follow_up", reason=""),
        "current_question": None,
        "last_answer": "answer",
    }
    with pytest.raises(ValueError, match="follow_up 分支缺少上一轮问题"):
        ask_question(state)
    assert fake_client["follow_up"] == []


def test_follow_up_missing_answer_raises(fake_client):
    """follow_up 分支缺少 last_answer：应抛 ValueError。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": DecisionResult(action="follow_up", reason=""),
        "current_question": _prev_question(),
        "last_answer": None,
    }
    with pytest.raises(ValueError, match="follow_up 分支缺少候选人回答"):
        ask_question(state)
    assert fake_client["follow_up"] == []


def test_follow_up_empty_answer_raises(fake_client):
    """follow_up 分支 last_answer 为纯空白：应抛 ValueError。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": DecisionResult(action="follow_up", reason=""),
        "current_question": _prev_question(),
        "last_answer": "   ",
    }
    with pytest.raises(ValueError, match="follow_up 分支缺少候选人回答"):
        ask_question(state)


# ====================== history 增量契约 ======================

def test_history_returns_single_interviewer_message(fake_client):
    """history 返回增量单条（不是全量），role=interviewer。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": None,
        "jd": _stub_jd(),
        "resume": _stub_resume(),
        "history": [],  # 节点不读 history 来构造增量，只用于 _format_history 传给 LLM
    }
    result = ask_question(state)
    assert isinstance(result["history"], list)
    assert len(result["history"]) == 1
    assert result["history"][0].role == "interviewer"
    assert result["history"][0].content == "首问：请介绍你的项目"


# ====================== 调用契约：首问传给 LLM 的完整上下文 ======================

def test_first_question_passes_full_context_to_client(fake_client):
    """首问把结构化 jd/resume + topic/knowledge/history 全量传给 client.ask_question。"""
    jd = _stub_jd()
    resume = _stub_resume()
    knowledge = [KnowledgeChunk(content="知识", source="d.md", score=0.8)]
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": None,
        "jd": jd,
        "resume": resume,
        "retrieval_results": knowledge,
        "history": [InterviewMessage(role="interviewer", content="hi")],
    }
    ask_question(state)
    call = fake_client["ask_question"][0]
    # 结构化画像透传同一个对象引用（非序列化新对象/非原始文本）
    assert call["jd_profile"] is jd
    assert call["resume_profile"] is resume
    # 第一个主题的文本包含标题和重点（新版多字段格式）
    assert "主题标题：分布式系统0" in call["topic"]
    assert "考查重点：一致性协议0" in call["topic"]
    # knowledge 直接透传（节点不格式化，client 内部用 format_knowledge）
    assert call["knowledge"] is knowledge
    # history 被格式化为人话对话
    assert "面试官: hi" in call["history"]


# ====================== 边界：outline / index / action ======================

def test_ask_question_missing_outline_raises(fake_client):
    """outline=None：应抛 ValueError。"""
    state = {"outline": None, "current_topic_index": 0, "last_decision": None}
    with pytest.raises(ValueError, match="outline 为空"):
        ask_question(state)


def test_ask_question_empty_topics_raises(fake_client):
    """outline.topics=[]：应抛 ValueError。"""
    from interview_agent.models import InterviewOutline as Outline
    state = {
        "outline": Outline(topics=[]),
        "current_topic_index": 0,
        "last_decision": None,
    }
    with pytest.raises(ValueError, match="outline 为空"):
        ask_question(state)


def test_ask_question_index_overflow_raises(fake_client):
    """index 越界：应抛 ValueError。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 5,
        "last_decision": None,
    }
    with pytest.raises(ValueError, match="current_topic_index 越界"):
        ask_question(state)


def test_ask_question_unknown_action_raises(fake_client):
    """未知 action 值：应抛 ValueError。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "last_decision": DecisionResult(action="end", reason=""),
    }
    with pytest.raises(ValueError, match="未知 last_decision.action"):
        ask_question(state)
