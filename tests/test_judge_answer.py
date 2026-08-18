"""judge_answer 节点单测：mock DashScopeClient.judge，验证调用契约 + 增量返回 + 维度对齐。

节点关键契约：
1. 返回 judgments=[单条]（增量，不返回全量）
2. 节点层再调一次 normalize_score_items 兜底维度对齐
3. 边界：outline/索引/问题/回答缺失时抛 ValueError
"""

from __future__ import annotations

import pytest

from interview_agent.evaluation.rubric import RUBRIC_DIMENSION_NAMES
from interview_agent.models import (
    AnswerJudgment,
    InterviewOutline,
    InterviewQuestion,
    InterviewTopic,
    KnowledgeChunk,
    ScoreItem,
)
from interview_agent.nodes.judge_answer import judge_answer


def _outline(n: int = 1) -> InterviewOutline:
    """构造 n 个主题的提纲。"""
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


def _question() -> InterviewQuestion:
    """当前问题。"""
    return InterviewQuestion(
        topic_id="t0",
        content="请讲讲你的分布式项目",
        question_type="open",
    )


def _judgment_with_partial_scores() -> AnswerJudgment:
    """stub 返回的 judgment：只含 2 个维度的 scores，用于验证节点层补齐。"""
    return AnswerJudgment(
        topic_id="t0",
        question="请讲讲你的分布式项目",
        answer="我用 Redis 做了分布式锁",
        overall_score=7.5,
        scores=[
            ScoreItem(dimension="专业准确性", score=8.0, evidence="提到了 Redis"),
            ScoreItem(dimension="表达结构", score=7.0, evidence="条理清晰"),
            # 故意缺 "岗位匹配度" 和 "应变能力"
        ],
        follow_up_suggestion="FOLLOW_UP",
        summary="回答不错",
    )


@pytest.fixture
def fake_client(monkeypatch):
    """注入 stub DashScopeClient，记录 judge 调用参数并返回固定 judgment。"""
    calls = {"judge": []}

    class _Stub:
        def judge(self, *, topic, question, answer, retrieval_results):
            calls["judge"].append({
                "topic": topic,
                "question": question,
                "answer": answer,
                "retrieval_results": retrieval_results,
            })
            return _judgment_with_partial_scores()

    monkeypatch.setattr(
        "interview_agent.nodes.judge_answer.DashScopeClient",
        lambda *args, **kwargs: _Stub(),
    )
    return calls


# ====================== 正常路径 + 调用契约 ======================

def test_judge_answer_happy_path(fake_client):
    """正常路径：返回 last_judgment + judgments=[单条]。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "我用 Redis 做了分布式锁",
        "retrieval_results": [],
    }
    result = judge_answer(state)
    assert "last_judgment" in result
    assert "judgments" in result
    assert isinstance(result["last_judgment"], AnswerJudgment)
    # 增量契约：judgments 只返回单条
    assert len(result["judgments"]) == 1
    assert result["judgments"][0] is result["last_judgment"]


def test_judge_answer_passes_topic_text_to_client(fake_client):
    """调用契约：传给 judge 的 topic = '{title}：{focus}'（中文全角冒号）。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
    }
    judge_answer(state)
    assert fake_client["judge"][0]["topic"] == "分布式系统0：一致性协议0"


def test_judge_answer_passes_question_content_to_client(fake_client):
    """调用契约：传给 judge 的 question 是 current_question.content。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
    }
    judge_answer(state)
    assert fake_client["judge"][0]["question"] == "请讲讲你的分布式项目"


def test_judge_answer_passes_answer_to_client(fake_client):
    """调用契约：传给 judge 的 answer 是 state.last_answer。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "我用 Redis 做了分布式锁",
    }
    judge_answer(state)
    assert fake_client["judge"][0]["answer"] == "我用 Redis 做了分布式锁"


def test_judge_answer_passes_retrieval_results_to_client(fake_client):
    """调用契约：retrieval_results 透传给 judge（节点不格式化，client 内部处理）。"""
    chunks = [KnowledgeChunk(content="知识1", source="d.md", score=0.9)]
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
        "retrieval_results": chunks,
    }
    judge_answer(state)
    assert fake_client["judge"][0]["retrieval_results"] is chunks


def test_judge_answer_invokes_judge_once(fake_client):
    """调用契约：每次只调用 judge 一次。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
    }
    judge_answer(state)
    assert len(fake_client["judge"]) == 1


# ====================== 维度对齐：节点层 normalize 兜底 ======================

def test_judge_answer_normalizes_scores_to_four_dimensions(fake_client):
    """stub 返回的 scores 只有 2 个维度，节点层 normalize 后应补齐到 4 个。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
    }
    result = judge_answer(state)
    scores = result["last_judgment"].scores
    assert len(scores) == 4
    actual_dims = [s.dimension for s in scores]
    assert actual_dims == list(RUBRIC_DIMENSION_NAMES)


def test_judge_answer_fills_missing_dims_with_overall_score(fake_client):
    """缺失维度用 overall_score 作为 default_score 兜底。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
    }
    result = judge_answer(state)
    scores = result["last_judgment"].scores
    # stub 的 overall_score=7.5，缺失的 "岗位匹配度"/"应变能力" 应为 7.5
    missing = {s.dimension: s.score for s in scores if s.dimension in ("岗位匹配度", "应变能力")}
    assert missing["岗位匹配度"] == 7.5
    assert missing["应变能力"] == 7.5


def test_judge_answer_preserves_existing_dims(fake_client):
    """已存在的维度分数不被 normalize 覆盖。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
    }
    result = judge_answer(state)
    scores = result["last_judgment"].scores
    by_dim = {s.dimension: s.score for s in scores}
    # stub 返回 "专业准确性"=8.0, "表达结构"=7.0，应保留
    assert by_dim["专业准确性"] == 8.0
    assert by_dim["表达结构"] == 7.0


# ====================== 边界：outline / 索引 ======================

def test_judge_answer_missing_outline_raises(fake_client):
    """outline=None：应抛 ValueError，不应触达 LLM。"""
    state = {
        "outline": None,
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
    }
    with pytest.raises(ValueError, match="outline 为空"):
        judge_answer(state)
    assert fake_client["judge"] == []


def test_judge_answer_empty_topics_raises(fake_client):
    """outline.topics=[]：应抛 ValueError。"""
    state = {
        "outline": InterviewOutline(topics=[]),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "answer",
    }
    with pytest.raises(ValueError, match="outline 为空"):
        judge_answer(state)


def test_judge_answer_index_overflow_raises(fake_client):
    """index 越界：应抛 ValueError。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 5,
        "current_question": _question(),
        "last_answer": "answer",
    }
    with pytest.raises(ValueError, match="current_topic_index 越界"):
        judge_answer(state)


# ====================== 边界：current_question / last_answer ======================

def test_judge_answer_missing_question_raises(fake_client):
    """current_question=None：应抛 ValueError，不应触达 LLM。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": None,
        "last_answer": "answer",
    }
    with pytest.raises(ValueError, match="缺少当前问题"):
        judge_answer(state)
    assert fake_client["judge"] == []


def test_judge_answer_missing_answer_raises(fake_client):
    """last_answer=None：应抛 ValueError。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": None,
    }
    with pytest.raises(ValueError, match="缺少候选人回答"):
        judge_answer(state)


def test_judge_answer_empty_answer_raises(fake_client):
    """last_answer 为纯空白：应抛 ValueError。"""
    state = {
        "outline": _outline(1),
        "current_topic_index": 0,
        "current_question": _question(),
        "last_answer": "   ",
    }
    with pytest.raises(ValueError, match="缺少候选人回答"):
        judge_answer(state)
