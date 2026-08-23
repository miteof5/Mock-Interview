"""LangGraph 状态定义：面试主流程的输入、中间状态与输出。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from interview_agent.config import Settings, get_settings
from interview_agent.models import (
    AnswerJudgment,
    DecisionResult,
    InterviewMessage,
    InterviewOutline,
    InterviewQuestion,
    InterviewReport,
    KnowledgeChunk,
    ParsedJD,
    ParsedResume,
)


def append_messages(
    left: list[InterviewMessage],
    right: list[InterviewMessage] | InterviewMessage,
) -> list[InterviewMessage]:
    """LangGraph Reducer：将新消息追加到历史对话末尾，支持单条或批量。"""
    if isinstance(right, list):
        return [*left, *right]
    return [*left, right]


def append_judgments(
    left: list[AnswerJudgment],
    right: list[AnswerJudgment] | AnswerJudgment,
) -> list[AnswerJudgment]:
    """LangGraph Reducer：将新评判追加到评判历史末尾，支持单条或批量。"""
    if isinstance(right, list):
        return [*left, *right]
    return [*left, right]


class _InterviewInputOptional(TypedDict, total=False):
    """InterviewInput 的可选项（3.10 无 NotRequired，用 total=False 子类表达可选字段）。"""

    difficulty: str  # 全局面试难度（简单/中等/困难），未传时由 build_initial_state 兜底为"中等"


class InterviewInput(_InterviewInputOptional):
    """面试启动时的外部输入：会话标识 + 简历原文 + JD 原文。"""

    session_id: str  # 本次面试唯一标识，用于持久化与恢复
    resume_text: str  # 简历原始文本（md/docx/pdf 转纯文本后）
    jd_text: str  # 岗位说明原始文本（md/docx/pdf 转纯文本后）


class InterviewState(TypedDict, total=False):
    """
    LangGraph 主状态。

    说明：
    - 使用 total=False 允许节点只回写部分字段；
    - 历史/评判列表字段带 Annotated reducer，便于节点只回写增量；
    - 边界控制字段（max_follow_ups / max_questions）从配置拷贝进状态，
      方便 decide_next 节点纯根据状态做边界判断。
    """

    # —— 启动入参（持久不变） ——
    session_id: str  # 会话唯一标识，与 Input 一致
    resume_text: str  # 简历原文，便于解析失败时回退或排查
    jd_text: str  # JD 原文，便于解析失败时回退或排查
    started_at: str  # 会话启动时间（UTC ISO 字符串），写入摘要时不变
    difficulty: str  # 全局面试难度（简单/中等/困难），贯穿 plan/ask/follow_up

    # —— parse_inputs 节点产出 ——
    resume: ParsedResume | None  # 结构化简历；解析失败仍可保持为 None（上层兜底）
    jd: ParsedJD | None  # 结构化 JD；解析失败仍可保持为 None（上层兜底）

    # —— plan_interview 节点产出 ——
    outline: InterviewOutline | None  # 面试提纲（有序主题列表）
    stage: Literal["parsing", "planning", "asking", "finished"]  # 当前大阶段

    # —— 循环中状态（检索→提问→评分→决策）——
    current_topic_id: str | None  # 正在进行的主题 id；None 表示尚未进入首题
    current_topic_index: int  # 正在进行的主题在 outline.topics 中的下标
    current_question: InterviewQuestion | None  # 刚刚向候选人提出的问题
    history: Annotated[list[InterviewMessage], append_messages]
    # 完整对话历史（含面试官提问+候选人回答），使用 reducer 追加
    last_answer: str | None  # 用户通过 interrupt 恢复后最新回答
    last_judgment: AnswerJudgment | None  # 最近一轮 judge_answer 的结果
    last_decision: DecisionResult | None  # 最近一轮 decide_next 的决策结果
    judgments: Annotated[list[AnswerJudgment], append_judgments]
    # 每轮评判的完整历史，使用 reducer 追加；聚合后用于生成最终报告
    retrieval_results: list[KnowledgeChunk]  # 当前主题检索到的知识库片段（随轮次刷新）

    # —— 边界控制计数器与阈值（由 build_initial_state 从配置写入）——
    follow_up_count: int  # 当前主题下已追问次数，命中 max_follow_ups 则强制切题
    question_count: int  # 已问出的总题数（含追问），命中 max_questions 则强制结束
    max_follow_ups: int  # 单主题最大追问次数阈值（源自 Settings）
    max_questions: int  # 整个面试最大题数阈值（源自 Settings）

    # —— generate_report 节点产出 ——
    report: InterviewReport | None  # 最终评估报告，finished 阶段由节点填入


class InterviewOutput(TypedDict):
    """面试结束时对外暴露的输出结构。"""

    report: InterviewReport  # 最终评估报告对象


def build_initial_state(
    input: InterviewInput,
    settings: Settings | None = None,
) -> InterviewState:
    """
    从外部输入构建初始 InterviewState。

    - 解析/提纲/对话/评分/报告等字段初始置空；
    - 阶段默认为 parsing（首节点 parse_inputs 的前置状态）；
    - max_follow_ups 与 max_questions 从 Settings 拷贝，避免节点跨层读配置；
    - difficulty 从 input 读取（未传默认"中等"），贯穿 plan/ask/follow_up 出题难度；
    - 可选入参 settings 用于测试时注入 mock 配置。
    """
    config = settings or get_settings()
    return {
        "session_id": input["session_id"],
        "resume_text": input["resume_text"],
        "jd_text": input["jd_text"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "difficulty": input.get("difficulty") or "中等",
        "resume": None,
        "jd": None,
        "outline": None,
        "stage": "parsing",
        "current_topic_id": None,
        "current_topic_index": 0,
        "current_question": None,
        "history": [],
        "last_answer": None,
        "last_judgment": None,
        "last_decision": None,
        "judgments": [],
        "retrieval_results": [],
        "follow_up_count": 0,
        "question_count": 0,
        "max_follow_ups": config.max_follow_ups,
        "max_questions": config.max_questions,
        "report": None,
    }
