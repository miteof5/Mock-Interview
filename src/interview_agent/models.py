"""领域模型：结构化输入、面试提纲、问答、评分与报告。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class EducationItem(BaseModel):
    """简历中的单条教育经历条目。"""

    school: str | None = None  # 院校名称
    degree: str | None = None  # 学位（如本科、硕士）
    major: str | None = None  # 所学专业


class ExperienceItem(BaseModel):
    """简历中的单条工作经历条目。"""

    company: str | None = None  # 公司名
    role: str | None = None  # 担任职位
    duration: str | None = None  # 在职时段（自由文本）
    highlights: list[str] = Field(default_factory=list)  # 关键业绩/职责亮点


class ParsedResume(BaseModel):
    """解析后的结构化简历对象，由 parse_inputs 节点产出。"""

    raw_text: str  # 简历原始文本，便于追溯与回显
    name: str | None = None  # 候选人姓名
    target_position: str | None = None  # 意向岗位（可能来自简历或显式传入）
    years_of_experience: float | None = None  # 工作年限
    skills: list[str] = Field(default_factory=list)  # 技能列表
    work_experience: list[ExperienceItem] = Field(default_factory=list)  # 工作经历
    education: list[EducationItem] = Field(default_factory=list)  # 教育经历


class ParsedJD(BaseModel):
    """解析后的结构化 JD（岗位说明）对象，由 parse_inputs 节点产出。"""

    raw_text: str  # JD 原始文本
    title: str | None = None  # 岗位名称
    company: str | None = None  # 招聘公司
    responsibilities: list[str] = Field(default_factory=list)  # 岗位职责
    requirements: list[str] = Field(default_factory=list)  # 任职要求
    must_have: list[str] = Field(default_factory=list)  # 硬性必备条件
    nice_to_have: list[str] = Field(default_factory=list)  # 加分项
    keywords: list[str] = Field(default_factory=list)  # 关键词（用于检索/匹配）


class InterviewTopic(BaseModel):
    """面试提纲中的单个主题，由 plan_interview 节点产出。"""

    id: str  # 主题唯一标识，用于关联提问与评分
    title: str  # 主题标题
    focus: str  # 该主题重点考察方向
    # 新版 prompt 增加：单主题难度档位（基础 / 中等 / 进阶），由 LLM 根据 candidate_basis 自适应推导
    difficulty: str = "中等"
    # 新版 prompt 增加：候选人在该主题方向上的实际经历证据等级，供出题端匹配深度
    candidate_basis: str = "仅JD要求"
    must_ask: bool = False  # 是否必问（覆盖 JD 核心要求时使用）


class InterviewOutline(BaseModel):
    """面试提纲：主题的有序集合。"""

    topics: list[InterviewTopic] = Field(default_factory=list)  # 主题列表


class InterviewMessage(BaseModel):
    """面试对话中的单条消息（面试官或候选人）。"""

    role: Literal["interviewer", "candidate"]  # 发言角色
    content: str  # 消息文本内容
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )  # 创建时间，默认当前 UTC 时间


class InterviewQuestion(BaseModel):
    """面试官生成的一道问题。"""

    topic_id: str  # 所属主题 id
    content: str  # 问题文本
    question_type: Literal["open", "technical", "behavioral", "follow_up"] = "open"
    # 问题类型：开放 / 技术 / 行为 / 追问


class KnowledgeChunk(BaseModel):
    """检索得到的一条知识库片段。"""

    content: str  # 片段正文
    source: str = ""  # 来源标识（如文档名/标题路径）
    score: float | None = None  # 检索相关性分数


class ScoreItem(BaseModel):
    """单维度评分项。"""

    dimension: str  # 维度名称（如专业性、表达结构）
    score: float = Field(ge=0, le=10)  # 该维度得分，范围 0-10
    # 允许 None：LLM 偶尔显式返回 null，会先让 Pydantic 严格模式直接拒绝解析；
    # 这里放宽到 str | None，由 normalize_score_items 兜底清洗为空串，
    # 让"代码侧兜底"逻辑（rubric.py）能真正生效，而不是被解析阶段提前拦截
    evidence: str | None = ""  # 评分依据（引用回答或知识库）
    suggestion: str | None = ""  # 针对该维度的改进建议


class AnswerJudgment(BaseModel):
    """对候选人单次回答的评判结果，由 judge_answer 节点产出。"""

    topic_id: str  # 关联的主题 id
    question: str  # 本轮问题文本
    answer: str  # 候选人回答文本
    overall_score: float = Field(ge=0, le=10)  # 本轮综合得分，范围 0-10
    scores: list[ScoreItem] = Field(default_factory=list)  # 各维度评分
    follow_up_suggestion: str = ""  # 是否/如何追问的建议
    summary: str = ""  # 本轮评判小结


# 决策动作类型：追问 / 切换下一主题 / 结束面试
NextAction = Literal["follow_up", "next_topic", "end"]


class DecisionResult(BaseModel):
    """decide_next 节点的结构化决策结果。"""

    action: NextAction  # 下一步动作
    reason: str = ""  # 决策理由（供调试与可解释性）


class ImprovementSuggestion(BaseModel):
    """最终报告中的单条改进建议。"""

    priority: Literal["high", "medium", "low"]  # 优先级
    dimension: str  # 关联维度
    suggestion: str  # 建议正文
    # 同 ScoreItem：放宽到 str | None，避免 LLM 显式 null 时 Pydantic 解析阶段直接抛错；
    # 兜底在 generate_report 方法里逐项 or "" 清洗
    evidence: str | None = ""  # 建议依据


class InterviewReport(BaseModel):
    """面试结束后的最终评估报告，由 generate_report 节点产出。"""

    session_id: str  # 会话唯一标识
    overall_score: float = Field(ge=0, le=10)  # 总评分，范围 0-10
    dimension_scores: list[ScoreItem] = Field(default_factory=list)  # 各维度汇总得分
    summary: str = ""  # 总体评语
    suggestions: list[ImprovementSuggestion] = Field(default_factory=list)
    # 按优先级排序的改进建议
    rounds: list[AnswerJudgment] = Field(default_factory=list)  # 各轮问答评判明细
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )  # 报告生成时间，默认当前 UTC 时间
