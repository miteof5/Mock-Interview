"""Web API 的请求与响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionCreated(BaseModel):
    session_id: str
    status: str


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=20_000)


class CandidatePayload(BaseModel):
    name: str = "候选人"
    target_position: str = "待定"
    company: str = "待定"


class QuestionPayload(BaseModel):
    question_index: int
    total_questions: int
    topic_index: int
    topic_id: str
    topic: str
    topics: list[dict[str, Any]] = Field(default_factory=list)
    question_type: str = "open"
    question: str
    candidate: CandidatePayload


class SessionStatus(BaseModel):
    session_id: str
    status: str
    error: str | None = None
    question: QuestionPayload | None = None
    report: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_session(cls, session: Any) -> "SessionStatus":
        return cls(
            session_id=session.session_id,
            status=session.status,
            error=session.error,
            question=session.question,
            report=session.report,
            history=session.history,
        )
