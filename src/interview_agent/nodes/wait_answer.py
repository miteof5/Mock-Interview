"""wait_answer 节点：interrupt 暂停等待候选人回答。"""

from __future__ import annotations

from langgraph.types import interrupt

from interview_agent.models import InterviewMessage
from interview_agent.state import InterviewState


def wait_answer(state: InterviewState) -> dict:
    """向调用方展示当前问题并暂停，恢复后把回答写入 state。"""
    question = state.get("current_question")
    answer: str = interrupt(
        {
            "question": question.content if question is not None else "",
        }
    )
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("候选人回答不能为空")
    answer = answer.strip()
    return {
        "last_answer": answer,
        "history": [InterviewMessage(role="candidate", content=answer)],
    }
