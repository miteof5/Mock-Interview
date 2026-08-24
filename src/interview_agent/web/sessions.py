"""Web 会话管理：LangGraph 运行器、事件队列与会话恢复。"""

from __future__ import annotations

import asyncio
import secrets
import threading
from datetime import datetime, timezone
from typing import Any

from langgraph.types import Command

_NODE_STATUS = {
    "parse_inputs": "analyzing",
    "plan_interview": "planning",
    "retrieve_knowledge": "preparing_knowledge",
    "ask_question": "asking",
    "judge_answer": "judging",
    "generate_report": "reporting",
}

# 初始流程节点完成后，下一个正在执行的节点状态（解决前端状态滞后一个节点的问题）
_NEXT_STATUS = {
    "parse_inputs": "planning",
    "plan_interview": "preparing_knowledge",
    "retrieve_knowledge": "asking",
}


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def build_question_payload(values: dict[str, Any]) -> dict[str, Any]:
    """从 LangGraph state 提取前端渲染问题页所需的全部信息。"""
    outline = values.get("outline")
    raw_topics = _attr(outline, "topics", []) or []
    topics = [
        {
            "id": _attr(topic, "id"),
            "title": _attr(topic, "title"),
            "focus": _attr(topic, "focus", ""),
        }
        for topic in raw_topics
    ]

    topic_index = int(values.get("current_topic_index", 0) or 0)
    topic = topics[topic_index] if topics and topic_index < len(topics) else {}
    question = values.get("current_question")
    resume = values.get("resume")
    jd = values.get("jd")

    candidate = {
        "name": _attr(resume, "name") or "候选人",
        "target_position": _attr(resume, "target_position") or _attr(jd, "title") or "待定",
        "company": _attr(jd, "company") or "待定",
    }

    return {
        "question_index": int(values.get("question_count", 1) or 1),
        "total_questions": int(values.get("max_questions", 15) or 15),
        "topic_index": topic_index,
        "topic_id": _attr(question, "topic_id") or topic.get("id", ""),
        "topic": topic.get("title") or "面试提问",
        "topics": topics,
        "question_type": _attr(question, "question_type") or "open",
        "question": _attr(question, "content") or "",
        "candidate": candidate,
    }


def build_history(values: dict[str, Any]) -> list[dict[str, Any]]:
    history = values.get("history") or []
    return [
        {"role": _attr(msg, "role"), "content": _attr(msg, "content")}
        for msg in history
    ]


class Session:
    """单个面试会话：事件队列、执行状态和 LangGraph 运行任务。"""

    def __init__(
        self,
        session_id: str,
        resume_name: str,
        jd_name: str,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.resume_name = resume_name
        self.jd_name = jd_name
        self.initial_state = initial_state
        self.loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.status = "created"
        self.error: str | None = None
        self.question: dict[str, Any] | None = None
        self.report: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._started = False

    def publish_from_thread(self, payload: dict[str, Any]) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)

    async def publish(self, payload: dict[str, Any]) -> None:
        await self.queue.put(payload)

    async def start(self, manager: "SessionManager") -> None:
        if self._started:
            return
        self._started = True
        self.status = "processing"
        await self.publish({"type": "status", "data": {"status": "analyzing"}})
        self._task = asyncio.create_task(self._run_initial(manager))

    async def resume(self, manager: "SessionManager", answer: str) -> None:
        self.status = "processing"
        await self.publish({"type": "status", "data": {"status": "judging"}})
        self._task = asyncio.create_task(self._run_resume(manager, answer))

    async def _run_initial(self, manager: "SessionManager") -> None:
        try:
            snapshot = await asyncio.to_thread(manager.run_initial, self)
            await self.apply_snapshot(snapshot)
        except Exception as exc:  # noqa: BLE001 - 统一转成 SSE error 事件
            await self._fail_initial(exc)

    async def _run_resume(self, manager: "SessionManager", answer: str) -> None:
        try:
            snapshot = await asyncio.to_thread(manager.run_resume, self, answer)
            await self.apply_snapshot(snapshot)
        except Exception as exc:  # noqa: BLE001 - 保留等待状态，允许用户重试
            self.status = "waiting"
            self.error = str(exc)
            await self.publish({"type": "error", "data": {"message": str(exc)}})

    async def apply_snapshot(self, snapshot: Any) -> None:
        values = getattr(snapshot, "values", None) or {}
        self.history = build_history(values)
        if getattr(snapshot, "next", None):
            self.status = "waiting"
            self.question = build_question_payload(values)
            self.report = None
            self.error = None
            await self.publish({"type": "question", "data": self.question})
        else:
            self.status = "finished"
            self.report = _dump(values.get("report"))
            self.question = None
            self.error = None
            await self.publish({"type": "report", "data": self.report})

    async def emit_snapshot(self) -> None:
        if self.status == "waiting" and self.question:
            await self.publish({"type": "question", "data": self.question})
        elif self.status == "finished" and self.report:
            await self.publish({"type": "report", "data": self.report})
        elif self.status == "failed":
            await self.publish(
                {"type": "error", "data": {"message": self.error or "面试处理失败"}}
            )

    async def _fail_initial(self, exc: Exception) -> None:
        self.status = "failed"
        self.error = str(exc)
        await self.publish({"type": "error", "data": {"message": str(exc)}})

    def close(self) -> None:
        self.status = "ended"
        if self._task is not None and not self._task.done():
            self._task.cancel()


class SessionManager:
    """维护内存中的会话注册表，并用全局锁串行执行 LangGraph。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._graph_lock = threading.Lock()

    def _config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def create(
        self,
        resume_name: str,
        jd_name: str,
        initial_state: dict[str, Any],
    ) -> Session:
        session_id = (
            f"web-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        )
        initial_state = dict(initial_state)
        initial_state["session_id"] = session_id
        session = Session(session_id, resume_name, jd_name, initial_state)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_restore(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                return session
            snapshot = self._get_snapshot(session_id)
            if snapshot is None:
                return None
            values = getattr(snapshot, "values", None) or {}
            if not values or values.get("session_id") != session_id:
                return None
            session = Session(session_id, "已恢复", "已恢复")
            session.history = build_history(values)
            if getattr(snapshot, "next", None):
                session.status = "waiting"
                session.question = build_question_payload(values)
            else:
                session.status = "finished"
                session.report = _dump(values.get("report"))
            self._sessions[session_id] = session
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            if session.status == "processing":
                return False
            self._sessions.pop(session_id, None)
        session.close()
        return True

    def _get_snapshot(self, session_id: str) -> Any | None:
        try:
            return self._app.get_state(self._config(session_id))
        except Exception:  # noqa: BLE001 - 会话不存在时按未找到处理
            return None

    def run_initial(self, session: Session) -> Any:
        config = self._config(session.session_id)
        with self._graph_lock:
            self._stream(session, initial=session.initial_state, config=config)
            return self._app.get_state(config)

    def run_resume(self, session: Session, answer: str) -> Any:
        config = self._config(session.session_id)
        with self._graph_lock:
            self._stream(session, resume=answer, config=config)
            return self._app.get_state(config)

    def _stream(
        self,
        session: Session,
        *,
        initial: dict[str, Any] | None = None,
        resume: str | None = None,
        config: dict[str, Any],
    ) -> None:
        if initial is not None:
            chunks = self._app.stream(initial, config=config, stream_mode="updates")
        else:
            chunks = self._app.stream(
                Command(resume=resume), config=config, stream_mode="updates"
            )
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            for node_name in chunk:
                # 优先发送下一个节点的状态（当前节点已完成，正在执行下一个）
                status = _NEXT_STATUS.get(node_name) or _NODE_STATUS.get(node_name)
                if status:
                    session.publish_from_thread(
                        {"type": "status", "data": {"status": status}}
                    )
