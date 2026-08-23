"""FastAPI 应用：会话接口、SSE 事件流与前端静态托管。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from interview_agent.config import PROJECT_ROOT
from interview_agent.graph import get_app
from interview_agent.state import build_initial_state
from interview_agent.web.parsers import parse_upload
from interview_agent.web.schemas import AnswerRequest, SessionCreated, SessionStatus
from interview_agent.web.sessions import SessionManager

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def create_app(
    manager: SessionManager | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Mock Interview API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    manager = manager or SessionManager(get_app())
    app.state.manager = manager

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/sessions", response_model=SessionCreated, status_code=202)
    async def create_session(
        resume_file: UploadFile = File(...),
        jd_file: UploadFile = File(...),
        difficulty: str = Form("中等"),
    ) -> SessionCreated:
        for field_name, upload in (("resume_file", resume_file), ("jd_file", jd_file)):
            if not upload.filename:
                raise HTTPException(status_code=400, detail=f"{field_name} 文件名为空")

        resume_bytes = await resume_file.read()
        jd_bytes = await jd_file.read()
        if len(resume_bytes) > MAX_UPLOAD_BYTES or len(jd_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件大小不能超过 5MB")

        try:
            resume_text = parse_upload(resume_file.filename, resume_bytes)
            jd_text = parse_upload(jd_file.filename, jd_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not resume_text.strip() or not jd_text.strip():
            raise HTTPException(status_code=400, detail="简历和 JD 不能为空")

        initial_state = build_initial_state(
            {
                "session_id": "",
                "resume_text": resume_text,
                "jd_text": jd_text,
                "difficulty": difficulty,
            }
        )
        session = manager.create(resume_file.filename, jd_file.filename, initial_state)
        return SessionCreated(session_id=session.session_id, status=session.status)

    @app.get("/api/sessions/{session_id}", response_model=SessionStatus)
    async def get_session(session_id: str) -> SessionStatus:
        session = manager.get_or_restore(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionStatus.from_session(session)

    @app.post("/api/sessions/{session_id}/answer", status_code=202)
    async def submit_answer(session_id: str, body: AnswerRequest) -> dict[str, str]:
        session = manager.get_or_restore(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        async with session.lock:
            if session.status != "waiting":
                raise HTTPException(
                    status_code=409, detail="当前面试不在等待回答状态"
                )
            await session.resume(manager, body.answer)
        return {"status": "processing"}

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> None:
        if not manager.delete(session_id):
            raise HTTPException(status_code=404, detail="会话不存在或正在处理中")

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(session_id: str) -> StreamingResponse:
        session = manager.get_or_restore(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")

        async def event_generator():
            if session.status == "created":
                await session.start(manager)
            else:
                await session.emit_snapshot()
            yield "retry: 3000\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(session.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if session.status == "ended":
                    break
                event_type = payload.get("type", "message")
                data = json.dumps(payload.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if static_dir is None:
        static_dir = PROJECT_ROOT / "frontend"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


app = create_app()
