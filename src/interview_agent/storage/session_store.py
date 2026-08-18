"""会话 JSON 存储：完整流水 + 轻量摘要，写入 data/sessions/{session_id}/。

分层职责（与 checkpoint.py 明确分工）：
- raw_history.json：将整个 InterviewState（含 Pydantic 嵌套）递归序列化为纯 JSON dict，
  用于调试回溯与离线分析。
- session.json：从 state 中提取人类可读摘要（姓名/岗位/公司/阶段/总题数/启动时间/总分
  等），供 CLI `interview list` 快速展示，不需要解析大体积 raw_history。
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from interview_agent.config import PROJECT_ROOT
from interview_agent.state import InterviewState  # 仅类型标注

SESSIONS_ROOT = PROJECT_ROOT / "data" / "sessions"
RAW_HISTORY_FILE = "raw_history.json"
SESSION_SUMMARY_FILE = "session.json"

_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]+$")


# ---------------------------------------------------------------------------
# 通用 helpers：Pydantic / dict / None 兼容取字段 + 递归序列化
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """从 Pydantic 或 dict 中取值，两者都不是或字段不存在时返回 default。"""
    if obj is None:
        return default
    if isinstance(obj, BaseModel):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _to_serializable(value: Any) -> Any:
    """将嵌套的 Pydantic / datetime / dict / list 递归转成 JSON 可序列化结构。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _to_serializable(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_serializable(v) for v in value]
    raise TypeError(f"无法序列化为 JSON: {type(value).__name__}")


def _json_default(value: Any) -> Any:
    """兜底：_to_serializable 漏网时再用一次 isoformat；其余抛错避免静默写坏。"""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"无法序列化为 JSON: {type(value).__name__}")


# ---------------------------------------------------------------------------
# 路径校验 + 原子写入
# ---------------------------------------------------------------------------
def _session_dir(session_id: str, sessions_root: Path | None) -> Path:
    """校验 session_id 并解析会话目录，防止路径穿越。"""
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        raise ValueError(f"非法的 session_id: {session_id!r}")
    root = (sessions_root or SESSIONS_ROOT).resolve()
    directory = (root / session_id).resolve()
    if directory.parent != root:
        raise ValueError(f"session_id 超出会话根目录: {session_id!r}")
    return directory


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """先写带 PID+随机后缀的临时文件，再 replace 替换，避免崩溃留半截或多进程冲突。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{secrets.token_hex(4)}.tmp"
    tmp_path = path.with_suffix(path.suffix + suffix)
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# 摘要提取：从 InterviewState 挖嵌套字段
# ---------------------------------------------------------------------------
def _extract_summary(state: InterviewState | dict[str, Any]) -> dict[str, Any]:
    """从状态中提取 CLI list 展示所需的摘要字段；自动兼容 Pydantic / dict 两种形态。"""
    resume = _get(state, "resume")
    jd = _get(state, "jd")
    report = _get(state, "report")
    history = _get(state, "history") or []

    # 取 history 首条时间当兜底 started_at（老版本 state 可能没 started_at）
    first_msg_created = None
    if history and isinstance(history, list):
        first_msg = history[0]
        created = _get(first_msg, "created_at")
        if isinstance(created, datetime):
            first_msg_created = created.isoformat()
        elif isinstance(created, str):
            first_msg_created = created

    started_at = _get(state, "started_at") or first_msg_created or datetime.now(timezone.utc).isoformat()

    summary = {
        "session_id": _get(state, "session_id"),
        "candidate_name": _get(resume, "name"),
        "target_position": _get(resume, "target_position"),
        "company": _get(jd, "company"),
        "started_at": started_at,
        "stage": _get(state, "stage"),
        "overall_score": _get(report, "overall_score"),
        "question_count": _get(state, "question_count", 0),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


# ---------------------------------------------------------------------------
# 对外 API：save / load / list / delete
# ---------------------------------------------------------------------------
def save_session(
    state: InterviewState | dict[str, Any],
    sessions_root: Path | None = None,
) -> Path:
    """保存当前面试状态（InterviewState 或等价 dict），返回会话目录。

    - 从 state["session_id"] 自动取会话 ID，不用单独再传；
    - raw_history.json：完整 state 序列化（含 Pydantic 嵌套 -> 纯 dict）；
    - session.json：从 state 中挖取人类可读摘要。
    """
    # 允许传 dict 形式的 state，但 session_id 必须有
    session_id = _get(state, "session_id")
    if not session_id:
        raise ValueError("save_session 传入的 state 缺少 session_id")

    directory = _session_dir(str(session_id), sessions_root)
    serializable_state = _to_serializable(dict(state) if isinstance(state, dict) else state)
    _atomic_write(directory / RAW_HISTORY_FILE, serializable_state)

    summary = _extract_summary(state)
    _atomic_write(directory / SESSION_SUMMARY_FILE, summary)
    return directory


def load_session(
    session_id: str,
    sessions_root: Path | None = None,
) -> dict[str, Any]:
    """读取 raw_history.json 完整状态（纯 dict 形式）；文件不存在时抛 FileNotFoundError。"""
    path = _session_dir(session_id, sessions_root) / RAW_HISTORY_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def load_session_summary(
    session_id: str,
    sessions_root: Path | None = None,
) -> dict[str, Any]:
    """读取 session.json 摘要；文件不存在时抛 FileNotFoundError。"""
    path = _session_dir(session_id, sessions_root) / SESSION_SUMMARY_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions(
    sessions_root: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """列出所有会话摘要，按 saved_at 倒序（最新在前）。

    - 子目录里没有 session.json 的跳过（未完成首次保存或损坏会话）；
    - JSON 解码失败的跳过，不影响其它会话展示；
    - limit 用于 CLI 最近 N 条展示，None 表示全部。
    """
    root = (sessions_root or SESSIONS_ROOT).resolve()
    if not root.is_dir():
        return []

    all_summaries: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        summary_path = entry / SESSION_SUMMARY_FILE
        if not summary_path.is_file():
            continue
        try:
            all_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue

    # 按 saved_at 倒序；缺失的放到最后
    all_summaries.sort(
        key=lambda s: (s.get("saved_at") or "",),
        reverse=True,
    )
    if limit is not None:
        all_summaries = all_summaries[:limit]
    return all_summaries


def delete_session(
    session_id: str,
    sessions_root: Path | None = None,
) -> None:
    """删除指定会话的整个目录（含流水+摘要）；目录不存在时静默成功。"""
    directory = _session_dir(session_id, sessions_root)
    if directory.is_dir():
        shutil.rmtree(directory)
