"""简历/JD 解析结果缓存：按文本内容 SHA256 命中，跳过重复 LLM 解析。

缓存粒度：对 resume_text / jd_text 的纯文本算 hash，内容不变则复用上次解析结果。
存储方式：data/parse_cache/ 目录下每个 key 一个 JSON 文件，单用户场景无需并发控制。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from interview_agent.config import get_settings


def content_hash(text: str) -> str:
    """对纯文本算 SHA256，作为缓存唯一标识。内容一字节不同则 hash 完全不同。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_dir() -> Path:
    path = get_settings().parse_cache_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get(key: str) -> dict[str, Any] | None:
    """读取缓存，命中返回 dict，未命中或损坏返回 None。"""
    cache_file = _cache_dir() / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def set(key: str, value: Any) -> None:
    """写入缓存，value 需可被 json.dumps 序列化（Pydantic 模型用 model_dump(mode="json")）。"""
    cache_file = _cache_dir() / f"{key}.json"
    cache_file.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
