"""pytest 全局 fixture。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """测试环境默认注入假 API Key，避免 Settings 校验在未配 Key 时炸掉。

    哑跑/单元测试不触达网络，假 Key 仅用于通过 config._require_api_key 校验；
    真正调用 LLM 的集成测试可自行覆盖环境变量。
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-for-pytest")
    # 清除 get_settings 的 lru_cache，确保本次测试读到新的环境变量
    from interview_agent.config import get_settings

    get_settings.cache_clear()
