"""项目配置：环境变量优先，默认值集中在 pydantic-settings 中。"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chroma 中存放面试知识库的 collection 名称（读写两侧约定同一个常量，
# 避免 ingest.py 和 retriever.py 各自写死导致漂移 — 两文件都从 config 独立 import，零互相依赖）
KB_COLLECTION_NAME = "interview_kb"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # 允许通过系统环境变量 DASHSCOPE_API_KEY（DashScope 官方约定的键名）提供
        # 与字段名 API_KEY / 旧字段 DASHSCOPE_API_KEY 三向兼容，
        # 解决"系统环境变量已设置但代码读不到"的常见坑。
        case_sensitive=False,
    )

    api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "api_key", "DASHSCOPE_API_KEY", "dashscope_api_key"
        ),
        description="阿里百炼 API Key；留空表示由系统环境变量 DASHSCOPE_API_KEY / API_KEY 提供",
    )
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices(
            "base_url", "DASHSCOPE_BASE_URL", "dashscope_base_url"
        ),
        description="整个项目统一的 OpenAI 兼容模式 base_url；聊天/嵌入都走这一条",
    )

    kb_path: Path = PROJECT_ROOT / "data" / "kb"
    kb_vector_path: Path = PROJECT_ROOT / "data" / "vectorstore"
    db_path: Path = PROJECT_ROOT / "data" / "sessions" / "interview.sqlite"

    interviewer_model: str = "qwen-plus"
    judge_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"

    retrieval_top_k: int = Field(default=4, ge=1)
    max_follow_ups: int = Field(default=2, ge=0)
    max_questions: int = Field(default=15, ge=1)

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        for name in ("kb_path", "kb_vector_path", "db_path"):
            path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, PROJECT_ROOT / path)
        return self

    @model_validator(mode="after")
    def _require_api_key(self) -> "Settings":
        """早失败：Settings 实例化时立刻检查 api_key，不要等到第一次调网络才炸。"""
        if not self.api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY / API_KEY 未配置：请设置系统环境变量，"
                "或在项目根目录的 .env 中填写"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = [
    "KB_COLLECTION_NAME",
    "PROJECT_ROOT",
    "Settings",
    "get_settings",
]
