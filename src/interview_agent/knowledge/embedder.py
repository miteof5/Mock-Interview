"""DashScope / OpenAI-compatible text embeddings（走项目统一的 base_url + OpenAI SDK）。

与 client.py 调用风格统一：
    两端都用 `openai` SDK 打兼容模式的 endpoints：
        - 聊天：`{base_url}/chat/completions`（见 llm/client.py）
        - 嵌入：`{base_url}/embeddings`（本文件）
    后续切供应商（OpenAI / DeepSeek / 本地 vLLM）只改 Settings.base_url + model 两个字符串，
    这里一行调用代码都不用动。

双塔参数（DashScope text-embedding-v4 特有，强烈建议保留）：
    请求 body 中加 `text_type: "document" / "query"`，召回时 query embedding
    才会和 document embedding 在同一个向量空间正确做点积。
    注意：v4 已废弃旧参数名 input_type，改用 text_type（扁平放在 body 顶层）。
    其它供应商若不识别该字段，会被 openai SDK 作为 extra_body 传入后被服务端忽略，无害。

对外双接口（与 LangChain / LlamaIndex Embedder 标准签名一致）：
    - embed_documents(list[str]) -> list[list[float]]
    - embed_query(str)          -> list[float]
"""

from __future__ import annotations

import warnings
from typing import Any

from openai import APIError, APIStatusError, APITimeoutError, OpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from interview_agent.config import Settings, get_settings

DEFAULT_BATCH_SIZE = 10
# DashScope compatible-mode embeddings 单批上限；超限会被 API 400 拒绝。
# text-embedding-v4 硬限制：单次请求最多 10 条。
_MAX_BATCH_SIZE = 10
# 超过这个字符数的单条文本，提前给 warning（v4 单条上限 8192 tokens ≈ 24000 中文，此处取保守值）
_SINGLE_TEXT_LEN_WARN = 6000

# ---------------------------------------------------------------------------
# 异常分类（与 llm/client.py 语义保持一致，但本地自包含 ——
#           避免"import embedder → 需要 client → client 顶层 import dashscope"的历史坑）
# ---------------------------------------------------------------------------
class DashScopeClientError(RuntimeError):
    """调用或结果解析失败的通用错误（不可重试语义）。"""


class DashScopeRetryableError(DashScopeClientError):
    """可重试错误（限流 / 服务端 5xx / 网络抖动 / 读超时）。"""


_RETRYABLE_EXCEPTIONS_BASE: tuple[type[BaseException], ...] = (
    APITimeoutError,
    ConnectionError,
    TimeoutError,
    OSError,
)


def _should_retry(exc: BaseException) -> bool:
    """tenacity retry 条件：基础重试类异常 或 自定义可重试异常。"""
    return isinstance(exc, DashScopeRetryableError) or isinstance(
        exc, _RETRYABLE_EXCEPTIONS_BASE
    )


def _translate_openai_error(exc: APIError) -> DashScopeClientError | DashScopeRetryableError:
    """把 openai SDK 的 APIError 翻译为统一异常，规则与 llm/client.py 一致。"""
    code = getattr(exc, "code", None)
    status_code: int | None = getattr(exc, "status_code", None)
    message = str(getattr(exc, "message", exc))[:200]
    body = f"code={code}, message={message}"

    if isinstance(exc, APIStatusError) and status_code is not None:
        if status_code == 429 or status_code >= 500:
            return DashScopeRetryableError(f"HTTP {status_code}: {body}")
        # 400/401/403/404 等 4xx：重试不会变好
        return DashScopeClientError(f"HTTP {status_code}: {body}")

    if isinstance(exc, APITimeoutError):
        return DashScopeRetryableError(f"Timeout: {body}")

    if type(exc).__name__.endswith("ConnectionError"):
        return DashScopeRetryableError(f"Connection: {body}")
    return DashScopeClientError(f"APIError: {body}")


def _extract_embeddings(
    data_list: list[Any], expected_count: int
) -> list[list[float]]:
    """从 openai SDK 返回的 List[Embedding] 对象列表提取二维向量。"""
    if len(data_list) != expected_count:
        raise DashScopeClientError(
            f"Embeddings endpoint returned {len(data_list)} vectors, expected {expected_count}"
        )
    try:
        # 按 index 排序，保证与输入顺序一致（有些代理/兼容模式会重排）
        ordered = sorted(
            data_list,
            key=lambda item: int(getattr(item, "index", 0)),
        )
        return [list(item.embedding) for item in ordered]
    except (AttributeError, TypeError, ValueError) as exc:
        raise DashScopeClientError(
            "Embedding data item missing the numeric 'embedding' field"
        ) from exc


class DashScopeEmbedder:
    """Embed documents or queries through the project's single base_url。

    所有对外参数保持与初始设计一致，新增能力：
        - input_type=document vs query：双塔区分，显著提升检索精度（DashScope 特有）；
        - 批处理上下限（1 <= batch_size <= 24）；
        - 构造期早失败（空 api_key / base_url 非法）；
        - 单条超长文本 warning；
        - 错误分级（4xx 不重试，429/5xx 走 tenacity 指数退避）。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.settings = settings or get_settings()

        # 构造期早失败
        if not self.settings.api_key:
            raise DashScopeClientError(
                "DashScopeEmbedder: api_key 为空，请检查环境变量"
            )
        base = (self.settings.base_url or "").strip()
        if not base.startswith(("http://", "https://")):
            raise DashScopeClientError(
                f"DashScopeEmbedder: base_url 非法: {base!r}"
            )

        # OpenAI SDK：显式传 key + url，避免 SDK 默认读你电脑上的真·OPENAI_* 环境变量
        self._client = OpenAI(
            api_key=self.settings.api_key,
            base_url=base,
            # 嵌入通常比聊天快，但保守给 10s connect + 60s read
            timeout=(10.0, 60.0),
        )

        # batch_size 上下限
        self.batch_size = max(1, min(int(batch_size), _MAX_BATCH_SIZE))

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    def embed_documents(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        text_type: str = "document",
    ) -> list[list[float]]:
        """Embed a list of chunk texts，保持输入顺序；空列表直接返回。"""
        if not texts:
            return []
        self._warn_long_texts(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(
                self._embed_batch(
                    model or self.settings.embedding_model, batch, text_type
                )
            )
        return vectors

    def embed_query(
        self,
        text: str,
        *,
        model: str | None = None,
    ) -> list[float]:
        """Embed 单条检索 query；走双塔 query 侧的 input_type。"""
        if not isinstance(text, str) or not text.strip():
            raise DashScopeClientError("embedding query text must not be empty")
        self._warn_long_texts([text])
        vectors = self._embed_batch(
            model or self.settings.embedding_model, [text], "query"
        )
        return vectors[0]

    # ------------------------------------------------------------------
    # 内部：批调用 + 重试 + 异常翻译
    # ------------------------------------------------------------------
    def _embed_batch(
        self, model: str, texts: list[str], text_type: str
    ) -> list[list[float]]:
        try:
            return self._call(model, texts, text_type)
        except (DashScopeClientError, DashScopeRetryableError):
            raise
        except APIError as exc:
            raise _translate_openai_error(exc) from exc
        except Exception as exc:
            raise DashScopeClientError(
                f"Embedding call failed unexpectedly: {exc}"
            ) from exc

    @retry(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call(self, model: str, texts: list[str], text_type: str) -> list[list[float]]:
        """真实调用 OpenAI SDK embeddings.create；带 tenacity 重试。"""
        resp = self._client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
            # v4 双塔参数：text_type 区分 document/query，扁平放在 body 顶层（通过 extra_body 传入）
            extra_body={"text_type": text_type},
        )
        return _extract_embeddings(resp.data, len(texts))

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _warn_long_texts(texts: list[str]) -> None:
        """单条超长文本 warning，代码块 >1500 tokens 时在 API 400 之前提醒用户。"""
        offenders = [i for i, t in enumerate(texts) if isinstance(t, str) and len(t) > _SINGLE_TEXT_LEN_WARN]
        if offenders:
            warnings.warn(
                f"{len(offenders)} 条待 embedding 文本长度 > {_SINGLE_TEXT_LEN_WARN} 字符"
                f"（索引 {offenders[:5]}{'…' if len(offenders) > 5 else ''}）。"
                "DashScope embedding 模型单条有 token 上限，超长度会被 API 400 拒绝。"
                "若为代码块，请在 chunker 侧评估是否需要手动拆分。",
                stacklevel=3,
            )

    def close(self) -> None:
        """释放底层 OpenAI SDK 连接池；小脚本用完即丢时可省略。"""
        self._client.close()


__all__ = ["DEFAULT_BATCH_SIZE", "DashScopeEmbedder", "DashScopeClientError", "DashScopeRetryableError"]
