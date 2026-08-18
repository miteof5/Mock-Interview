"""百炼 DashScope 客户端（统一走 OpenAI-compatible 协议 + 全局 Settings.base_url）。

与项目配置的设计哲学对齐（整个项目只有一个 base_url + api_key）：
    聊天客户端与嵌入端一致，都通过 OpenAI 协议打 `/chat/completions`，
    不再顶层 `import dashscope`，彻底消除「dashscope 包未安装就 import 失败」的传导性依赖。

对外语义化方法（节点直接调用，无需自己拼 prompt/schema）
--------------------------------------------------------
parse_resume      简历原文 → ParsedResume
parse_jd          JD 原文 → ParsedJD
plan_interview    JD 原文 → InterviewOutline
ask_question      简历+JD+主题+知识+对话 → InterviewQuestion（首问/换题）
follow_up         主题+原问题+回答 → InterviewQuestion（question_type 固定 follow_up）
judge             主题+问题+回答+检索结果 → AnswerJudgment（评分，temperature=0.2）
generate_report   会话ID+评判历史 → InterviewReport（最终报告）

底层通用方法
------------
chat              任意 user/system prompt + Pydantic schema → 解析后的模型对象
_parse_json_output 用「response_format 强约束 + 兜底手工 parse」双保险解析
_generate         包一层异常翻译：openai.API* → DashScopeClient*Error（外层统一捕获）
_call             带 tenacity 重试：网络抖动/限流/服务端 3 次指数退避
format_knowledge  把检索结果拼成 [1] xxx（来源：yy）格式；空返回"(无参考知识)"
"""

from __future__ import annotations

from typing import Any, TypeVar

from openai import APIError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from interview_agent.config import Settings, get_settings
from interview_agent.evaluation.rubric import (
    RUBRIC_DIMENSION_NAMES,
    normalize_score_items,
)
from interview_agent.llm.prompts import (
    ASK_QUESTION_PROMPT,
    FOLLOW_UP_PROMPT,
    GENERATE_REPORT_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    PARSE_JD_PROMPT,
    PARSE_RESUME_PROMPT,
    PLAN_INTERVIEW_PROMPT,
)
from interview_agent.models import (
    AnswerJudgment,
    ImprovementSuggestion,
    InterviewOutline,
    InterviewQuestion,
    InterviewReport,
    KnowledgeChunk,
    ParsedJD,
    ParsedResume,
)

T = TypeVar("T", bound=BaseModel)

# 向后兼容别名：之前代码 / 冒烟测试里可能 import llm.client.RUBRIC_DIMENSIONS，
# 这里保留一个指向共享常量的引用，避免全局 grep 改名时漏掉。
RUBRIC_DIMENSIONS = RUBRIC_DIMENSION_NAMES

# 可重试的 openai 原生异常类 + 我们自定义的重试异常：
#   APITimeoutError = 读/连接超时
#   APIStatusError.status_code 429 / 5xx = 限流/服务端异常（在 _translate_openai_error 再细分）
#   ConnectionError / TimeoutError / OSError = 底层网络（tenacity 常见兜底）
_RETRYABLE_OPENAI_BASE: tuple[type[BaseException], ...] = (
    APITimeoutError,
    ConnectionError,
    TimeoutError,
    OSError,
)


# ---------------------------------------------------------------------------
# 异常分类（与 embedder 语义保持一致：ClientError 不重试 / Retryable 要重试）
# ---------------------------------------------------------------------------
class DashScopeClientError(RuntimeError):
    """调用或结果解析失败的通用错误（不可重试语义）。"""


class DashScopeRetryableError(DashScopeClientError):
    """可重试错误（限流 / 服务端 5xx / 网络抖动 / 读超时）。

    被 tenacity 装饰器识别并重试；最后一次失败后仍会原样抛出，
    外层可以根据该类型区分"连续重试仍失败"与"参数/鉴权等立刻失败"。
    """


def _should_retry(exc: BaseException) -> bool:
    """tenacity retry 条件：可重试基础类 或 自定义 DashScopeRetryableError。"""
    return isinstance(exc, DashScopeRetryableError) or isinstance(
        exc, _RETRYABLE_OPENAI_BASE
    )


def _translate_openai_error(exc: APIError) -> DashScopeClientError | DashScopeRetryableError:
    """把 openai SDK 的 APIError 子类型翻译为我们统一的 DashScopeClient*Error。

    为什么不直接把 openai.APIError 透出给上层？
        1. 上层的 except 分支只认识 DashScopeClientError / Retryable 两种；
        2. 当嵌入端未来也切回 openai SDK（而不是 httpx）时，两端异常类完全一致，
           排障心智模型相同；
        3. 脱敏：不直接回显完整响应体（可能包含 prompt 全文），只摘 code + message 前 200 字符。
    """
    code = getattr(exc, "code", None)
    status_code: int | None = getattr(exc, "status_code", None)
    message = str(getattr(exc, "message", exc))[:200]
    body = f"code={code}, message={message}"

    # 1) HTTP 层的状态码：429/5xx → 可重试
    if isinstance(exc, APIStatusError) and status_code is not None:
        if status_code == 429 or status_code >= 500:
            return DashScopeRetryableError(f"HTTP {status_code}: {body}")
        # 400 Invalid / 401 Auth / 403 Deny / 404 Not Found / 其它 4xx → 不可重试
        return DashScopeClientError(f"HTTP {status_code}: {body}")

    # 2) APITimeoutError：可重试（上面的 _RETRYABLE_OPENAI_BASE 已覆盖，这里再保一次险）
    if isinstance(exc, APITimeoutError):
        return DashScopeRetryableError(f"Timeout: {body}")

    # 3) 其他：APIError（请求已发出、SDK 层解析失败、APIConnectionError 类……）
    # 若 err 类型名含 Connection → 可重试；否则 ClientError。
    if type(exc).__name__.endswith("ConnectionError"):
        return DashScopeRetryableError(f"Connection: {body}")
    return DashScopeClientError(f"APIError: {body}")


# ---------------------------------------------------------------------------
# 辅助：JSON 解析双保险（强约束 + 手工 strip + Pydantic 解析）
# ---------------------------------------------------------------------------
def _strip_code_fence(text: str) -> str:
    """去掉模型偶尔包裹的 Markdown 代码围栏（```json ... ``` 或 ``` ... ```）。

    即便用了 response_format=json_object，兼容模式下部分模型仍可能多包一层 fence，
    这里作为兜底。同时处理：只有开头围栏、围栏后有语言标记、首尾空白。
    """
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_output(raw_text: str, schema: type[T]) -> T:
    """把模型返回的字符串解析为目标 Pydantic 模型；失败时把原文前 200 字进异常。"""
    content = _strip_code_fence(raw_text)
    try:
        return schema.model_validate_json(content)
    except Exception as exc:
        raise DashScopeClientError(
            f"模型输出无法解析为 {schema.__name__}: {exc}\n原始输出片段: {content[:200]}"
        ) from exc


# ===========================================================================
# 主客户端
# ===========================================================================
class DashScopeClient:
    """百炼 DashScope 统一客户端（OpenAI-compatible 协议）。

    参数
    ----
    settings : Settings | None
        可选，便于测试时注入 mock 配置；留空走全局 get_settings()。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # OpenAI SDK：api_key + base_url 统一传，真正吃全局 Settings 的那一个 URL；
        # SDK 默认读 OPENAI_API_KEY / OPENAI_BASE_URL 环境变量，这里显式传，
        # 避免与用户电脑上为真·OpenAI 设置的环境变量串扰。
        self._client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            # 兼容模式下 timeout 稍微放宽：聊天模型单次长输出较慢
            timeout=(10.0, 120.0),
        )

    # ====================== 底层通用：chat / _generate / _call ======================

    def chat(
        self,
        prompt: str,
        schema: type[T],
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> T:
        """发送一次对话请求，并将响应 JSON 解析为指定 Pydantic 模型。

        关键实现点
        ----------
        1.  同时用两条路径约束输出为合法 JSON：
            - 传 `response_format={"type": "json_object"}`（OpenAI-compatible 强制字段）
            - 在 system_prompt 最末尾追加「你必须只输出合法 JSON」—— 防止 compatible-mode
              对 response_format 的实现有差异时，LLM 仍会输出解释性文本。
        2.  parser 走 _parse_json_output（带 strip_code_fence + 清晰的错误摘要）。

        参数
        ----
        prompt / schema / model / system_prompt / temperature：与原签名完全一致，不做语义变更。

        返回 / 异常
        ----------
        T 或 DashScopeClientError / DashScopeRetryableError。
        """
        messages: list[dict[str, str]] = []
        effective_system = (system_prompt or "").rstrip()
        if effective_system:
            # system_prompt 末尾显式强调 JSON——做双保险
            effective_system += "\n\n请严格只输出合法 JSON，禁止任何解释、Markdown 代码围栏或多余文本。"
        else:
            effective_system = "你是一个严格输出合法 JSON 的助手，禁止任何解释或多余文本。"
        messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        text = self._generate(
            model=model or self.settings.interviewer_model,
            messages=messages,
            temperature=temperature,
        )
        return _parse_json_output(text, schema)

    # ====================== 语义化入口：输入解析 ======================

    def parse_resume(
        self,
        resume_text: str,
        *,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> ParsedResume:
        """解析简历原文 → ParsedResume。

        temperature 设为 0.1：结构化抽取要求确定性，不要让 LLM "创造性" 地造字段。
        """
        prompt = PARSE_RESUME_PROMPT.format(resume_text=resume_text)
        parsed = self.chat(
            prompt,
            ParsedResume,
            model=model or self.settings.interviewer_model,
            temperature=temperature,
        )
        # raw_text 必须回填原始输入（即便 LLM 写错也以调用方参数为准）
        parsed.raw_text = resume_text
        return parsed

    def parse_jd(
        self,
        jd_text: str,
        *,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> ParsedJD:
        """解析 JD 原文 → ParsedJD。解析类同样使用低 temperature。"""
        prompt = PARSE_JD_PROMPT.format(jd_text=jd_text)
        parsed = self.chat(
            prompt,
            ParsedJD,
            model=model or self.settings.interviewer_model,
            temperature=temperature,
        )
        parsed.raw_text = jd_text  # 以调用方参数为准，不接受 LLM 篡改原文
        return parsed

    # ====================== 语义化入口：面试规划 ======================

    def plan_interview(
        self,
        jd: ParsedJD,
        resume: ParsedResume | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> InterviewOutline:
        """根据结构化 JD + 结构化简历画像生成面试提纲 → InterviewOutline。

        新版 prompts 原则：下游不再传递 raw 文本，只使用解析后的结构化数据。
        temperature=0.3：既保证覆盖 must_have，又允许主题组织形式有一定灵活度。
        """
        # 把结构化对象序列化为缩进 JSON，便于 LLM 理解字段层级
        jd_profile = jd.model_dump_json(indent=2)
        resume_profile = (
            resume.model_dump_json(indent=2)
            if resume is not None
            else "（候选人画像暂无，按 JD 要求与通用情况出题）"
        )
        prompt = PLAN_INTERVIEW_PROMPT.format(
            jd_profile=jd_profile, resume_profile=resume_profile
        )
        return self.chat(
            prompt,
            InterviewOutline,
            model=model or self.settings.interviewer_model,
            temperature=temperature,
        )

    # ====================== 语义化入口：提问与追问 ======================

    def ask_question(
        self,
        jd_profile: ParsedJD | None,
        resume_profile: ParsedResume | None,
        topic: str,
        knowledge: list[str] | list[KnowledgeChunk] | None,
        history: str,
        *,
        model: str | None = None,
        temperature: float = 0.8,
    ) -> InterviewQuestion:
        """首问或切换主题时出题 → InterviewQuestion。

        新版 prompts：使用结构化画像（jd_profile / resume_profile）替代原始文本传递。
        难度不再靠全局 difficulty 参数，而是依赖主题内 difficulty + candidate_basis 自适应。
        temperature=0.8：出题允许灵活多样，避免千篇一律；prompt 内已约束首问/换题
        时不得使用 question_type=follow_up。
        """
        # 结构化画像序列化为缩进 JSON；若解析失败，给显式占位，避免 LLM 误解为空意图
        jd_text = (
            jd_profile.model_dump_json(indent=2)
            if jd_profile is not None
            else "（无岗位画像）"
        )
        resume_text = (
            resume_profile.model_dump_json(indent=2)
            if resume_profile is not None
            else "（无候选人画像）"
        )
        prompt = ASK_QUESTION_PROMPT.format(
            resume_profile=resume_text,
            jd_profile=jd_text,
            topic=topic,
            knowledge=self.format_knowledge(knowledge),
            history=history,
        )
        return self.chat(
            prompt,
            InterviewQuestion,
            model=model or self.settings.interviewer_model,
            temperature=temperature,
        )

    def follow_up(
        self,
        topic: str,
        question: str,
        answer: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> InterviewQuestion:
        """针对上一轮回答生成追问 → InterviewQuestion。

        temperature=0.7：追问需要创造力但不能太跳脱；prompt 内强制
        question_type="follow_up"，即便 LLM 误输出其它枚举值，此处也会覆盖。
        """
        prompt = FOLLOW_UP_PROMPT.format(
            topic=topic, question=question, answer=answer
        )
        result = self.chat(
            prompt,
            InterviewQuestion,
            model=model or self.settings.interviewer_model,
            temperature=temperature,
        )
        # 代码侧强约束：追问的 question_type 必须是 follow_up（PLANNING 硬规则）
        result.question_type = "follow_up"
        return result

    # ====================== 语义化入口：评分 ======================

    def judge(
        self,
        topic: str,
        question: str,
        answer: str,
        retrieval_results: list[str] | list[KnowledgeChunk] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AnswerJudgment:
        """单轮评分 → AnswerJudgment。

        temperature=0.2（出题与评分分离：评分端保持严格低随机性）。
        额外做两件代码侧兜底（PLANNING："代码做边界否决，Agent 做内容决策"）：
        1. 知识库空值 → 格式化为 "(无参考知识)"，避免 LLM 看到 "[]" 后误解。
        2. LLM 漏输出某评分维度时，按默认分（overall_score 的均值）补齐，并记录默认标记。
           防止下游聚合时维度名对不齐。
        """
        user_prompt = JUDGE_USER_PROMPT.format(
            topic=topic,
            question=question,
            answer=answer,
            knowledge=self.format_knowledge(retrieval_results),
        )
        judgment = self.chat(
            user_prompt,
            AnswerJudgment,
            model=model or self.settings.judge_model,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=temperature,
        )

        # —— 代码侧兜底：维度集合对齐 + 缺字段清洗（LLM 常漏 suggestion / 少维度）——
        # 以 overall_score 作为缺失维度兜底锚点（比中位值 5 更贴近真实水平）
        judgment.scores = normalize_score_items(
            judgment.scores, default_score=judgment.overall_score
        )

        # 回填 question / answer（即便 LLM 传错，也以调用方参数为准）
        judgment.question = question
        judgment.answer = answer

        return judgment

    # ====================== 语义化入口：最终报告 ======================

    def generate_report(
        self,
        session_id: str,
        judgments: list[AnswerJudgment],
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> InterviewReport:
        """根据评判历史生成最终报告 → InterviewReport。

        参数
        ----
        session_id : str
            从 InterviewState 取，显式传入并在 prompt 中强调"必须与上方一致"，
            避免 LLM 臆造；生成后代码侧再强制覆盖一次。
        judgments : list[AnswerJudgment]
            完整的各轮评判，直接传对象即可，本函数会 dump 成 JSON 文本填入 prompt。

        代码侧兜底
        ----------
        - report.session_id 强制覆盖为入参 session_id；
        - report.rounds 直接填入入参 judgments（不依赖 LLM 重新生成，避免数值漂移）；
        - generated_at 由 InterviewReport 模型的 default_factory 自动填 UTC 时间。
        """
        judgments_json = "[\n" + ",\n".join(
            j.model_dump_json(indent=2) for j in judgments
        ) + "\n]"
        prompt = GENERATE_REPORT_PROMPT.format(
            session_id=session_id,
            judgments=judgments_json,
        )
        report = self.chat(
            prompt,
            InterviewReport,
            model=model or self.settings.judge_model,
            temperature=temperature,
        )

        # —— 代码侧强约束：关键数据不依赖 LLM 正确输出 ——
        report.session_id = session_id          # 覆盖可能的 LLM 臆造
        report.rounds = judgments                # 不依赖 LLM 重新生成明细

        # —— 代码侧兜底 1：dimension_scores 维度集合对齐 + 缺字段清洗
        # 锚点选 judgments 的 overall_score 均值（比全局默认 5 分更贴近真实水平）
        if judgments:
            anchor = sum(j.overall_score for j in judgments) / len(judgments)
        else:
            anchor = None
        report.dimension_scores = normalize_score_items(
            report.dimension_scores, default_score=anchor
        )

        # —— 代码侧兜底 2：suggestions 逐项清洗 evidence/suggestion 空值
        # （Pydantic 字段有 default=""，但 LLM 可能显式返回 null，这里扫一遍保险）
        cleaned_suggestions: list[ImprovementSuggestion] = []
        for sug in report.suggestions:
            cleaned_suggestions.append(
                ImprovementSuggestion(
                    priority=sug.priority,
                    dimension=sug.dimension,
                    suggestion=sug.suggestion or "",
                    evidence=sug.evidence or "",
                )
            )
        report.suggestions = cleaned_suggestions

        return report

    # ====================== 辅助工具 ======================

    @staticmethod
    def format_knowledge(
        retrieval_results: list[str] | list[KnowledgeChunk] | None,
    ) -> str:
        """把检索结果拼成每行 "[1] 内容（来源：xxx）" 的文本。

        空列表/None → 返回 "(无参考知识)"，避免 LLM 看到空串或 "[]" 产生误解。
        """
        if not retrieval_results:
            return "(无参考知识)"
        lines: list[str] = []
        for index, item in enumerate(retrieval_results, start=1):
            if isinstance(item, KnowledgeChunk):
                content = item.content
                source = f"（来源：{item.source}）" if item.source else ""
            else:
                content = item
                source = ""
            lines.append(f"[{index}] {content}{source}")
        return "\n".join(lines)

    # ====================== 底层调用（含重试 + 异常翻译）======================

    def _generate(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
    ) -> str:
        """异常翻译包一层：
            - DashScopeClientError / RetryableError → 原样透出
            - openai.API*Error → 翻译为我们的统一异常（上层 except 分支不感知 openai SDK）
            - 其它未知异常 → 包成 DashScopeClientError（避免把 Python 原生异常打到节点）
        """
        try:
            return self._call(model, messages, temperature)
        except (DashScopeClientError, DashScopeRetryableError):
            raise
        except APIError as exc:
            raise _translate_openai_error(exc) from exc
        except Exception as exc:
            raise DashScopeClientError(
                f"LLM 调用失败（未分类异常）: {exc}"
            ) from exc

    @retry(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
    ) -> str:
        """实际调用 OpenAI-compatible `/chat/completions`；带 tenacity 重试装饰器。

        关键字段
        --------
        - response_format={"type":"json_object"}：OpenAI 标准 JSON 模式；
          与 chat() 里在 system 末尾追加的「只输出 JSON」组成双保险。
        - stream=False：节点层都是同步逻辑，不走流式。
        - temperature：显式传 None/浮点（None 时省略该字段，走模型默认）。

        响应解析
        --------
        OpenAI SDK 返回 ChatCompletion 对象：
          - choices[0].finish_reason 非 stop / tool_calls 直接当错误（防止 content 为 None）
          - choices[0].message.content 作为文本送入 parser
          - 解析失败不在这里 catch，留给上面的 _parse_json_output（保持错误位置清晰）
        """
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature

        completion = self._client.chat.completions.create(**kwargs)

        if not completion.choices:
            raise DashScopeClientError("LLM 返回空 choices 列表")
        choice = completion.choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish not in ("stop", "json_object", "tool_calls", None):
            # length / content_filter / 其它非停止原因 → 抛错（但不重试，避免重复触发长度限制）
            raise DashScopeClientError(
                f"LLM finish_reason 异常: finish_reason={finish}, snippet={str(choice.message)[:200]}"
            )
        content = choice.message.content
        if content is None:
            raise DashScopeClientError(
                "LLM 返回 content=None，通常由 response_format 或 token 限制导致"
            )
        return content
