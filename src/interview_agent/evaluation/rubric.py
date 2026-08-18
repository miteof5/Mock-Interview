"""评分维度的共享常量与辅助函数（全局单一权威来源）。

为什么单独放这里？
    Lessons Learned：之前 RUBRIC_DIMENSIONS 定义在 llm/client.py 里，
    evaluation/report_generator.py / nodes/judge_answer.py 要引用时只能
    反向 import client，形成 "import client → 顶层 import dashscope" 的
    传导性依赖坑（历史上真实踩过）。抽到这里后：
        - llm/client.py        → from evaluation.rubric import RUBRIC_DIMENSION_NAMES
        - evaluation/report_generator.py → 同上
        - nodes/*              → 同上
    任何模块都不需要感知 LLM SDK，依赖方向干净。
"""

from __future__ import annotations

from typing import Iterable

from interview_agent.models import ScoreItem

# ---------------------------------------------------------------------------
# 评分固定维度名：与 JUDGE_SYSTEM_PROMPT / generate_report 约定严格一致
# ---------------------------------------------------------------------------
# 注意：这里只定义"名字集合"，具体锚点（0-10 打分规则）写在 prompts.py
# 的 JUDGE_SYSTEM_PROMPT 里，两者文本必须人工保持一致。
RUBRIC_DIMENSION_NAMES: tuple[str, ...] = (
    "专业准确性",
    "表达结构",
    "岗位匹配度",
    "应变能力",
)

# 缺失维度兜底时的默认分（与 judge() 里"用 overall_score 做锚点"不冲突）
# 当无法拿到 overall_score 时（如 report 维度聚合）退回到这个中位值
_DEFAULT_MISSING_SCORE = 5.0


# ---------------------------------------------------------------------------
# 公共辅助：维度对齐 + 缺字段兜底
# ---------------------------------------------------------------------------
def normalize_score_items(
    items: Iterable[ScoreItem],
    *,
    default_score: float | None = None,
) -> list[ScoreItem]:
    """把任意来源的 ScoreItem 列表对齐到 RUBRIC_DIMENSION_NAMES。

    三件事（顺序重要）：
        1) 去重：同一 dimension 出现多次时保留第一个（防止 LLM 手抖重复输出）；
        2) 补缺：RUBRIC_DIMENSION_NAMES 中有但 items 里没有的维度，按
           default_score 写入一条"LLM 未输出"的默认项；
        3) 扫尾：每个已存在的 ScoreItem 的 evidence / suggestion 字段
           若为 None 则替换成空串（避免 Pydantic 严格模式或序列化时炸）。

    返回顺序严格等于 RUBRIC_DIMENSION_NAMES，下游按 index 取不会错位。
    """
    score_value = (
        float(default_score)
        if default_score is not None
        else _DEFAULT_MISSING_SCORE
    )
    # clamp 到 [0, 10] 防止负数或爆表
    score_value = max(0.0, min(10.0, score_value))

    seen: dict[str, ScoreItem] = {}
    for it in items:
        if it.dimension in seen:
            # 重复维度：忽略后续项（LLM 偶尔抖出两条同维度）
            continue
        # evidence / suggestion 字段 Pydantic 有默认 ""，但兼容 LLM 显式返回 null
        evidence = it.evidence or ""
        suggestion = it.suggestion or ""
        # score Pydantic 已校验 [0,10]，这里只保守再夹一次
        safe_score = max(0.0, min(10.0, float(it.score)))
        seen[it.dimension] = ScoreItem(
            dimension=it.dimension,
            score=safe_score,
            evidence=evidence,
            suggestion=suggestion,
        )

    result: list[ScoreItem] = []
    for dim in RUBRIC_DIMENSION_NAMES:
        if dim in seen:
            result.append(seen[dim])
        else:
            result.append(
                ScoreItem(
                    dimension=dim,
                    score=round(score_value, 2),
                    evidence=f"(LLM 未输出该维度，按默认分 {round(score_value, 2)} 兜底)",
                    suggestion="",
                )
            )
    return result


__all__ = [
    "RUBRIC_DIMENSION_NAMES",
    "normalize_score_items",
]
