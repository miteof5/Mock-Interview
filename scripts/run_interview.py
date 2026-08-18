"""面试 CLI：读入简历/JD，跑通完整面试流程（含 interrupt 交互循环）。

执行流程：
  START → parse_inputs → plan_interview → retrieve_knowledge → ask_question
       → wait_answer [interrupt 暂停，从 stdin 读用户回答]
       → judge_answer → decide_next
       → (follow_up / next_topic) → retrieve_knowledge → ask_question → ...
       → (end) → generate_report → END

使用示例：
  python -m scripts.run_interview --resume my_resume.md --jd my_jd.md
  python -m scripts.run_interview --resume r.md --jd j.md --checkpoint-mode sqlite
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from langgraph.types import Command

from interview_agent.graph import build_app
from interview_agent.state import build_initial_state


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit(f"文件为空: {path}")
    return text.strip()


def _print_interrupt_question(interrupt_value: object) -> None:
    """从 interrupt payload 提取问题并打印给用户。

    wait_answer 节点的 interrupt 入参是 {"question": "..."} 形式。
    """
    question = ""
    if isinstance(interrupt_value, dict):
        question = interrupt_value.get("question", "")
    elif isinstance(interrupt_value, str):
        question = interrupt_value
    print("\n" + "=" * 60)
    print(f"面试官: {question}")
    print("=" * 60)


def _read_answer_from_stdin() -> str:
    """从 stdin 读用户回答，支持多行（空行结束）。

    - 已输入内容后遇到空行 → 结束输入
    - 输入 'quit' → 主动结束面试（不加二次确认，直接退出整个流程）
    - 空回答会要求重新输入
    """
    while True:
        # 提示文案明确两种操作的语义边界，避免用户误以为"空行"是退出
        print("\n请输入你的回答（输入完成后按空行提交本题；输入 'quit' 退出整个面试）:")
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            stripped = line.strip()
            if stripped.lower() == "quit":
                raise SystemExit("用户主动结束面试")
            if not stripped and lines:
                # 已有内容后遇到空行 → 结束
                break
            if stripped:
                lines.append(line)
        answer = "\n".join(lines).strip()
        if answer:
            return answer
        print("回答不能为空，请重新输入")


def _print_final_report(result: dict) -> None:
    """流程结束后打印最终报告摘要。"""
    print("\n" + "=" * 60)
    print("面试结束 - 最终报告")
    print("=" * 60)

    report = result.get("report")
    if report is not None:
        print(f"会话 ID: {report.session_id}")
        print(f"总评分: {report.overall_score:.1f} / 10")
        print(f"总体评语: {report.summary}")
        if report.dimension_scores:
            print("\n各维度得分:")
            for score in report.dimension_scores:
                print(f"  - {score.dimension}: {score.score:.1f}")
                if score.evidence:
                    print(f"    依据: {score.evidence}")
        if report.suggestions:
            print("\n改进建议:")
            for sug in report.suggestions:
                print(f"  [{sug.priority.upper()}] {sug.dimension}: {sug.suggestion}")
    else:
        print("（未生成报告）")

    print(f"\n总题数: {result.get('question_count', 0)}")
    print(f"评判轮数: {len(result.get('judgments', []))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="面试 CLI：读入简历/JD，跑通完整面试流程"
    )
    parser.add_argument("--resume", type=Path, required=True, help="简历原文 md 文件")
    parser.add_argument("--jd", type=Path, required=True, help="JD 原文 md 文件")
    # 默认时间戳 session-id，避免重复跑污染同一 thread_id 的 checkpoint
    default_session = f"interview-{datetime.now():%Y%m%d-%H%M%S}"
    parser.add_argument("--session-id", default=default_session, help="会话 id")
    # 默认 memory 模式，避免污染生产 sqlite；正式面试可显式切 sqlite 持久化
    parser.add_argument(
        "--checkpoint-mode",
        choices=["memory", "sqlite"],
        default="memory",
        help="checkpoint 模式：memory（默认，进程内）或 sqlite（持久化到 Settings.db_path）",
    )
    args = parser.parse_args(argv)

    # 按需构建 app，避免 import 时副作用；memory 模式不触达生产 sqlite
    app = build_app(mode=args.checkpoint_mode)

    initial = build_initial_state(
        {
            "session_id": args.session_id,
            "resume_text": _read_text(args.resume),
            "jd_text": _read_text(args.jd),
        }
    )

    config = {"configurable": {"thread_id": args.session_id}}

    # 第一次 invoke：从 START 跑到 wait_answer 节点的 interrupt 暂停
    result = app.invoke(initial, config=config)

    # interrupt 交互循环：遇到 __interrupt__ 就读用户输入，用 Command(resume=) 恢复
    while "__interrupt__" in result:
        # result["__interrupt__"] 是列表，取第一个（本项目只有一个 interrupt 点）
        interrupt_info = result["__interrupt__"][0]
        _print_interrupt_question(interrupt_info.value)
        answer = _read_answer_from_stdin()
        # 状态反馈：让用户知道流程没卡死，是在等 LLM 评分/出下一题
        # 实测 DashScope 兼容模式下单次评分耗时 5-30s，没提示会以为 hang 住
        print("\n⏳ 评分中，请稍候（LLM 正在评判你的回答，通常 5-30 秒）...")
        # Command(resume=answer) 恢复执行，answer 会作为 interrupt() 的返回值
        result = app.invoke(Command(resume=answer), config=config)

    _print_final_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
