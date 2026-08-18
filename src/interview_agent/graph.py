"""Graph 组装：真实节点 + interrupt + 条件路由。"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from interview_agent.nodes.ask_question import ask_question
from interview_agent.nodes.decide_next import decide_next
from interview_agent.nodes.generate_report import generate_report
from interview_agent.nodes.judge_answer import judge_answer
from interview_agent.nodes.parse_inputs import parse_inputs
from interview_agent.nodes.plan_interview import plan_interview
from interview_agent.nodes.retrieve_knowledge import retrieve_knowledge
from interview_agent.nodes.wait_answer import wait_answer
from interview_agent.state import InterviewState
from interview_agent.storage.checkpoint import get_checkpointer

_LOOP_TARGET = "retrieve_knowledge"
_REPORT_TARGET = "generate_report"


def _route_after_decide(state: InterviewState) -> str:
    """按 last_decision.action 路由：追问/切题回到检索，结束进报告。"""
    decision = state.get("last_decision")
    action = decision.action if decision is not None else "end"
    if action in ("follow_up", "next_topic"):
        return _LOOP_TARGET
    return _REPORT_TARGET


_app = None


def build_app(mode: str = "sqlite", db_path: Path | None = None):
    """组装真实面试节点，挂载 checkpoint 后编译。

    参数
    ----
    mode : {"sqlite", "memory"}
        sqlite：持久化到 Settings.db_path（生产路径）；
        memory：进程内 InMemorySaver，适合测试与哑跑，重启即丢。
    db_path : Path | None
        仅 sqlite 模式生效；留空用 Settings.db_path。
    """
    builder = StateGraph(InterviewState)
    builder.add_node("parse_inputs", parse_inputs)
    builder.add_node("plan_interview", plan_interview)
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("ask_question", ask_question)
    builder.add_node("wait_answer", wait_answer)
    builder.add_node("judge_answer", judge_answer)
    builder.add_node("decide_next", decide_next)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "parse_inputs")
    builder.add_edge("parse_inputs", "plan_interview")
    builder.add_edge("plan_interview", "retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", "ask_question")
    builder.add_edge("ask_question", "wait_answer")
    builder.add_edge("wait_answer", "judge_answer")
    builder.add_edge("judge_answer", "decide_next")
    builder.add_conditional_edges(
        "decide_next",
        _route_after_decide,
        {
            _LOOP_TARGET: _LOOP_TARGET,
            _REPORT_TARGET: _REPORT_TARGET,
        },
    )
    builder.add_edge("generate_report", END)

    saver, _ = get_checkpointer(mode, db_path)
    return builder.compile(checkpointer=saver)


def get_app():
    """懒加载编译后的 app（默认 sqlite 模式）。

    供生产/正式面试入口使用；测试与哑跑应直接调 build_app(mode="memory")
    以避免污染生产 sqlite。
    """
    global _app
    if _app is None:
        _app = build_app()
    return _app
