"""LangGraph 检查点管理：SQLite 生命周期包装 + 内存模式切换 + 统一入口。

职责分层（与 session.py 分工明确）：
- 本文件（checkpoint.py）：存储 LangGraph 运行期的「完整 state 快照」，
  用于进程关闭后再次打开时恢复到 interrupt() 之前的精确状态（含 history、
  judgments、计数器等全量字段）。SQLite 中是二进制/JSON blob，不适合直接
  做人类可读的界面展示。
- session.py：存储「会话摘要」（候选人姓名、岗位、启动时间、阶段、总体得分
  等展示字段），用于 CLI list 等命令快速渲染列表，不需读 checkpoint 大 blob。

使用方式：
1) 推荐统一入口 get_checkpointer()，直接拿到 saver + close 函数，graph.py
   不需要区分 memory/sqlite 两种底层实现差异。
2) 若需精细管理（如 list_thread_ids/delete_thread），用 open_checkpointer()
   拿 SqliteCheckpointStore 容器对象。
3) 纯临时测试用 get_memory_checkpointer()。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError as exc:  # 依赖未安装时给出可执行的修复提示
    raise ImportError(
        "缺少 langgraph-checkpoint-sqlite 依赖，请先安装："
        "uv add langgraph-checkpoint-sqlite"
    ) from exc

from interview_agent import models as interview_agent_models
from interview_agent.config import get_settings

# LangGraph 新版默认对未注册的 Pydantic 类型只警告、未来版本会直接拦截；
# 这里显式放行 interview_agent.models 下的领域模型，保证 checkpoint 跨版本可恢复。
_ALLOWED_MSG_MODULES = [
    (cls.__module__, cls.__name__)
    for cls in vars(interview_agent_models).values()
    if isinstance(cls, type)
    and issubclass(cls, BaseModel)
    and cls.__module__ == interview_agent_models.__name__
]
_CHECKPOINT_SERIALIZER = JsonPlusSerializer(
    allowed_msgpack_modules=_ALLOWED_MSG_MODULES
)

# SQLite 连接按 db_path 缓存，避免同一个进程对同一路径重复开多个连接。
# key = 解析后的绝对路径；value = SqliteCheckpointStore 实例。
_sqlite_store_cache: dict[Path, SqliteCheckpointStore] = {}


class SqliteCheckpointStore:
    """持有 SQLite 连接与 SqliteSaver 的包装对象，负责连接生命周期和 thread 管理。

    一般不直接实例化，走 open_checkpointer()（带缓存）或 get_checkpointer()
    （统一入口，返回 saver+close_fn）。
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self.saver = SqliteSaver(self._conn, serde=_CHECKPOINT_SERIALIZER)

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        """返回可供 StateGraph.compile(checkpointer=...) 直接使用的 saver。"""
        if self._conn is None:
            raise RuntimeError("checkpoint 连接已关闭")
        return self.saver

    def close(self) -> None:
        """关闭 SQLite 连接，避免 Windows 下文件句柄被长期占用。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        # 关闭后从全局缓存中移除，允许后续重新打开同一路径
        _sqlite_store_cache.pop(self.db_path, None)

    # ------------------------------------------------------------------
    # 上下文管理器：与 with 语句配合，退出时自动关闭连接
    # ------------------------------------------------------------------
    def __enter__(self) -> SqliteCheckpointStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 会话（thread_id）管理：提供给 CLI 的 list / delete 命令使用
    # ------------------------------------------------------------------
    def list_thread_ids(self) -> list[str]:
        """列出 checkpoint 中记录的全部会话 thread_id。"""
        if self._conn is None:
            raise RuntimeError("checkpoint 连接已关闭")
        table_exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'"
        ).fetchone()
        if not table_exists:
            return []
        rows = self._conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        return [row[0] for row in rows]

    def delete_thread(self, thread_id: str) -> None:
        """删除指定会话的全部 checkpoint（含 writes / blobs 关联表）。

        - 新版 LangGraph（>=0.2.22）的 SqliteSaver 自带 delete_thread()；
        - 老版本没有该方法时，回退到手搓 SQL 清理三张关联表，保证功能可用。
        """
        if self._conn is None:
            raise RuntimeError("checkpoint 连接已关闭")
        if hasattr(self.saver, "delete_thread"):
            self.saver.delete_thread(thread_id)
            return
        # 老版本兜底：按 LangGraph 0.2.x 默认的表结构清理
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            with suppress(sqlite3.OperationalError):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?",
                    (thread_id,),
                )
        self._conn.commit()


def open_checkpointer(db_path: Path | None = None) -> SqliteCheckpointStore:
    """打开 SQLite checkpoint 存储（**同一 db_path 多次调用返回同一对象**）。

    若 db_path 留空则使用 Settings.db_path；返回的 SqliteCheckpointStore
    内部会创建父目录并建立 sqlite 连接。

    使用方式：
        with open_checkpointer() as store:
            graph = builder.compile(checkpointer=store.checkpointer)
            ...
    """
    path = (db_path or get_settings().db_path).resolve()
    cached = _sqlite_store_cache.get(path)
    if cached is not None and cached._conn is not None:  # noqa: SLF001
        return cached
    store = SqliteCheckpointStore(path)
    _sqlite_store_cache[path] = store
    return store


def get_memory_checkpointer() -> BaseCheckpointSaver:
    """返回内存 checkpoint 实例，用于测试或不需持久化的场景。"""
    return InMemorySaver(serde=_CHECKPOINT_SERIALIZER)


# ---------------------------------------------------------------------------
# 对外统一入口：graph.py / scripts/interview.py 首选这个函数
# ---------------------------------------------------------------------------
def get_checkpointer(
    mode: Literal["memory", "sqlite"] = "sqlite",
    db_path: Path | None = None,
) -> tuple[BaseCheckpointSaver, Callable[[], None] | None]:
    """获取 LangGraph checkpointer 的统一入口，屏蔽底层实现差异。

    参数
    ----
    mode : {"memory", "sqlite"}
        memory：进程内保存，重启即丢失，适合 pytest 和临时验证；
        sqlite：持久化到 Settings.db_path（或传入的 db_path）。
    db_path : Path | None
        仅 sqlite 模式生效；留空用 Settings.db_path。

    返回
    ----
    (saver, close_fn_or_None)
        saver：可直接传给 StateGraph.compile(checkpointer=...)；
        close_fn：如果是 sqlite 模式，返回 store.close 的绑定方法，
        调用方用完后在 finally 里执行即可。memory 模式无需关闭，
        返回 None。

    示例
    ----
        saver, close_fn = get_checkpointer("sqlite")
        try:
            graph = workflow.compile(checkpointer=saver)
            for event in graph.stream(inputs, config=config):
                ...
        finally:
            if close_fn:
                close_fn()
    """
    if mode == "memory":
        return get_memory_checkpointer(), None

    # sqlite 模式：通过 open_checkpointer 走缓存，避免重复连接
    store = open_checkpointer(db_path=db_path)
    return store.checkpointer, store.close
