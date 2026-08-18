"""知识库入库 CLI：调用 knowledge/ingest.py 完成切块、向量化、写入 Chroma。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from interview_agent.config import get_settings
from interview_agent.knowledge.ingest import (
    collect_markdown_files,
    ingest_markdown_file,
    ingest_markdown_files,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建本地 Chroma 知识库")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--kb-path",
        type=Path,
        default=None,
        help="Markdown 知识库目录（默认取配置 KB_PATH）",
    )
    source_group.add_argument(
        "--file",
        type=Path,
        default=None,
        help="只入库单个 Markdown 文件",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重建前删除旧 collection",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出调试日志",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        # force=True 允许重复调用时重置 handler/level（pytest 或其它入口二次调用 main 时生效）
        force=True,
    )

    try:
        # ---- 单文件模式入口防御性校验 ----
        if args.file is not None:
            # P0-3 修复：--file 是增量语义，绝不允许带 --reset（会清空整个 collection）
            if args.reset:
                logger.error(
                    "--file（单文件增量更新）不能与 --reset 同时使用。"
                    "如需重建全库，请用 --kb-path + --reset。"
                )
                return 1
            # 存在性校验
            if not args.file.is_file():
                logger.error("文件不存在: %s", args.file)
                return 1
            # P1-6 修复：后缀校验，避免 docx/pdf 二进制被误传给 markdown chunker
            if args.file.suffix.lower() != ".md":
                logger.error(
                    "只支持 .md 后缀的 Markdown 文件，当前为 %s",
                    args.file.suffix or "无后缀",
                )
                return 1
            # P1-7 修复：resolve 成绝对路径，保证 Chroma metadata 的 source 字段
            # 不受 cwd 影响；同一个文件在不同目录下入库不会产生两条 source
            file_path = args.file.resolve()
            settings = get_settings()
            logger.info("单文件入库: %s", file_path)
            ingest_markdown_file(file_path, settings=settings)
            return 0

        # ---- 目录模式入口 ----
        settings = get_settings()
        kb_path = args.kb_path or settings.kb_path
        files = collect_markdown_files(kb_path)
        # P1-1 修复：空目录时显式提示并正常退出，避免静默 return 0 让用户误以为入库成功
        if not files:
            logger.warning("知识库目录 %s 下没有 .md 文件，结束", kb_path)
            return 0
        logger.info("发现 %d 个 Markdown 文件: %s", len(files), kb_path)
        ingest_markdown_files(files, settings=settings, reset=args.reset)
        return 0
    except Exception:
        logger.exception("知识库入库失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
