"""Markdown 知识库切块：实现《Markdown 知识库切分策略》的全部规则。

切分流程（自上而下）：
    1. _parse()              按 Markdown 标题建立文档结构树（_Node），同时保持代码块不被误解析
    2. _chunk_node()         按节点递归切分：
                             - level 2（##）：策略文档 §2.1 优先级 1，默认每个 ## 独立成块
                             - level 3+（### / ####）：超过 max_tokens 时才继续按子标题下切
                             - 超过 level 4（##### / ######）：不再继续按标题切，用段落兜底
    3. _split_leaf()         标题下无子标题但内容过长时：按 _Block 粒度聚合，代码块/表格/
                             Mermaid 永不切分，段落用句子单位组合
    4. _merge_small()        最后一遍合并 < min_tokens 的 Chunk
    5. chunk_text() overlap 注入 + 元数据打包成 Chunk 列表

切分策略文档的"结构优先 / 语义完整 / 代码保护 / 元数据继承 / 图文协同"5 大原则在
对应 helper 函数里有单独注释说明。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 正则常量
# ---------------------------------------------------------------------------
# Markdown 标题行，如 "# 标题"、"### 标题"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# Markdown 代码块围栏：支持 ``` 和 ~~~，允许 language tag，允许围栏多余字符（`````）
_FENCE_OPEN_RE = re.compile(r"^(```+|~~~+)\s*([\w+-]*)\s*$")
# CJK 字符范围：中日韩越统一表意文字 + 中文标点 + 假名 + 全角符号等
_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)
# 英文"词"：非空白连续字符
_WORD_RE = re.compile(r"\S+")
# 兜底句级切分：在 CJK 句号/问号/感叹号/分号后或换行处拆分句子
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s+|\n+")

# 与策略文档 §2.1 保持一致：不再继续按标题下切的层级
# 策略文档只支持切到 #### 为止；##### / ###### 以下都当作普通段落处理
_MAX_SPLIT_LEVEL = 4
# 策略文档 §2.1 优先级 1：## 级标题默认独立成块，不用 "<=max_tokens" 整坨保存
_FORCE_SPLIT_LEVEL = 2


def estimate_tokens(text: str) -> int:
    """估算 token 数，与 DashScope/text-embedding-v2 的中文 1 token/字、英文约 4 char/
    token 对齐。仅用于切分阈值判断，不追求与真正 tokenizer 完全一致。"""
    if not text:
        return 0
    cjk_count = len(_CJK_RE.findall(text))
    # 把 CJK 字符替换成空格，剩下的用英文词统计
    rest = _CJK_RE.sub(" ", text)
    word_count = sum(max(1, (len(word) + 3) // 4) for word in _WORD_RE.findall(rest))
    return cjk_count + word_count


# ---------------------------------------------------------------------------
# 对外数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    """单个可独立理解的知识单元：向量化 + 检索的最小单位。"""

    text: str                                 # Chunk 原文（可能含 overlap）
    metadata: dict[str, str | int | bool]     # 继承的元数据：source/doc_title/heading_path/...
    token_count: int                          # estimate_tokens 估算值，便于检索后裁剪上下文


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    """Markdown 标题树节点：level=0 是根（没有标题），level=1 是 #，level=2 是 ## 等。"""

    level: int                          # 标题等级，0=根节点
    title: str                          # 标题文本
    parent: "_Node | None"              # 父节点，用于回溯 heading_path
    body: list[str] = field(default_factory=list)     # 本级标题"自身 body"（不含 children 的文本）
    children: list["_Node"] = field(default_factory=list)  # 子标题节点列表

    def __post_init__(self) -> None:
        # 子节点构造时自动挂到父节点下；保持与策略文档"继承父标题路径"一致
        if self.parent is not None:
            self.parent.children.append(self)

    @property
    def heading_path(self) -> str:
        """生成元数据用的完整标题路径，如 '## 一、基础 / ### 1.1 介绍'。"""
        parts: list[str] = []
        node: _Node | None = self
        while node is not None and node.level > 0:
            parts.append(f"{'#' * node.level} {node.title}")
            node = node.parent
        return " / ".join(reversed(parts))

    @property
    def body_text(self) -> str:
        """当前节点自身 body（不含 children）的纯文本。"""
        return "\n".join(self.body).strip()

    def full_text(self) -> str:
        """当前节点（含所有子节点）的完整 Markdown 文本。用于判断整体大小。"""
        lines: list[str] = []
        if self.level:
            lines.append(f"{'#' * self.level} {self.title}")
        if self.body:
            lines.extend(self.body)
        for child in self.children:
            lines.append(child.full_text())
        return "\n".join(lines)


@dataclass
class _Block:
    """_split_blocks() 产出的语法块：段落 / 代码 / Mermaid / 表格。"""

    kind: str           # "paragraph" | "code" | "mermaid" | "table"
    text: str           # 整块原始文本（包含 fence / 表格分隔线等原始内容）


@dataclass
class _RawChunk:
    """chunk_text() 前的原始 Chunk：还没加 overlap、还没注入最终元数据。"""

    text: str                           # 纯文本
    node: _Node                         # 归属的标题节点（用来拿 heading_path 等）
    flags: dict[str, bool]              # is_code_block / is_mermaid / is_table


# ---------------------------------------------------------------------------
# 步骤 1：_parse() 解析标题树 + fence 状态机（P0-1 修复）
# ---------------------------------------------------------------------------
def _parse(text: str) -> tuple[str | None, _Node]:
    """解析 Markdown 为 (doc_title, root_node) 结构。

    关键机制（P0-1 修复）：围栏代码块（``` / ~~~）内部的行一律直接 append 到 body，
    不匹配 _HEADING_RE，避免 Python/Ruby/... 中以 `# 注释` 开头的行被误判成标题。
    """
    root = _Node(level=0, title="", parent=None)
    stack: list[_Node] = [root]
    doc_title: str | None = None

    # fence 状态机：正在 fence 内部时为 (marker_char, required_len) 的 tuple；None 代表正文
    fence_state: tuple[str, int] | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        # ---- fence 状态：正文里出现 opening fence ----
        if fence_state is None:
            fence_match = _FENCE_OPEN_RE.match(stripped)
            if fence_match:
                marker = fence_match.group(1)   # "```" 或 "~~ ~"
                stack[-1].body.append(raw_line)
                fence_state = (marker[0], len(marker))
                continue
            # ---- fence 状态：处于 fence 内部，遇到 closing fence 才退出 ----
        else:
            marker_char, required_len = fence_state
            # closing fence 的判断：相同字符、数量 >= opening、前后无有效内容
            if (
                stripped.startswith(marker_char)
                and len(stripped) >= required_len
                and all(ch == marker_char for ch in stripped)
            ):
                stack[-1].body.append(raw_line)
                fence_state = None
                continue
            # fence 内：不做任何标题解析，原封不动塞 body（核心修复点）
            stack[-1].body.append(raw_line)
            continue

        # ---- 正常正文：先尝试匹配标题 ----
        match = _HEADING_RE.match(raw_line)
        if match is None:
            stack[-1].body.append(raw_line)
            continue

        level = len(match.group(1))
        title = match.group(2).strip()
        # 策略文档：doc_title 取文档中第一个一级标题（只取一次，不被后续 # 覆盖）
        if level == 1 and stack[-1] is root:
            doc_title = doc_title or title

        # 弹出 stack 直到找到当前标题的父节点：当 stack[-1].level >= level 时需要回退
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        _Node(level=level, title=title, parent=stack[-1])
        stack.append(stack[-1].children[-1])

    return doc_title, root


# ---------------------------------------------------------------------------
# 代码块/表格/段落的语法块切分 + Mermaid 与前置文字组合（策略文档 §2.3）
# ---------------------------------------------------------------------------
def _split_blocks(text: str) -> list[_Block]:
    """把一段纯文本拆成语法块：fence(代码/Mermaid) / 表格 / 段落。

    对应策略文档 §2.3 / §4.1 / §4.3：
    - fence 内永不切分（作为一整块 `_Block` 输出）
    - Markdown 表格永不切分（连续的以 `|` 开头的行合并成一整块）
    """
    blocks: list[_Block] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- fence 代码块 ----
        fence_match = _FENCE_OPEN_RE.match(stripped)
        if fence_match:
            marker = fence_match.group(1)
            language = fence_match.group(2).lower()
            marker_char = marker[0]
            required_len = len(marker)
            fence_lines = [line]
            i += 1
            while i < len(lines):
                fence_lines.append(lines[i])
                cur_stripped = lines[i].strip()
                if (
                    cur_stripped.startswith(marker_char)
                    and len(cur_stripped) >= required_len
                    and all(ch == marker_char for ch in cur_stripped)
                ):
                    i += 1
                    break
                i += 1
            kind = "mermaid" if language == "mermaid" else "code"
            blocks.append(_Block(kind, "\n".join(fence_lines)))
            continue

        # ---- Markdown 表格 ----
        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(_Block("table", "\n".join(table_lines)))
            continue

        # ---- 普通段落：遇空行 / fence / 表格时结束 ----
        paragraph_lines: list[str] = []
        while i < len(lines):
            next_line = lines[i].strip()
            if not next_line:
                break
            if _FENCE_OPEN_RE.match(next_line) or next_line.startswith("|"):
                break
            paragraph_lines.append(lines[i])
            i += 1
        if paragraph_lines:
            blocks.append(_Block("paragraph", "\n".join(paragraph_lines)))
        else:
            i += 1
    return blocks


def _group_mermaid_blocks(blocks: list[_Block]) -> list[_Block]:
    """策略文档 §4.2 图文协同：Mermaid 块与其**紧前的段落**合并，保证检索时图文一起命中。"""
    grouped: list[_Block] = []
    for block in blocks:
        if (
            block.kind == "mermaid"
            and grouped
            and grouped[-1].kind == "paragraph"
        ):
            # 图文合并：保留"段落说明文字 + Mermaid 代码"的顺序
            previous = grouped.pop()
            grouped.append(_Block("paragraph", f"{previous.text}\n\n{block.text}"))
        else:
            grouped.append(block)
    return grouped


# ---------------------------------------------------------------------------
# 兜底切分：标题下无子标题但段落过长时，句子级组合
# ---------------------------------------------------------------------------
def _hard_split(text: str, max_tokens: int) -> list[str]:
    """最后手段的字符级兜底切分：句子组合仍超大时，优先在句末标点处断开。"""
    max_chars = max(1, max_tokens * 4)
    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        piece = remaining[:max_chars]
        # 优先在 CJK/英文的句末标点或空格处断开，尽量保持语义完整
        cut = max(
            piece.rfind("。"),
            piece.rfind("！"),
            piece.rfind("？"),
            piece.rfind("."),
            piece.rfind(" "),
        )
        cut = cut if cut > 0 else max_chars
        pieces.append(piece[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_long_paragraph(text: str) -> list[str]:
    """按句子拆成"语义单位"，供 _combine_paragraph_units 再组装。"""
    units = [unit.strip() for unit in _SENTENCE_SPLIT_RE.split(text) if unit.strip()]
    return units or [text.strip()]


def _combine_paragraph_units(
    units: list[str], max_tokens: int, target_tokens: int
) -> list[str]:
    """句子 → Chunk 组合器：尽量接近 target_tokens；超过 max_tokens 必定拆分。"""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        # 单个句子就超 max：先 flush 现有，再用 _hard_split 递归处理
        if unit_tokens > max_tokens:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_tokens = 0
            chunks.extend(
                _combine_paragraph_units(
                    _hard_split(unit, max_tokens), max_tokens, target_tokens
                )
            )
            continue
        # 已接近 target / 再塞一句会超 max：先 flush 再新起一个
        if current and (
            current_tokens + unit_tokens > max_tokens
            or current_tokens >= target_tokens
        ):
            chunks.append("\n".join(current))
            current = [unit]
            current_tokens = unit_tokens
        else:
            current.append(unit)
            current_tokens += unit_tokens

    if current:
        chunks.append("\n".join(current))
    return chunks


def _block_kind(text: str) -> str:
    """判断一段完整文本"纯 block 类型"：若不是单一 block 就返回 'mixed'。"""
    blocks = _split_blocks(text)
    if len(blocks) == 1:
        return blocks[0].kind
    return "mixed"


# ---------------------------------------------------------------------------
# MarkdownChunker：对外主类
# ---------------------------------------------------------------------------
class MarkdownChunker:
    """策略文档驱动的 Markdown 切块器：结构优先 + 代码保护 + 元数据继承。

    参数（与策略文档 §2.2 对齐）：
        max_tokens      最大 Chunk 大小，默认 1500（策略文档 §2.2）
        min_tokens      最小 Chunk 大小，低于此值会被合并，默认 200
        target_tokens   目标 Chunk 大小，段落组合时的"软上限"，默认 1000
        overlap_chars   Chunk 间重叠字符数（策略文档 §2.2 推荐 100-200），默认 150
    """

    def __init__(
        self,
        max_tokens: int = 1500,
        min_tokens: int = 200,
        target_tokens: int = 1000,
        overlap_chars: int = 150,
    ) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.overlap_chars = overlap_chars

    # ------------------------------ 对外 API ------------------------------
    def chunk_text(self, text: str, source: str = "") -> list[Chunk]:
        """把一整篇 Markdown 文本切块，source 用于元数据记录来源路径。"""
        doc_title, root = _parse(text)
        raw_chunks = self._chunk_node(root)
        raw_chunks = self._merge_small(raw_chunks)
        chunks: list[Chunk] = []

        for index, raw in enumerate(raw_chunks):
            chunk_text = raw.text
            # 策略文档 §2.2：overlap 注入——当前 Chunk 开头追加前一个 Chunk 的尾部
            if index > 0:
                overlap = self._safe_overlap(raw_chunks[index - 1].text)
                if overlap:
                    chunk_text = f"{overlap}\n\n{chunk_text}"
            metadata: dict[str, str | int | bool] = {
                "source": source,                        # 来源：文件路径（chunk_file 会写入）
                "doc_title": doc_title or "",             # 文档的一级标题
                "heading_path": raw.node.heading_path,    # 标题路径（元数据继承）
                "heading_level": raw.node.level,          # 归属标题层级
                "chunk_index": index,                     # 该文档中的第几个 Chunk（从 0 起）
                "is_code_block": raw.flags.get("is_code_block", False),
                "is_mermaid": raw.flags.get("is_mermaid", False),
                "is_table": raw.flags.get("is_table", False),
            }
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=metadata,
                    token_count=estimate_tokens(chunk_text),
                )
            )
        return chunks

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        """便捷接口：读取 Markdown 文件后切块，source 自动写入文件路径。"""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return self.chunk_text(text, source=str(path))

    # ------------------------------ 步骤 2：递归按标题/节点拆分 ------------------------------
    def _chunk_node(self, node: _Node) -> list[_RawChunk]:
        """按标题节点递归切分：与策略文档 §2.1 优先级表严格对齐。

        策略文档 §2.1：
            优先级 1 → ##（level==_FORCE_SPLIT_LEVEL）：默认切分点，独立成块
            优先级 2 → ###：仅当父节点 > max_tokens 时继续下切
            优先级 3 → ####：仅当父节点 > max_tokens 时继续下切
            优先级 4 → 段落：没有标题时的兜底
        额外：超过 _MAX_SPLIT_LEVEL (=4) 的标题（#####/######）不再按标题下切，直接按叶子段落处理。
        """
        full_text = node.full_text().strip()
        if not full_text:
            return []

        # ---- 整体保存条件（"整坨保存" vs "继续按 children/leaf 拆"） ----
        whole_save_ok: bool
        if node.level == _FORCE_SPLIT_LEVEL:
            # 策略文档 P0-2 修复：## 级永远不整坨保存，至少保证每个 ## 独立一个 Chunk
            whole_save_ok = False
        elif node.level > _MAX_SPLIT_LEVEL:
            # 超过最大下切层级（##### / ######）：也不能整坨保存，需要按段落兜底
            whole_save_ok = False
        else:
            # level==1 / 3 / 4 的常规判断：
            #   要么整体 <=max_tokens（直接保存），要么是"无 children 的纯代码/纯表格/纯 Mermaid 块"
            whole_save_ok = estimate_tokens(full_text) <= self.max_tokens or (
                not node.children and _block_kind(full_text) != "mixed"
            )
        if whole_save_ok:
            return [_RawChunk(full_text, node, self._flags(full_text))]

        # ---- 需要继续拆：优先按 children 拆 ----
        if node.children:
            chunks: list[_RawChunk] = []
            # 父节点自己的 body（如果有）作为独立 Chunk 放在最前面
            if node.body_text:
                chunks.append(_RawChunk(node.body_text, node, self._flags(node.body_text)))
            # 递归处理每一个子标题节点
            for child in node.children:
                chunks.extend(self._chunk_node(child))
            return chunks

        # ---- 叶子节点（无 children）且内容过长：按 _Block + 段落兜底（优先级 4） ----
        return [
            _RawChunk(part, node, self._flags(part))
            for part in self._split_leaf(node, full_text)
        ]

    # ------------------------------ 步骤 3：叶子节点按 _Block 聚合 ------------------------------
    def _split_leaf(self, node: _Node, text: str) -> list[str]:
        """标题下无子标题但整体 >max_tokens：按 _Block 粒度组合，代码/表格/Mermaid 永不切。"""
        # 图文协同：先把 Mermaid 和其紧前的段落合并
        blocks = _group_mermaid_blocks(_split_blocks(text))
        chunk_texts: list[str] = []
        current: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if current:
                chunk_texts.append("\n\n".join(current))
                current = []
                current_tokens = 0

        for block in blocks:
            block_tokens = estimate_tokens(block.text)
            # 单个 block 已超 max：
            # - 段落：句子拆分后再组合
            # - 代码/表格/Mermaid：策略文档 §2.3/§4 永不切分，整体作为超大 Chunk 入库
            if block_tokens > self.max_tokens:
                flush()
                if block.kind == "paragraph":
                    chunk_texts.extend(
                        _combine_paragraph_units(
                            _split_long_paragraph(block.text),
                            self.max_tokens,
                            self.target_tokens,
                        )
                    )
                else:
                    chunk_texts.append(block.text)
                continue
            # 已接近 target / 再塞一块会超 max：先 flush
            if current and (
                current_tokens + block_tokens > self.max_tokens
                or current_tokens >= self.target_tokens
            ):
                flush()
            current.append(block.text)
            current_tokens += block_tokens

        flush()
        return chunk_texts

    # ------------------------------ 辅助：flags / merge_small / safe_overlap ------------------------------
    def _flags(self, text: str) -> dict[str, bool]:
        """从纯文本推断三种特殊块标记，用于元数据写入 chunk.metadata。"""
        kind = _block_kind(text)
        return {
            "is_code_block": kind == "code",
            "is_mermaid": kind == "mermaid",
            "is_table": kind == "table",
        }

    def _merge_small(self, raw_chunks: list[_RawChunk]) -> list[_RawChunk]:
        """一遍扫描合并 <min_tokens 的 Chunk（P1-2：简化为单向一遍）。

        逻辑：顺序 append → 每一步判断"最后 2 个加起来 ≤max_tokens 且最后一个 <min_tokens"，
        成立则合并，与 _combine_paragraph_units 的风格统一。
        """
        merged: list[_RawChunk] = []
        for item in raw_chunks:
            merged.append(item)
            # 尝试把 merged 末尾的两个小 chunk 合并成一个
            while len(merged) >= 2:
                prev_, last_ = merged[-2], merged[-1]
                last_tokens = estimate_tokens(last_.text)
                if last_tokens >= self.min_tokens:
                    break
                combined_tokens = estimate_tokens(prev_.text) + last_tokens
                if combined_tokens > self.max_tokens:
                    break
                # 合并：保留 prev_.node 作为归属（prev_ 更靠近"父标题"，语义更自然）
                new_flags = {
                    k: prev_.flags.get(k, False) or last_.flags.get(k, False)
                    for k in ("is_code_block", "is_mermaid", "is_table")
                }
                merged[-2:] = [
                    _RawChunk(f"{prev_.text}\n\n{last_.text}", prev_.node, new_flags)
                ]
        return merged

    def _safe_overlap(self, prev_text: str) -> str:
        """P1-1：把 overlap 截断在 fence 外部，避免半截 fence 被拼到下一个 Chunk 破坏结构。

        逻辑：先朴素取末尾 overlap_chars 个字符，然后检查"overlap 范围内的 fence 数量"。
        若 fence 数量为奇数，说明 overlap 落在一个未闭合的 fence 内部——向前回溯到最近一个
        closing fence 行外，保证 overlap 中的 fence 总是成对。
        """
        overlap = prev_text[-self.overlap_chars:].lstrip("\n")
        if not overlap:
            return ""
        # 统计 fence 行数量（只认 start 行）
        fence_count = sum(
            1 for ln in overlap.splitlines() if _FENCE_OPEN_RE.match(ln.strip())
        )
        if fence_count % 2 == 0:
            return overlap

        # fence 数量奇数：overlap 截断在 fence 内部。找到 prev_text 中 overlap_start 之前
        # 最近的一个 closing fence（即 "```" 或 "~~~" 整行），作为 overlap 的新起点
        overlap_start_idx = len(prev_text) - self.overlap_chars
        search_from = max(0, overlap_start_idx - 1)
        lines_before = prev_text[:search_from].splitlines(keepends=True)

        # 从后往前找最近的 closing fence
        safe_cut_idx = 0
        char_cursor = 0
        for i, line in enumerate(lines_before):
            stripped = line.strip()
            if _FENCE_OPEN_RE.match(stripped):
                # 这一行在 overlap 之前，作为安全起点
                safe_cut_idx = char_cursor
                # 但是 closing fence 自己要在 overlap 里（保证 fence 闭合）
                # 所以 safe_cut_idx 指向这一行的开头，作为 overlap 的起点
                # 每次 closing fence 都更新一次，保证最终取到的是 overlap 之前的最后一次 closing
            char_cursor += len(line)

        if safe_cut_idx == 0 and prev_text[: overlap_start_idx or 1].count("```") % 2 == 1:
            # 特殊情况：往前找不到独立的 closing fence（说明文本开头就在 fence 里），
            # 就放弃 overlap，避免拼接异常结构
            return ""

        safe_overlap = prev_text[safe_cut_idx:].lstrip("\n")
        # 限制 overlap 不超过 overlap_chars * 2（极端情况 closing fence 很远）
        if len(safe_overlap) > max(self.overlap_chars * 2, 400):
            return ""
        return safe_overlap


__all__ = ["Chunk", "MarkdownChunker", "estimate_tokens"]
