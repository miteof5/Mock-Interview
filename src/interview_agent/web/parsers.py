"""上传文件解析：把简历与 JD 文件转成纯文本交给 LangGraph。"""

from __future__ import annotations

import io
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".md", ".txt", ".docx", ".pdf"}


def parse_upload(filename: str, content: bytes) -> str:
    """按后缀解析上传文件，返回纯文本；不支持或解析为空时抛 ValueError。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("暂不支持该文件格式，请上传 .md / .txt / .docx / .pdf")

    if suffix in {".md", ".txt"}:
        return content.decode("utf-8-sig", errors="replace").strip()

    if suffix == ".docx":
        document = Document(io.BytesIO(content))
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page.strip() for page in pages if page.strip()).strip()
