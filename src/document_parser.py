from __future__ import annotations

import re
from pathlib import Path

from .agent_state import DocumentChunk, SourceDocument


SUPPORTED_SUFFIXES = {".md", ".txt"}


def parse_documents(file_paths: list[str]) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for index, file_path in enumerate(file_paths, start=1):
        path = Path(file_path).expanduser().resolve()
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path}")
        raw_text = path.read_text(encoding="utf-8")
        documents.append(
            SourceDocument(
                doc_id=f"doc_{index}",
                doc_name=path.name,
                file_path=str(path),
                raw_text=raw_text,
            )
        )
    return documents


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(r"^#{1,6}\s+\S+", stripped)
        or re.match(r"^[一二三四五六七八九十]+[、.．]\s*\S+", stripped)
        or re.match(r"^\d+(?:\.\d+)*[、.．]\s*\S+", stripped)
    )


def _heading_title(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"^#{1,6}\s+", "", stripped)
    stripped = re.sub(r"^[一二三四五六七八九十]+[、.．]\s*", "", stripped)
    stripped = re.sub(r"^\d+(?:\.\d+)*[、.．]\s*", "", stripped)
    return stripped.strip()


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]|\d+[.)、.．])\s+\S+", line))


def _flush_block(
    chunks: list[DocumentChunk],
    doc: SourceDocument,
    section_title: str,
    block: list[str],
    position: int,
) -> int:
    text = "\n".join(line.rstrip() for line in block).strip()
    if not text:
        return position
    position += 1
    chunks.append(
        DocumentChunk(
            chunk_id=f"{doc.doc_id}_chunk_{position}",
            doc_id=doc.doc_id,
            doc_name=doc.doc_name,
            section_title=section_title,
            text=text,
            position=position,
        )
    )
    return position


def chunk_documents(documents: list[SourceDocument]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for doc in documents:
        section_title = "未命名章节"
        block: list[str] = []
        position = 0
        previous_was_list = False

        for raw_line in doc.raw_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if _is_heading(line):
                position = _flush_block(chunks, doc, section_title, block, position)
                block = []
                section_title = _heading_title(line)
                position += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.doc_id}_chunk_{position}",
                        doc_id=doc.doc_id,
                        doc_name=doc.doc_name,
                        section_title=section_title,
                        text=stripped,
                        position=position,
                    )
                )
                previous_was_list = False
                continue

            if not stripped:
                position = _flush_block(chunks, doc, section_title, block, position)
                block = []
                previous_was_list = False
                continue

            current_is_list = _is_list_line(line)
            if block and current_is_list != previous_was_list:
                position = _flush_block(chunks, doc, section_title, block, position)
                block = []

            block.append(line)
            previous_was_list = current_is_list

        _flush_block(chunks, doc, section_title, block, position)

    return chunks

