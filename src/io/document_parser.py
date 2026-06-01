from __future__ import annotations

import re
from pathlib import Path

from ..core.agent_state import DocumentChunk, DocumentPage, SourceDocument
from .document_structure import StructureSignal, detect_structure_signal
from .pdf_document_parser import parse_pdf_document


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
STRUCTURE_HEADING_KINDS = {
    "appendix_heading",
    "classification_title",
    "table_title",
    "continued_table_title",
}


def parse_documents(file_paths: list[str], enable_ocr: bool = False) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for index, file_path in enumerate(file_paths, start=1):
        path = Path(file_path).expanduser().resolve()
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path}")

        doc_id = f"doc_{index}"
        if path.suffix.lower() == ".pdf":
            documents.append(parse_pdf_document(path, doc_id, enable_ocr=enable_ocr))
            continue

        raw_text = path.read_text(encoding="utf-8")
        documents.append(
            SourceDocument(
                doc_id=doc_id,
                doc_name=path.name,
                file_path=str(path),
                raw_text=raw_text,
                pages=[DocumentPage(page_number=None, text=raw_text, source_method="text")],
            )
        )
    return documents


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    structure_signal = detect_structure_signal(stripped)
    return bool(
        (structure_signal and structure_signal.kind in STRUCTURE_HEADING_KINDS)
        or re.match(r"^#{1,6}\s+\S+", stripped)
        or re.match(r"^[一二三四五六七八九十]+[、.．]\s*\S+", stripped)
        or re.match(r"^\d+(?:\.\d+)*[、.．]\s*\S+", stripped)
    )


def _heading_title(line: str) -> str:
    stripped = line.strip()
    structure_signal = detect_structure_signal(stripped)
    if structure_signal and structure_signal.kind in STRUCTURE_HEADING_KINDS:
        return structure_signal.title
    stripped = re.sub(r"^#{1,6}\s+", "", stripped)
    stripped = re.sub(r"^[一二三四五六七八九十]+[、.．]\s*", "", stripped)
    stripped = re.sub(r"^\d+(?:\.\d+)*[、.．]\s*", "", stripped)
    return stripped.strip()


def _heading_level(signal: StructureSignal | None) -> int:
    if signal is None:
        return 1
    if signal.kind == "appendix_heading":
        return 1
    if signal.kind == "classification_title":
        return 2
    if signal.kind in {"table_title", "continued_table_title"}:
        return 3
    return 1


def _section_title(section_stack: list[tuple[int, str]]) -> str:
    return " / ".join(title for _, title in section_stack if title) or "未命名章节"


def _update_section_stack(section_stack: list[tuple[int, str]], level: int, title: str) -> list[tuple[int, str]]:
    return [(item_level, item_title) for item_level, item_title in section_stack if item_level < level] + [
        (level, title)
    ]


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]|\d+[.)、.．])\s+\S+", line))


def _flush_block(
    chunks: list[DocumentChunk],
    doc: SourceDocument,
    section_title: str,
    block: list[tuple[int, str]],
    position: int,
    page_number: int | None,
    source_method: str,
    source_warning: str,
) -> int:
    text = "\n".join(line.rstrip() for _, line in block).strip()
    if not text:
        return position
    position += 1
    effective_section_title = section_title
    if effective_section_title == "未命名章节" and page_number is not None:
        effective_section_title = f"Page {page_number}"
    chunks.append(
        DocumentChunk(
            chunk_id=f"{doc.doc_id}_chunk_{position}",
            doc_id=doc.doc_id,
            doc_name=doc.doc_name,
            section_title=effective_section_title,
            text=text,
            position=position,
            page_number=page_number,
            line_start=block[0][0],
            line_end=block[-1][0],
            source_method=source_method,
            source_warning=source_warning,
        )
    )
    return position


def chunk_documents(documents: list[SourceDocument]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for doc in documents:
        section_stack: list[tuple[int, str]] = []
        position = 0
        pages = doc.pages or [DocumentPage(page_number=None, text=doc.raw_text, source_method="text")]

        for page in pages:
            block: list[tuple[int, str]] = []
            previous_was_list = False
            source_method = page.source_method
            source_warning = page.warning

            for line_number, raw_line in enumerate(page.text.splitlines(), start=1):
                line = raw_line.rstrip()
                stripped = line.strip()

                if _is_heading(line):
                    structure_signal = detect_structure_signal(line, line_number=line_number)
                    position = _flush_block(
                        chunks,
                        doc,
                        _section_title(section_stack),
                        block,
                        position,
                        page.page_number,
                        source_method,
                        source_warning,
                    )
                    block = []
                    heading_title = _heading_title(line)
                    section_stack = _update_section_stack(
                        section_stack,
                        _heading_level(structure_signal),
                        heading_title,
                    )
                    position += 1
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc.doc_id}_chunk_{position}",
                            doc_id=doc.doc_id,
                            doc_name=doc.doc_name,
                            section_title=_section_title(section_stack),
                            text=stripped,
                            position=position,
                            page_number=page.page_number,
                            line_start=line_number,
                            line_end=line_number,
                            source_method=source_method,
                            source_warning=source_warning,
                        )
                    )
                    previous_was_list = False
                    continue

                if not stripped:
                    position = _flush_block(
                        chunks,
                        doc,
                        _section_title(section_stack),
                        block,
                        position,
                        page.page_number,
                        source_method,
                        source_warning,
                    )
                    block = []
                    previous_was_list = False
                    continue

                current_is_list = _is_list_line(line)
                if block and current_is_list != previous_was_list:
                    position = _flush_block(
                        chunks,
                        doc,
                        _section_title(section_stack),
                        block,
                        position,
                        page.page_number,
                        source_method,
                        source_warning,
                    )
                    block = []

                block.append((line_number, line))
                previous_was_list = current_is_list

            position = _flush_block(
                chunks,
                doc,
                _section_title(section_stack),
                block,
                position,
                page.page_number,
                source_method,
                source_warning,
            )

    return chunks
