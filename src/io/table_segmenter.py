from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.agent_state import DocumentChunk
from .document_structure import build_structure_context, detect_structure_signal


HIERARCHICAL_CODE_RE = re.compile(r"(?<!\d)(\d+(?:\s*[.．]\s*\d+)+)(?!\d)")


@dataclass(frozen=True)
class TableSegment:
    segment_id: str
    source_chunk_id: str
    doc_name: str
    section_title: str
    text: str
    position: int
    page_number: int | None
    line_start: int | None
    line_end: int | None
    source_method: str
    source_warning: str
    block_signal: str
    header_text: str = ""
    structure_context: dict[str, Any] = field(default_factory=dict)
    flattened_row_hints: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            self.structure_context
            and self.structure_context.get("section_title") == self.section_title
            and self.structure_context.get("hierarchy_header") == self.header_text
        ):
            return
        object.__setattr__(
            self,
            "structure_context",
            build_structure_context(
                section_title=self.section_title,
                header_text=self.header_text,
                page_number=self.page_number,
                line_start=self.line_start,
                line_end=self.line_end,
            ),
        )


def _is_table_header(line: str) -> bool:
    signal = detect_structure_signal(line)
    return bool(signal and signal.kind == "hierarchy_header")


def _line_number(chunk: DocumentChunk, offset: int) -> int | None:
    if chunk.line_start is None:
        return None
    return chunk.line_start + offset


def _normalize_code(value: str) -> str:
    return re.sub(r"\s+", "", value).replace("．", ".")


def _flattened_row_hints(text: str) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        detected_codes: list[str] = []
        seen_codes: set[str] = set()
        for match in HIERARCHICAL_CODE_RE.finditer(line):
            code = _normalize_code(match.group(1))
            if code in seen_codes:
                continue
            seen_codes.add(code)
            detected_codes.append(code)
        if len(detected_codes) >= 2:
            hints.append({"line_text": line.strip(), "detected_codes": detected_codes})
    return hints


def _make_segment(
    chunk: DocumentChunk,
    segment_index: int,
    block_signal: str,
    header_text: str,
    text: str,
    first_offset: int,
    last_offset: int,
) -> TableSegment:
    line_start = _line_number(chunk, first_offset)
    line_end = _line_number(chunk, last_offset)
    return TableSegment(
        segment_id=f"{chunk.chunk_id}_seg_{segment_index}",
        source_chunk_id=chunk.chunk_id,
        doc_name=chunk.doc_name,
        section_title=chunk.section_title,
        text=text,
        position=chunk.position,
        page_number=chunk.page_number,
        line_start=line_start,
        line_end=line_end,
        source_method=chunk.source_method,
        source_warning=chunk.source_warning,
        block_signal=block_signal,
        header_text=header_text,
        structure_context=build_structure_context(
            section_title=chunk.section_title,
            header_text=header_text,
            page_number=chunk.page_number,
            line_start=line_start,
            line_end=line_end,
        ),
        flattened_row_hints=_flattened_row_hints(text),
    )


def _make_segment_from_lines(
    chunk: DocumentChunk,
    segment_index: int,
    block_signal: str,
    header_text: str,
    lines: list[tuple[int, str]],
) -> TableSegment:
    return _make_segment(
        chunk=chunk,
        segment_index=segment_index,
        block_signal=block_signal,
        header_text=header_text,
        text="\n".join(line for _, line in lines),
        first_offset=lines[0][0],
        last_offset=lines[-1][0],
    )


def _append_line_length(current_lines: list[tuple[int, str]], line: str) -> int:
    return len(line) if not current_lines else len(line) + 1


def _split_chunk(chunk: DocumentChunk, block_signal: str, max_chars: int) -> list[TableSegment]:
    max_chars = max(1, int(max_chars or 1))
    lines = [(index, line) for index, line in enumerate((chunk.text or "").split("\n"))]
    segments: list[TableSegment] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    header_text = ""

    def flush() -> None:
        nonlocal current, current_chars
        if current and any(line.strip() for _, line in current):
            segments.append(
                _make_segment_from_lines(
                    chunk=chunk,
                    segment_index=len(segments) + 1,
                    block_signal=block_signal,
                    header_text=header_text,
                    lines=current,
                )
            )
            current = []
            current_chars = 0

    for offset, line in lines:
        line_is_header = _is_table_header(line)
        if line_is_header:
            if current:
                flush()
            header_text = line.strip()
            continue

        line_chars = len(line)
        if not line.strip():
            if not current:
                continue
            projected_chars = current_chars + _append_line_length(current, line)
            if projected_chars > max_chars:
                flush()
                continue
            current.append((offset, line))
            current_chars = projected_chars
            continue

        if line_chars > max_chars:
            flush()
            for start in range(0, line_chars, max_chars):
                segments.append(
                    _make_segment(
                        chunk=chunk,
                        segment_index=len(segments) + 1,
                        block_signal=block_signal,
                        header_text=header_text,
                        text=line[start : start + max_chars],
                        first_offset=offset,
                        last_offset=offset,
                    )
                )
            continue

        projected_chars = current_chars + _append_line_length(current, line)
        if current and projected_chars > max_chars:
            flush()
            projected_chars = line_chars

        current.append((offset, line))
        current_chars = projected_chars

    flush()
    return segments


def segment_table_chunks_for_row_extraction(
    chunks: list[DocumentChunk],
    block_signals: dict[str, dict],
    max_chars: int = 5000,
) -> list[TableSegment]:
    segments: list[TableSegment] = []
    for chunk in chunks:
        signal = block_signals.get(chunk.chunk_id, {})
        block_signal = str(signal.get("block_signal") or "normal")
        text_len = len(chunk.text or "")
        if not (chunk.text or "").strip():
            continue
        should_split = block_signal in {"table_like", "hierarchy_like"} or text_len > max_chars
        if should_split:
            segments.extend(_split_chunk(chunk, block_signal, max_chars))
            continue
        segments.append(
            _make_segment(
                chunk=chunk,
                segment_index=1,
                block_signal=block_signal,
                header_text="",
                text=chunk.text or "",
                first_offset=0,
                last_offset=(
                    chunk.line_end - chunk.line_start
                    if chunk.line_start is not None and chunk.line_end is not None
                    else max(0, len((chunk.text or "").splitlines()) - 1)
                ),
            )
        )
    return segments
