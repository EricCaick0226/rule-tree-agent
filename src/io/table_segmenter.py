from __future__ import annotations

from dataclasses import dataclass
import re

from ..core.agent_state import DocumentChunk


TABLE_HEADER_PATTERNS = [
    re.compile(r"类\s+项\s+目\s+数据范围及示例\s+数据加工程度\s+影响对象\s+影响程度\s+数据级别"),
    re.compile(r"资源属性\s+类.*项.*目"),
]


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


def _is_table_header(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line.strip())
    return any(pattern.search(compact) for pattern in TABLE_HEADER_PATTERNS)


def _line_number(chunk: DocumentChunk, offset: int) -> int | None:
    if chunk.line_start is None:
        return None
    return chunk.line_start + offset


def _make_segment(
    chunk: DocumentChunk,
    segment_index: int,
    block_signal: str,
    header_text: str,
    text: str,
    first_offset: int,
    last_offset: int,
) -> TableSegment:
    return TableSegment(
        segment_id=f"{chunk.chunk_id}_seg_{segment_index}",
        source_chunk_id=chunk.chunk_id,
        doc_name=chunk.doc_name,
        section_title=chunk.section_title,
        text=text,
        position=chunk.position,
        page_number=chunk.page_number,
        line_start=_line_number(chunk, first_offset),
        line_end=_line_number(chunk, last_offset),
        source_method=chunk.source_method,
        source_warning=chunk.source_warning,
        block_signal=block_signal,
        header_text=header_text,
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

        line_chars = len(line)
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
                last_offset=0
                if chunk.line_end is not None
                else max(0, len((chunk.text or "").split("\n")) - 1),
            )
        )
    return segments
