from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from src.io.document_structure import detect_structure_signal


@dataclass(frozen=True)
class CleanedLineMapping:
    clean_line_number: int
    source_line_start: int
    source_line_end: int
    transform: str


@dataclass(frozen=True)
class CleanResult:
    text: str
    mapping: list[CleanedLineMapping]
    stats: dict[str, int]
    review_items: list[dict[str, Any]]


_PAGE_FOOTER_RE = re.compile(r"^\s*[-－—–]\s*\d{1,4}\s*[-－—–]\s*$")
_STANDALONE_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_CHINESE_SINGLE_CHAR_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]$")
_ITEM_CODE_RE = re.compile(r"^\s*\d{3,4}(?:\s|$)")
_DOTTED_CODE_RE = re.compile(r"^\s*\d+(?:[.．]\d+)+(?:\s|$)")
_CLASSIFICATION_ROW_RE = re.compile(r"^\s*\d{1,2}\s+\S.{2,}\s+\d{1,3}(?:\s|$)")
_POLICY_HEADING_RE = re.compile(r"^\s*\d{1,2}\s+[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaffA-Za-z][^\d]{0,24}$")
_ROW_CONTINUATION_TERMS_RE = re.compile(
    r"(?:原始数据|衍生数据|个人|组织|严重危害|一般危害|轻微危害|数据\d*级|一般数据|重要数据|核心数据|姓名|身份证|门诊号|住院号)"
)


def clean_wps_txt_file(input_path: Path, output_path: Path, review_path: Path | None = None) -> CleanResult:
    text = input_path.read_text(encoding="utf-8")
    result = clean_wps_txt_text(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_with_final_newline(result.text), encoding="utf-8")
    if review_path is not None:
        write_review_json(result, review_path)
    return result


def write_review_json(result: CleanResult, review_path: Path) -> None:
    payload = {
        "stats": result.stats,
        "mapping": [
            {
                "clean_line_number": item.clean_line_number,
                "source_line_start": item.source_line_start,
                "source_line_end": item.source_line_end,
                "transform": item.transform,
            }
            for item in result.mapping
        ],
        "review_items": result.review_items,
    }
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_wps_txt_text(text: str) -> CleanResult:
    stats = {
        "removed_page_noise_lines": 0,
        "merged_single_char_lines": 0,
        "merged_wrapped_rows": 0,
        "merged_wrapped_sentences": 0,
    }
    review_items: list[dict[str, Any]] = []
    normalized_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if normalized_lines and normalized_lines[-1] == "":
        normalized_lines = normalized_lines[:-1]

    first_pass_lines: list[str] = []
    first_pass_mappings: list[CleanedLineMapping] = []
    index = 0
    while index < len(normalized_lines):
        source_line_number = index + 1
        line = normalized_lines[index].rstrip()
        stripped = line.strip()

        if _is_page_noise(stripped):
            stats["removed_page_noise_lines"] += 1
            review_items.append(
                {
                    "kind": "removed_page_noise",
                    "source_line_start": source_line_number,
                    "source_line_end": source_line_number,
                    "text": stripped,
                }
            )
            index += 1
            continue

        if _is_single_chinese_char(stripped):
            chars = [stripped]
            start_line = source_line_number
            end_index = index + 1
            while end_index < len(normalized_lines):
                next_stripped = normalized_lines[end_index].rstrip().strip()
                if not _is_single_chinese_char(next_stripped):
                    break
                chars.append(next_stripped)
                end_index += 1

            if len(chars) > 1:
                first_pass_lines.append("".join(chars))
                first_pass_mappings.append(
                    CleanedLineMapping(
                        clean_line_number=len(first_pass_lines),
                        source_line_start=start_line,
                        source_line_end=end_index,
                        transform="merge_single_char",
                    )
                )
                stats["merged_single_char_lines"] += 1
                index = end_index
                continue

        first_pass_lines.append(line)
        first_pass_mappings.append(
            CleanedLineMapping(
                clean_line_number=len(first_pass_lines),
                source_line_start=source_line_number,
                source_line_end=source_line_number,
                transform="preserve",
            )
        )
        index += 1

    final_lines: list[str] = []
    final_mappings: list[CleanedLineMapping] = []
    active_row_merge = False
    active_sentence_merge = False

    for line, mapping in zip(first_pass_lines, first_pass_mappings, strict=True):
        stripped = line.strip()
        if not final_lines or stripped == "":
            final_lines.append(line)
            final_mappings.append(_renumber_mapping(mapping, len(final_lines)))
            active_row_merge = False
            active_sentence_merge = False
            continue

        previous = final_lines[-1]
        previous_mapping = final_mappings[-1]
        if _should_merge_wrapped_row(previous, stripped):
            final_lines[-1] = _join_lines(previous, stripped)
            final_mappings[-1] = CleanedLineMapping(
                clean_line_number=previous_mapping.clean_line_number,
                source_line_start=previous_mapping.source_line_start,
                source_line_end=mapping.source_line_end,
                transform="merge_wrapped_row",
            )
            if not active_row_merge:
                stats["merged_wrapped_rows"] += 1
                active_row_merge = True
            active_sentence_merge = False
            continue

        if _should_merge_wrapped_sentence(previous, stripped):
            final_lines[-1] = _join_lines(previous, stripped)
            final_mappings[-1] = CleanedLineMapping(
                clean_line_number=previous_mapping.clean_line_number,
                source_line_start=previous_mapping.source_line_start,
                source_line_end=mapping.source_line_end,
                transform="merge_wrapped_sentence",
            )
            if not active_sentence_merge:
                stats["merged_wrapped_sentences"] += 1
                active_sentence_merge = True
            active_row_merge = False
            continue

        final_lines.append(line)
        final_mappings.append(_renumber_mapping(mapping, len(final_lines)))
        active_row_merge = False
        active_sentence_merge = False

    return CleanResult(
        text="\n".join(final_lines),
        mapping=final_mappings,
        stats=stats,
        review_items=review_items,
    )


def _is_page_noise(stripped: str) -> bool:
    if not stripped:
        return False
    return bool(_PAGE_FOOTER_RE.match(stripped) or _STANDALONE_PAGE_NUMBER_RE.match(stripped))


def _is_single_chinese_char(stripped: str) -> bool:
    return bool(_CHINESE_SINGLE_CHAR_RE.match(stripped))


def _is_structure_line(stripped: str) -> bool:
    return detect_structure_signal(stripped) is not None


def _is_policy_heading(stripped: str) -> bool:
    if not _POLICY_HEADING_RE.match(stripped):
        return False
    return not bool(re.search(r"\s+\d{1,4}(?:\s|$)", stripped))


def _is_row_start(stripped: str) -> bool:
    if _is_policy_heading(stripped):
        return False
    return bool(
        _ITEM_CODE_RE.match(stripped)
        or _DOTTED_CODE_RE.match(stripped)
        or _CLASSIFICATION_ROW_RE.match(stripped)
    )


def _is_row_like(line: str) -> bool:
    stripped = line.strip()
    return _is_row_start(stripped) or bool(_ROW_CONTINUATION_TERMS_RE.search(stripped))


def _should_merge_wrapped_row(previous: str, current: str) -> bool:
    if not current or _is_structure_line(current) or _is_policy_heading(current) or _is_row_start(current):
        return False
    return _is_row_like(previous)


def _should_merge_wrapped_sentence(previous: str, current: str) -> bool:
    return False


def _join_lines(previous: str, current: str) -> str:
    return f"{previous.rstrip()} {current.strip()}"


def _renumber_mapping(mapping: CleanedLineMapping, clean_line_number: int) -> CleanedLineMapping:
    if mapping.clean_line_number == clean_line_number:
        return mapping
    return CleanedLineMapping(
        clean_line_number=clean_line_number,
        source_line_start=mapping.source_line_start,
        source_line_end=mapping.source_line_end,
        transform=mapping.transform,
    )


def _with_final_newline(text: str) -> str:
    if text.endswith("\n"):
        return text
    return text + "\n"
