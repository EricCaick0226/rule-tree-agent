from __future__ import annotations

import re
from typing import Any

from .description_context_kb import _clean_text


GRADE_OR_RISK_PATTERNS = [
    re.compile(r"影响程度"),
    re.compile(r"数据级别"),
    re.compile(r"一般数据\d级"),
    re.compile(r"重要数据"),
    re.compile(r"核心数据"),
    re.compile(r"严重危害"),
    re.compile(r"特别严重危害"),
    re.compile(r"泄露后"),
]
DEFINITION_RE = re.compile(r"^(?:[a-z]\)\s*)?[^\s：:]{2,30}(?:类数据|数据|分类)[：:].+", re.IGNORECASE)
PARENT_HEADING_RE = re.compile(r"^\s*\d{1,2}\S{1,30}\s*$")
CHILD_HEADING_RE = re.compile(r"^\s*\d{2}\S{1,30}\s*$")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _contains_grade_or_risk(text: str) -> bool:
    return any(pattern.search(text) for pattern in GRADE_OR_RISK_PATTERNS)


def _contains_description_signal(text: str) -> bool:
    return bool(re.search(r"[：:，。；、,.;]", text)) and not text.strip().startswith("影响程度")


def _looks_like_table_item_row(line: str) -> bool:
    return bool(re.match(r"^\s*\d{3}(?!\d)\S+", str(line or "").strip()))


def _context_unit(
    kind: str,
    text: str,
    line_start: int,
    line_end: int,
    table_title: str = "",
    path_hint: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "unit_id": f"{kind}_{line_start}_{line_end}",
        "kind": kind,
        "text": text.strip(),
        "line_start": line_start,
        "line_end": line_end,
        "table_title": table_title,
        "section_title": "",
        "path_hint": path_hint or [],
        "contains_grade_signal": _contains_grade_or_risk(text),
        "contains_description_signal": _contains_description_signal(text),
        "confidence": 0.8,
    }


def _row_code(line: str) -> str:
    match = re.match(r"^\s*(\d{3})", line)
    return match.group(1) if match else ""


def build_description_context_index(text: str) -> list[dict[str, Any]]:
    lines = str(text or "").splitlines()
    units: list[dict[str, Any]] = []
    table_title = ""
    parent_path: list[str] = []
    row_buffer: list[tuple[int, str, list[str]]] = []
    index = 0

    def flush_sibling_group() -> None:
        nonlocal row_buffer
        if len(row_buffer) < 2:
            row_buffer = []
            return
        group_text = "\n".join(row_text for _line_number, row_text, _path_hint in row_buffer)
        line_start = row_buffer[0][0]
        line_end = row_buffer[-1][0]
        path_hint = row_buffer[0][2][:-1]
        units.append(
            _context_unit(
                "sibling_group_unit",
                group_text,
                line_start,
                line_end,
                table_title=table_title,
                path_hint=path_hint,
            )
        )
        row_buffer = []

    while index < len(lines):
        line_number = index + 1
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line.startswith("表"):
            table_title = line
            flush_sibling_group()
            units.append(_context_unit("title_unit", line, line_number, line_number, table_title=table_title))
            continue
        if DEFINITION_RE.match(line):
            flush_sibling_group()
            units.append(_context_unit("definition_unit", line, line_number, line_number, table_title=table_title))
            continue
        if re.match(r"^\s*\d{1,2}\S+", line) and not _looks_like_table_item_row(line):
            flush_sibling_group()
            if PARENT_HEADING_RE.match(line):
                parent_path = [line]
            elif CHILD_HEADING_RE.match(line):
                parent_path = [*parent_path[:1], line]
            continue
        if _looks_like_table_item_row(line):
            row_lines = [line]
            line_end = line_number
            while index < len(lines):
                next_line = lines[index].strip()
                if (
                    not next_line
                    or next_line.startswith("表")
                    or DEFINITION_RE.match(next_line)
                    or _looks_like_table_item_row(next_line)
                    or re.match(r"^\s*\d{1,2}\S+", next_line)
                    or _contains_grade_or_risk(next_line) and next_line.startswith("影响程度")
                ):
                    break
                row_lines.append(next_line)
                line_end = index + 1
                index += 1
            row_text = "\n".join(row_lines)
            path_hint = [*parent_path, line.split()[0]]
            units.append(
                _context_unit(
                    "table_row_unit",
                    row_text,
                    line_number,
                    line_end,
                    table_title=table_title,
                    path_hint=path_hint,
                )
            )
            row_buffer.append((line_number, row_text, path_hint))
            continue
        if _contains_grade_or_risk(line):
            flush_sibling_group()
            units.append(_context_unit("negative_unit", line, line_number, line_number, table_title=table_title))

    flush_sibling_group()
    return units


def _query_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add(value: object) -> None:
        text = str(value or "").strip()
        if text and text not in terms:
            terms.append(text)

    for level in _string_list(row.get("path_levels")):
        add(level)
    for example in _string_list(row.get("data_range_examples")):
        add(example)
    add(row.get("processing_degree"))
    add(row.get("impact_object"))
    return terms


def _score_unit(unit: dict[str, Any], row: dict[str, Any]) -> int:
    text = str(unit.get("text") or "")
    compact_text = _clean_text(text)
    score = 0
    path_levels = _string_list(row.get("path_levels"))
    leaf = path_levels[-1] if path_levels else ""
    if leaf and (_clean_text(leaf) in compact_text or _row_code(leaf) in compact_text):
        score += 20
    for term in _query_terms(row):
        compact_term = _clean_text(term)
        if compact_term and compact_term in compact_text:
            score += 4 if len(compact_term) >= 4 else 2
    if unit.get("kind") == "table_row_unit":
        score += 6
    elif unit.get("kind") == "definition_unit":
        score += 3
    elif unit.get("kind") == "sibling_group_unit":
        score += 4
    return score


def _ranked_units(units: list[dict[str, Any]], row: dict[str, Any], kind: str | None = None) -> list[dict[str, Any]]:
    candidates = [unit for unit in units if kind is None or unit.get("kind") == kind]
    scored: list[dict[str, Any]] = []
    for unit in candidates:
        score = _score_unit(unit, row)
        if score <= 0 and kind != "definition_unit":
            continue
        item = dict(unit)
        item["score"] = score
        scored.append(item)
    scored.sort(key=lambda item: (-int(item.get("score") or 0), int(item.get("line_start") or 0)))
    return scored


def retrieve_description_context_pack(
    row: dict[str, Any],
    units: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    excluded = [
        dict(unit, score=_score_unit(unit, row))
        for unit in units
        if unit.get("kind") == "negative_unit"
    ]
    primary = [unit for unit in _ranked_units(units, row) if unit.get("kind") == "table_row_unit"]
    definitions = _ranked_units(units, row, kind="definition_unit")
    siblings = _ranked_units(units, row, kind="sibling_group_unit")
    warnings = []
    if excluded:
        warnings.append("excluded_grade_or_risk_context")
    if not primary:
        warnings.append("missing_primary_table_row")

    return {
        "primary_contexts": primary[:top_k],
        "definition_contexts": definitions[:2],
        "sibling_contexts": siblings[:2],
        "excluded_contexts": excluded[:top_k],
        "retrieval_warnings": warnings,
    }
