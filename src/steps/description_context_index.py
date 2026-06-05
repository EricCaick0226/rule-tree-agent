from __future__ import annotations

import re
from typing import Any

from ..io.data_classification_profile import (
    build_description_query_terms,
    build_row_evidence_pack,
    contains_grade_or_risk,
    is_resource_type_definition,
    resource_type_terms_for_row,
)

DEFINITION_RE = re.compile(r"^(?:[a-z]\)\s*)?[^\s：:]{2,30}(?:类数据|数据|分类)[：:].+", re.IGNORECASE)
PARENT_HEADING_RE = re.compile(r"^\s*\d{1,2}\S{1,30}\s*$")
CHILD_HEADING_RE = re.compile(r"^\s*\d{2}\S{1,30}\s*$")
ITEM_CODE_RE = re.compile(r"(?:^|\s)(\d{3,4})(?=[^\d\-—－])")


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _contains_grade_or_risk(text: str) -> bool:
    return contains_grade_or_risk(text)


def _contains_description_signal(text: str) -> bool:
    return bool(re.search(r"[：:，。；、,.;]", text)) and not text.strip().startswith("影响程度")


def _looks_like_table_item_row(line: str) -> bool:
    return _find_item_code_match(line) is not None


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
    match = _find_item_code_match(line)
    return match.group(1) if match else ""


def _find_item_code_match(line: str) -> re.Match[str] | None:
    text = str(line or "").strip()
    for match in ITEM_CODE_RE.finditer(text):
        code = match.group(1)
        if code.startswith(("19", "20")):
            continue
        return match
    return None


def _split_code_and_name(value: str) -> tuple[str, str]:
    match = re.match(r"^\s*(\d{3,4})(?=[^\d\-—－])(.+)$", str(value or "").strip())
    if not match:
        return "", str(value or "").strip()
    return match.group(1), match.group(2).strip(" “\"")


def _inline_path_and_row_text(line: str, parent_path: list[str]) -> tuple[list[str], str]:
    match = _find_item_code_match(line)
    if not match:
        return parent_path, line
    prefix = line[: match.start()].strip()
    row_text = line[match.start() :].strip()
    if not prefix:
        return parent_path, row_text

    parts = prefix.split()
    if len(parts) >= 2 and PARENT_HEADING_RE.match(parts[-2]) and CHILD_HEADING_RE.match(parts[-1]):
        return [parts[-2], parts[-1]], row_text
    if len(parts) == 1 and CHILD_HEADING_RE.match(parts[0]) and parent_path:
        return [*parent_path[:1], parts[0]], row_text
    if len(parts) == 1 and PARENT_HEADING_RE.match(parts[0]):
        return [parts[0]], row_text
    return parent_path, row_text


def _resource_type_terms(row: dict[str, Any]) -> list[str]:
    return resource_type_terms_for_row(row)


def _is_process_definition(text: str) -> bool:
    return bool(re.search(r"实施数据分类|开展数据分类工作|按照第\d+章", text))


def _is_resource_type_definition(text: str) -> bool:
    return is_resource_type_definition(text)


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
        if _looks_like_table_item_row(line):
            parent_path, row_start = _inline_path_and_row_text(line, parent_path)
            row_lines = [row_start]
            line_end = line_number
            while index < len(lines):
                next_line = lines[index].strip()
                if (
                    not next_line
                    or next_line.startswith("表")
                    or DEFINITION_RE.match(next_line)
                    or _looks_like_table_item_row(next_line)
                    or PARENT_HEADING_RE.match(next_line)
                    or CHILD_HEADING_RE.match(next_line)
                    or _contains_grade_or_risk(next_line) and next_line.startswith("影响程度")
                ):
                    break
                row_lines.append(next_line)
                line_end = index + 1
                index += 1
            row_text = "\n".join(row_lines)
            path_hint = [*parent_path, row_start.split()[0]]
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
        if re.match(r"^\s*\d{1,2}\S+", line):
            flush_sibling_group()
            if CHILD_HEADING_RE.match(line) and parent_path:
                parent_path = [*parent_path[:1], line]
            elif PARENT_HEADING_RE.match(line):
                parent_path = [line]
            continue
        if _contains_grade_or_risk(line):
            flush_sibling_group()
            units.append(_context_unit("negative_unit", line, line_number, line_number, table_title=table_title))

    flush_sibling_group()
    return units


def _query_terms(row: dict[str, Any]) -> list[str]:
    return build_description_query_terms(row)


def _score_unit(unit: dict[str, Any], row: dict[str, Any]) -> int:
    text = str(unit.get("text") or "")
    compact_text = _clean_text(text)
    score = 0
    path_levels = _string_list(row.get("path_levels"))
    leaf = path_levels[-1] if path_levels else ""
    leaf_code, leaf_name = _split_code_and_name(leaf)
    if leaf and _clean_text(leaf) in compact_text:
        score += 80
    if leaf_name and _clean_text(leaf_name) in compact_text:
        score += 60
    if leaf_code and leaf_code == _row_code(text):
        score += 10
    for term in _query_terms(row):
        compact_term = _clean_text(term)
        if compact_term and compact_term in compact_text:
            score += 4 if len(compact_term) >= 4 else 2
    if unit.get("kind") == "table_row_unit":
        score += 6
    elif unit.get("kind") == "definition_unit":
        score += 3
        if _is_resource_type_definition(text):
            score += 8
        if _is_process_definition(text):
            score -= 20
        for term in _resource_type_terms(row):
            if f"{term}类数据" in text or term in text:
                score += 40
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
    definition_row = dict(row)
    if primary:
        definition_row["_context_table_title"] = primary[0].get("table_title", "")
    definitions = _ranked_units(units, definition_row, kind="definition_unit")
    siblings = _ranked_units(units, row, kind="sibling_group_unit")
    warnings = []
    if excluded:
        warnings.append("excluded_grade_or_risk_context")
    if not primary:
        warnings.append("missing_primary_table_row")

    context_pack = {
        "primary_contexts": primary[:top_k],
        "definition_contexts": definitions[:2],
        "sibling_contexts": siblings[:2],
        "excluded_contexts": excluded[:top_k],
        "retrieval_warnings": warnings,
    }
    context_pack["row_evidence_pack"] = build_row_evidence_pack(row, context_pack)
    return context_pack
