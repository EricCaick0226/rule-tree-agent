from __future__ import annotations

import re
from typing import Any

from ..llm.task_utils import call_llm_json


INSUFFICIENT_DESCRIPTION = "证据不足，无法从当前文档确定"
ALLOWED_GENERATED_DESCRIPTION_SOURCES = {"summarized", "insufficient"}
DESCRIPTION_GENERATION_PROMPT = "generate_classification_descriptions_prompt.md"


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _looks_like_table_row_start(line: str) -> bool:
    return bool(re.match(r"^\s*\d{3}\S+", str(line or "").strip()))


def _looks_like_context_boundary(line: str) -> bool:
    text = str(line or "").strip()
    return bool(
        re.match(r"^表[A-Z]?\d", text)
        or text.startswith("附录")
        or text in {"数据分类 数据分级", "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"}
    )


def flag_description_quality(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    path_levels = _string_list(row.get("path_levels"))
    leaf = path_levels[-1] if path_levels else ""
    description = str(row.get("description") or "").strip()
    clean_description = _clean_text(description)

    if not description or description == INSUFFICIENT_DESCRIPTION:
        flags.append("description_insufficient")
        return flags

    if leaf and clean_description == _clean_text(leaf):
        flags.append("description_equals_leaf")

    for example in _string_list(row.get("data_range_examples")):
        clean_example = _clean_text(example)
        if clean_example and clean_description == clean_example:
            flags.append("description_duplicates_data_range")
            break

    if len(clean_description) <= 4 and not re.search(r"[，。；、,.;]", description):
        flags.append("description_label_like")

    return flags


def build_context_units(text: str, window_lines: int = 3) -> list[dict[str, Any]]:
    lines = str(text or "").splitlines()
    window = max(1, int(window_lines or 1))
    units: list[dict[str, Any]] = []

    for index, line in enumerate(lines):
        if not _looks_like_table_row_start(line):
            continue
        end = index + 1
        while end < len(lines):
            next_line = lines[end]
            if _looks_like_context_boundary(next_line) or _looks_like_table_row_start(next_line):
                break
            end += 1
        row_text = "\n".join(item for item in lines[index:end] if item.strip()).strip()
        if not row_text:
            continue
        units.append(
            {
                "unit_id": f"txt_table_row_{index + 1}_{end}",
                "line_start": index + 1,
                "line_end": end,
                "text": row_text,
            }
        )

    for start in range(0, len(lines), window):
        window_text = "\n".join(line for line in lines[start : start + window] if line.strip()).strip()
        if not window_text:
            continue
        units.append(
            {
                "unit_id": f"txt_lines_{start + 1}_{min(start + window, len(lines))}",
                "line_start": start + 1,
                "line_end": min(start + window, len(lines)),
                "text": window_text,
            }
        )
    return units


def build_row_query_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add(value: object) -> None:
        term = str(value or "").strip()
        if term and term not in terms:
            terms.append(term)

    for level in _string_list(row.get("path_levels")):
        add(level)
    for example in _string_list(row.get("data_range_examples")):
        add(example)
    add(row.get("processing_degree"))
    add(row.get("impact_object"))
    add(row.get("impact_degree"))
    add(row.get("recommended_grade"))

    return terms


def _term_score(text: str, term: str) -> int:
    if not term:
        return 0
    if term in text:
        return 4 if len(term) >= 4 else 2
    compact_text = _clean_text(text)
    compact_term = _clean_text(term)
    if compact_term and compact_term in compact_text:
        return 2
    return 0


def retrieve_contexts(
    units: list[dict[str, Any]],
    query_terms: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for unit in units:
        text = str(unit.get("text") or "")
        matched_terms = [term for term in query_terms if _term_score(text, term) > 0]
        if not matched_terms:
            continue
        score = sum(_term_score(text, term) for term in matched_terms) + len(set(matched_terms))
        scored.append(
            {
                "unit_id": unit.get("unit_id", ""),
                "line_start": unit.get("line_start"),
                "line_end": unit.get("line_end"),
                "score": score,
                "matched_terms": matched_terms,
                "text": text,
            }
        )

    scored.sort(key=lambda item: (-int(item["score"]), int(item.get("line_start") or 0)))
    return scored[: max(0, int(top_k or 0))]


def _generation_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "基于检索上下文为弱分类说明生成候选说明。",
        "rows": [
            {
                "row_id": row.get("row_id", ""),
                "path": row.get("path", ""),
                "current_description": row.get("current_description", ""),
                "description_quality_flags": row.get("description_quality_flags", []),
                "query_terms": row.get("query_terms", []),
                "retrieved_contexts": row.get("retrieved_contexts", [])[:5],
            }
            for row in rows
        ],
        "description_policy": {
            "insufficient_text": INSUFFICIENT_DESCRIPTION,
            "allowed_description_sources": sorted(ALLOWED_GENERATED_DESCRIPTION_SOURCES),
            "needs_review": True,
        },
        "output_schema": {
            "description_candidates": [
                {
                    "row_id": "",
                    "proposed_description": "",
                    "description_source": "summarized | insufficient",
                    "description_evidence_quote": "",
                    "needs_review": True,
                    "review_reason": "",
                }
            ]
        },
    }


def _normalize_description_candidate(item: dict[str, Any]) -> dict[str, Any]:
    source = str(item.get("description_source") or "").strip()
    description = str(item.get("proposed_description") or "").strip()
    if source not in ALLOWED_GENERATED_DESCRIPTION_SOURCES:
        source = "insufficient" if description == INSUFFICIENT_DESCRIPTION else "summarized"
    if not description or source == "insufficient":
        source = "insufficient"
        description = INSUFFICIENT_DESCRIPTION

    return {
        "row_id": str(item.get("row_id") or "").strip(),
        "proposed_description": description,
        "description_source": source,
        "description_evidence_quote": (
            "" if source == "insufficient" else str(item.get("description_evidence_quote") or "").strip()
        ),
        "needs_review": True,
        "review_reason": str(item.get("review_reason") or "基于检索上下文总结生成，需要人工确认。").strip(),
    }


def generate_description_candidates(
    llm_client: Any,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if not rows:
        return [], ""

    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="生成 classification description candidates",
        prompt_file=DESCRIPTION_GENERATION_PROMPT,
        payload=_generation_payload(rows),
        required_keys={"description_candidates": list},
        max_tokens=3000,
        temperature=0.0,
        disable_thinking=True,
    )
    candidates = [
        _normalize_description_candidate(item)
        for item in data.get("description_candidates") or []
        if isinstance(item, dict)
    ]
    return candidates, raw_response
