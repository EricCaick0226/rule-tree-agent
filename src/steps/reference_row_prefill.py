from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..core.agent_state import AgentState, ClassificationRow
from ..io.reference_rule_library import load_reference_library
from ..io.rule_table_linker import RuleTableReference
from ..llm.task_utils import append_step_trace, stable_id


INSUFFICIENT_DESCRIPTION = "证据不足，无法从当前文档确定"
REFERENCE_LIBRARY_ENV = "REFERENCE_LIBRARY_DIR"
REVIEW_CANDIDATE_REASON = "该行来自参考库，当前文档未找到基本一致的分类行，需人工确认是否应纳入。"
DIRECT_REUSE_POLICIES = {"direct"}
DIRECT_REUSE_TRUST_LEVELS = {"authoritative", "trusted"}
CURATED_DESCRIPTION_SOURCES = {"classification_standard_excel"}
CURATED_SOURCE_CONFIDENCE = {"curated_answer"}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _clean_label(value: object) -> str:
    text = str(value or "").strip()
    text = text.replace("．", ".").replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^[A-Za-z]?[、.)）.．]+", "", text)
    text = re.sub(r"^\d+(?:[.．]\d+)*", "", text)
    return text.strip("、.)）.．:-：")


def _clean_path_levels(row: Any) -> list[str]:
    return [cleaned for level in _path_levels(row) if (cleaned := _clean_label(level))]


def _path_levels(row: Any) -> list[str]:
    if isinstance(row, ClassificationRow):
        return row.path_levels
    return _string_list(row.get("path_levels") or row.get("path"))


def _path_key(row: Any) -> str:
    return "/".join(_clean_path_levels(row))


def _leaf_key(row: Any) -> str:
    levels = _clean_path_levels(row)
    return levels[-1] if levels else ""


def _parent_leaf_key(row: Any) -> str:
    levels = _clean_path_levels(row)
    return "/".join(levels[-2:]) if len(levels) >= 2 else ""


def _is_reference_path_suffix_match(current: ClassificationRow, reference_row: dict[str, Any]) -> bool:
    current_levels = _clean_path_levels(current)
    reference_levels = _clean_path_levels(reference_row)
    return (
        bool(current_levels)
        and len(current_levels) <= len(reference_levels)
        and reference_levels[-len(current_levels):] == current_levels
    )


def _code_key(row: Any) -> str:
    text = " ".join(_path_levels(row))
    match = re.search(r"\d+(?:[.．]\d+)+", text)
    return match.group(0).replace("．", ".") if match else ""


def _alias_keys(row: dict[str, Any]) -> set[str]:
    return {_clean_label(alias) for alias in _string_list(row.get("aliases")) if _clean_label(alias)}


def _reference_description(row: dict[str, Any]) -> str:
    return str(row.get("description") or "").strip()


def _is_complete_reference_row(row: dict[str, Any]) -> bool:
    description = _reference_description(row)
    if not _path_levels(row) or not description or description == INSUFFICIENT_DESCRIPTION:
        return False
    if _string_list(row.get("data_range_examples")):
        return True
    description_source = str(row.get("description_source") or "").strip()
    source_confidence = str(row.get("source_confidence") or "").strip()
    return (
        description_source in CURATED_DESCRIPTION_SOURCES
        and source_confidence in CURATED_SOURCE_CONFIDENCE
    )


def _allows_direct_reuse(reference: RuleTableReference) -> bool:
    reuse_policy = str(reference.reuse_policy or "").strip().lower()
    trust_level = str(reference.reference_trust_level or "").strip().lower()
    return (
        reuse_policy in DIRECT_REUSE_POLICIES
        and trust_level in DIRECT_REUSE_TRUST_LEVELS
    )


def _is_direct_reuse_row(reference: RuleTableReference, row: dict[str, Any]) -> bool:
    return _allows_direct_reuse(reference) and _is_complete_reference_row(row)


def _strong_match(current: ClassificationRow, reference_row: dict[str, Any]) -> dict[str, Any] | None:
    current_path = _path_key(current)
    reference_path = _path_key(reference_row)
    current_leaf = _leaf_key(current)
    reference_leaf = _leaf_key(reference_row)
    if not current_leaf or not reference_leaf:
        return None

    if current_path and current_path == reference_path:
        return {"match_type": "exact_path", "score": 1.0}

    current_code = _code_key(current)
    reference_code = _code_key(reference_row)
    if current_code and current_code == reference_code and current_leaf == reference_leaf:
        return {"match_type": "code_and_leaf", "score": 0.97}

    if (
        _parent_leaf_key(current)
        and _parent_leaf_key(current) == _parent_leaf_key(reference_row)
        and _is_reference_path_suffix_match(current, reference_row)
    ):
        return {"match_type": "parent_and_leaf", "score": 0.94}

    if current_leaf in _alias_keys(reference_row):
        return {"match_type": "exact_alias", "score": 0.92}

    return None


def _reference_match_payload(
    reference: RuleTableReference,
    reference_row: dict[str, Any],
    match: dict[str, Any],
    usage: str,
) -> dict[str, Any]:
    return {
        "reference_name": reference.name,
        "reference_type": reference.source_type,
        "reference_file": reference.path,
        "reference_row_id": str(reference_row.get("row_id") or ""),
        "reference_path": _path_levels(reference_row),
        "score": match["score"],
        "match_type": match["match_type"],
        "usage": usage,
    }


def _direct_reuse_fields(row: ClassificationRow, reference_row: dict[str, Any]) -> list[str]:
    reused: list[str] = []

    ref_path = _path_levels(reference_row)
    if ref_path and row.path_levels != ref_path:
        if not row.original_path_levels:
            row.original_path_levels = list(row.path_levels)
        row.path_levels = ref_path
        reused.append("path_levels")

    ref_description = _reference_description(reference_row)
    if ref_description and ref_description != INSUFFICIENT_DESCRIPTION:
        row.description = ref_description
        row.description_source = str(reference_row.get("description_source") or "").strip() or "reference_library"
        row.description_evidence_quote = str(reference_row.get("description_evidence_quote") or "")
        reused.append("description")

    ref_examples = _string_list(reference_row.get("data_range_examples"))
    if ref_examples:
        row.data_range_examples = ref_examples
        reused.append("data_range_examples")

    ref_data_element_refs = _string_list(reference_row.get("data_element_refs"))
    row.data_element_refs = ref_data_element_refs
    if ref_data_element_refs:
        reused.append("data_element_refs")

    for field_name in ("processing_degree", "impact_object", "impact_degree"):
        ref_value = str(reference_row.get(field_name) or "")
        setattr(row, field_name, ref_value)
        if ref_value:
            reused.append(field_name)

    if reused:
        row.content_source = "reference_library"
    return reused


def _current_row_match_keys(rows: list[ClassificationRow]) -> set[str]:
    return {_path_key(row) for row in rows if _path_key(row)}


def _candidate_from_reference(reference: RuleTableReference, row: dict[str, Any]) -> ClassificationRow:
    path_levels = _path_levels(row)
    ref_id = str(row.get("row_id") or "/".join(path_levels))
    description = _reference_description(row) or INSUFFICIENT_DESCRIPTION
    description_source = "reference_library" if description != INSUFFICIENT_DESCRIPTION else "insufficient"
    return ClassificationRow(
        row_id=stable_id("row_ref", reference.path + "|" + ref_id),
        path_levels=path_levels,
        recommended_grade=None,
        description=description,
        description_source=description_source,
        data_range_examples=_string_list(row.get("data_range_examples")),
        data_element_refs=_string_list(row.get("data_element_refs")),
        processing_degree=str(row.get("processing_degree") or ""),
        impact_object=str(row.get("impact_object") or ""),
        impact_degree=str(row.get("impact_degree") or ""),
        support_level="weak",
        confidence=0.0,
        needs_review=True,
        review_reason=REVIEW_CANDIDATE_REASON,
        status="proposed",
        row_source="reference_library",
        content_source="reference_library",
        inclusion_status="review_candidate",
        evidence_status="reference_only",
        reference_matches=[
            {
                "reference_name": reference.name,
                "reference_type": reference.source_type,
                "reference_file": reference.path,
                "reference_row_id": ref_id,
                "reference_path": path_levels,
                "score": 1.0,
                "match_type": "missing_reference_candidate",
                "usage": "review_candidate",
            }
        ],
    )


def prefill_rows_from_reference_library(
    state: AgentState,
    library_dir: str | None = None,
) -> AgentState:
    return apply_reference_row_reuse(state, library_dir=library_dir)


def apply_reference_row_reuse(
    state: AgentState,
    library_dir: str | None = None,
) -> AgentState:
    _load_dotenv_if_available()
    configured_library = (
        os.getenv(REFERENCE_LIBRARY_ENV, "").strip()
        if library_dir is None
        else str(library_dir or "").strip()
    )
    if not configured_library:
        append_step_trace(
            state.step_traces,
            "apply_reference_row_reuse",
            "skipped",
            "Reference row reuse is disabled.",
            {"enabled": False},
            {
                "direct_reused_rows": 0,
                "reused_fields": 0,
                "candidate_rows": 0,
                "classification_rows": len(state.classification_rows),
            },
        )
        return state

    references, warnings = load_reference_library(Path(configured_library))
    accepted_rows = [row for row in state.classification_rows if row.inclusion_status == "accepted"]
    direct_reused_rows = 0
    reused_fields = 0
    matched_reference_keys: set[tuple[str, str]] = set()

    for row in accepted_rows:
        best: tuple[float, RuleTableReference, dict[str, Any], dict[str, Any]] | None = None
        for reference in references:
            for reference_row in reference.rows:
                if not _is_direct_reuse_row(reference, reference_row):
                    continue
                match = _strong_match(row, reference_row)
                if not match:
                    continue
                candidate = (float(match["score"]), reference, reference_row, match)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            continue

        _score, reference, reference_row, match = best
        row.reference_matches.append(
            _reference_match_payload(
                reference,
                reference_row,
                match,
                "direct_reuse",
            )
        )
        reused = _direct_reuse_fields(row, reference_row)
        if reused:
            row.reference_prefilled_fields.extend(
                field for field in reused if field not in row.reference_prefilled_fields
            )
            direct_reused_rows += 1
            reused_fields += len(reused)
        matched_reference_keys.add((reference.path, str(reference_row.get("row_id") or "")))

    current_path_keys = _current_row_match_keys(accepted_rows)
    candidate_rows: list[ClassificationRow] = []
    for reference in references:
        for reference_row in reference.rows:
            reference_key = (reference.path, str(reference_row.get("row_id") or ""))
            if reference_key in matched_reference_keys:
                continue
            if _path_key(reference_row) in current_path_keys:
                continue
            if not _is_direct_reuse_row(reference, reference_row):
                continue
            candidate_rows.append(_candidate_from_reference(reference, reference_row))

    state.classification_rows.extend(candidate_rows)
    append_step_trace(
        state.step_traces,
        "apply_reference_row_reuse",
        "success",
        "",
        {
            "enabled": True,
            "library_dir": configured_library,
            "current_rows": len(accepted_rows),
            "references": len(references),
            "warnings": warnings,
        },
        {
            "direct_reused_rows": direct_reused_rows,
            "reused_fields": reused_fields,
            "candidate_rows": len(candidate_rows),
            "classification_rows": len(state.classification_rows),
        },
    )
    return state
