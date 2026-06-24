from __future__ import annotations

import re

from ..core.agent_state import AgentState, ClassificationRow, ClassificationSchema
from ..io.description_evidence_policy import should_reject_label_only_description
from ..llm.task_utils import append_step_trace, stable_id


INSUFFICIENT_DESCRIPTION = "证据不足，无法从当前文档确定"
DEFAULT_INSUFFICIENT_REVIEW_REASON = "当前文档未提供该分类项的说明或范围描述。"
GRADE_CONFLICT_REVIEW_REASON = "同一分类路径存在不同推荐分级候选，需要人工确认。"
GRADE_HIGH_REVIEW_REASON = "同一分类路径存在多个分级证据，已按当前文档分级顺序采用就高不就低原则。"
UNKNOWN_GRADE_ORDER_REVIEW_REASON = "同一分类路径存在多个分级证据，但无法从当前文档确定分级高低顺序，需要人工确认。"
SUFFIX_CONTEXT_MERGE_REVIEW_REASON = "通过表格上下文继承合并短路径分级候选，需要人工确认。"
SUPPORT_RANK = {"weak": 0, "structural": 1, "explicit": 2}
DESCRIPTION_SOURCE_RANK = {"insufficient": 0, "summarized": 1, "quoted": 2}
NON_CLASSIFICATION_REVIEW_MARKERS = (
    "术语定义条目",
    "术语定义/注释条目",
    "原文为术语定义段落",
)
TITLE_ONLY_REVIEW_MARKERS = (
    "仅含分类标题",
    "仅含分类编号",
    "仅含当前层级编号",
    "仅包含分类层级标题",
)
STRUCTURAL_FRAGMENT_REVIEW_MARKERS = (
    "孤立短语",
    "分级判定因素片段",
    "疑似表格表头",
    "文本断裂",
)
ROW_ROLE_CLASSIFICATION_DETAIL = "classification_detail"
ROW_ROLE_STRUCTURAL_HEADING = "structural_heading"
ROW_ROLE_DEFINITION_TERM = "definition_term"
ROW_ROLE_GRADING_CRITERION = "grading_criterion"
NON_DETAIL_ROW_ROLES = {
    ROW_ROLE_STRUCTURAL_HEADING,
    ROW_ROLE_DEFINITION_TERM,
    ROW_ROLE_GRADING_CRITERION,
}


def _row_key(row: ClassificationRow) -> tuple[str, ...]:
    return tuple(level.strip() for level in row.path_levels if level.strip())


def _row_rank(row: ClassificationRow) -> tuple[int, int, int, int, float]:
    return (
        1 if row.evidence_refs else 0,
        SUPPORT_RANK.get(row.support_level, 0),
        1 if row.description_source != "insufficient" and row.description.strip() else 0,
        DESCRIPTION_SOURCE_RANK.get(row.description_source, 0),
        row.confidence,
    )


def _merge_review_context(selected: ClassificationRow, other: ClassificationRow) -> None:
    selected.needs_review = selected.needs_review or other.needs_review
    reasons = [reason for reason in [selected.review_reason, other.review_reason] if reason]
    selected.review_reason = "；".join(dict.fromkeys(reasons))


def _append_review_reason(row: ClassificationRow, reason: str) -> None:
    reasons = [item for item in [row.review_reason, reason] if item]
    row.review_reason = "；".join(dict.fromkeys(reasons))


def _grade_rank_from_text(grade: str) -> int | None:
    match = re.search(r"(\d+)\s*级", str(grade or "").strip())
    if match:
        return int(match.group(1))
    return None


def _grade_rank(row: ClassificationRow, state: AgentState) -> int | None:
    grade_name = str(row.recommended_grade or "").strip()
    if not grade_name:
        return None

    return _grade_rank_from_text(grade_name)


def _merge_text_values(*values: str) -> str:
    parts: list[str] = []
    for value in values:
        for part in str(value or "").split("；"):
            clean = part.strip()
            if clean and clean not in parts:
                parts.append(clean)
    return "；".join(parts)


def _merge_list_values(*values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in values:
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in merged:
                merged.append(clean)
    return merged


def _evidence_ref_key(ref) -> tuple[str, str, str, str]:
    return (
        str(getattr(ref, "evidence_id", "") or ""),
        str(getattr(ref, "chunk_id", "") or ""),
        str(getattr(ref, "doc_name", "") or ""),
        str(getattr(ref, "text", "") or ""),
    )


def _merge_evidence_refs(existing, other):
    merged = list(existing)
    seen = {_evidence_ref_key(ref) for ref in merged}
    for ref in other:
        key = _evidence_ref_key(ref)
        if key not in seen:
            merged.append(ref)
            seen.add(key)
    return merged


def _normalize_row(row: ClassificationRow) -> ClassificationRow | None:
    row.path_levels = list(_row_key(row))
    if not row.path_levels:
        return None

    if row.description_source == "insufficient" or not row.description.strip():
        row.description = INSUFFICIENT_DESCRIPTION
        row.description_source = "insufficient"
        row.needs_review = True
        if not row.review_reason:
            row.review_reason = DEFAULT_INSUFFICIENT_REVIEW_REASON
    else:
        description_quote = row.description_evidence_quote or row.evidence_quote
        decision = should_reject_label_only_description(row, row.description, description_quote)
        if decision.force:
            row.description = INSUFFICIENT_DESCRIPTION
            row.description_source = "insufficient"
            row.description_evidence_quote = ""
            row.needs_review = True
            row.status = "proposed"
            _append_review_reason(row, decision.reason)

    row.row_id = stable_id(
        "row",
        " / ".join(row.path_levels) + "|" + str(row.recommended_grade or ""),
    )
    return row


def _is_non_classification_term_row(row: ClassificationRow) -> bool:
    if row.path_levels and any(marker in row.path_levels[0] for marker in ("术语", "定义")):
        return True
    review_reason = str(row.review_reason or "")
    return any(marker in review_reason for marker in NON_CLASSIFICATION_REVIEW_MARKERS)


def _is_prefix_path(prefix: tuple[str, ...], path: tuple[str, ...]) -> bool:
    return len(prefix) < len(path) and path[: len(prefix)] == prefix


def _is_insufficient_parent_title_row(
    row: ClassificationRow,
    row_keys: list[tuple[str, ...]],
) -> bool:
    key = _row_key(row)
    if not key:
        return False
    if row.description_source != "insufficient":
        return False
    if row.recommended_grade or row.data_range_examples:
        return False
    review_reason = str(row.review_reason or "")
    if not any(marker in review_reason for marker in TITLE_ONLY_REVIEW_MARKERS):
        return False
    return any(_is_prefix_path(key, other_key) for other_key in row_keys)


def _is_structural_title_or_fragment_row(row: ClassificationRow) -> bool:
    if row.description_source != "insufficient":
        return False
    if row.support_level != "structural":
        return False
    if row.recommended_grade or row.data_range_examples:
        return False
    review_reason = str(row.review_reason or "")
    return any(
        marker in review_reason
        for marker in (*TITLE_ONLY_REVIEW_MARKERS, *STRUCTURAL_FRAGMENT_REVIEW_MARKERS)
    )


def _classify_row_role(
    row: ClassificationRow,
    row_keys: list[tuple[str, ...]],
) -> str:
    existing_role = str(row.row_role or "").strip()
    if existing_role in NON_DETAIL_ROW_ROLES:
        return existing_role
    if _is_non_classification_term_row(row):
        return ROW_ROLE_DEFINITION_TERM
    if _is_insufficient_parent_title_row(row, row_keys):
        return ROW_ROLE_STRUCTURAL_HEADING
    if _is_structural_title_or_fragment_row(row):
        return ROW_ROLE_STRUCTURAL_HEADING
    return ROW_ROLE_CLASSIFICATION_DETAIL


def filter_non_classification_rows(state: AgentState) -> int:
    row_keys = [_row_key(row) for row in state.classification_rows]
    kept: list[ClassificationRow] = []
    excluded_count = 0
    for row in state.classification_rows:
        row.row_role = _classify_row_role(row, row_keys)
        if row.row_role != ROW_ROLE_CLASSIFICATION_DETAIL:
            excluded_count += 1
            continue
        kept.append(row)
    state.classification_rows = kept
    return excluded_count


def _choose_row(existing: ClassificationRow, candidate: ClassificationRow) -> ClassificationRow:
    selected = candidate if _row_rank(candidate) > _row_rank(existing) else existing
    other = existing if selected is candidate else candidate
    grade_changed = (existing.recommended_grade or "") != (candidate.recommended_grade or "")

    _merge_review_context(selected, other)
    if selected is candidate or grade_changed:
        selected.needs_review = selected.needs_review or other.needs_review or grade_changed
        if grade_changed:
            _append_review_reason(selected, GRADE_CONFLICT_REVIEW_REASON)
    return selected


def _merge_duplicate_rows(
    existing: ClassificationRow,
    candidate: ClassificationRow,
    state: AgentState,
) -> ClassificationRow:
    selected = _choose_row(existing, candidate)
    other = candidate if selected is existing else existing

    selected.description = _merge_text_values(selected.description, other.description)
    selected.description_evidence_quote = _merge_text_values(
        selected.description_evidence_quote,
        other.description_evidence_quote,
    )
    selected.evidence_quote = _merge_text_values(selected.evidence_quote, other.evidence_quote)
    selected.grade_evidence_quote = _merge_text_values(
        selected.grade_evidence_quote,
        other.grade_evidence_quote,
    )
    selected.data_range_examples = _merge_list_values(
        selected.data_range_examples,
        other.data_range_examples,
    )
    if not selected.processing_degree:
        selected.processing_degree = other.processing_degree
    if not selected.impact_object:
        selected.impact_object = other.impact_object
    if not selected.impact_degree:
        selected.impact_degree = other.impact_degree
    selected.evidence_refs = _merge_evidence_refs(selected.evidence_refs, other.evidence_refs)

    if (existing.recommended_grade or "") == (candidate.recommended_grade or ""):
        return selected

    existing_rank = _grade_rank(existing, state)
    candidate_rank = _grade_rank(candidate, state)
    if existing_rank is not None and candidate_rank is not None:
        selected.recommended_grade = (
            existing.recommended_grade
            if existing_rank >= candidate_rank
            else candidate.recommended_grade
        )
        selected.needs_review = True
        _append_review_reason(selected, GRADE_HIGH_REVIEW_REASON)
    else:
        selected.needs_review = True
        _append_review_reason(selected, UNKNOWN_GRADE_ORDER_REVIEW_REASON)
    return selected


def _is_suffix_path(short_key: tuple[str, ...], long_key: tuple[str, ...]) -> bool:
    return len(short_key) < len(long_key) and long_key[-len(short_key) :] == short_key


def _evidence_chunk_ids(row: ClassificationRow) -> set[str]:
    return {
        str(getattr(ref, "chunk_id", "") or "")
        for ref in row.evidence_refs
        if str(getattr(ref, "chunk_id", "") or "")
    }


def _has_shared_source_chunk(left: ClassificationRow, right: ClassificationRow) -> bool:
    left_chunks = _evidence_chunk_ids(left)
    right_chunks = _evidence_chunk_ids(right)
    return bool(left_chunks and right_chunks and left_chunks.intersection(right_chunks))


def _merge_suffix_grade_context(
    longer: ClassificationRow,
    shorter: ClassificationRow,
) -> ClassificationRow:
    longer.recommended_grade = shorter.recommended_grade
    longer.description = _merge_text_values(longer.description, shorter.description)
    longer.description_evidence_quote = _merge_text_values(
        longer.description_evidence_quote,
        shorter.description_evidence_quote,
    )
    longer.evidence_quote = _merge_text_values(longer.evidence_quote, shorter.evidence_quote)
    longer.grade_evidence_quote = _merge_text_values(
        longer.grade_evidence_quote,
        shorter.grade_evidence_quote,
    )
    longer.data_range_examples = _merge_list_values(
        longer.data_range_examples,
        shorter.data_range_examples,
    )
    if not longer.processing_degree:
        longer.processing_degree = shorter.processing_degree
    if not longer.impact_object:
        longer.impact_object = shorter.impact_object
    if not longer.impact_degree:
        longer.impact_degree = shorter.impact_degree
    longer.evidence_refs = _merge_evidence_refs(longer.evidence_refs, shorter.evidence_refs)
    longer.support_level = (
        shorter.support_level
        if SUPPORT_RANK.get(shorter.support_level, 0) > SUPPORT_RANK.get(longer.support_level, 0)
        else longer.support_level
    )
    longer.confidence = max(longer.confidence, shorter.confidence)
    longer.needs_review = True
    longer.status = "proposed"
    _append_review_reason(longer, SUFFIX_CONTEXT_MERGE_REVIEW_REASON)
    return longer


def _merge_suffix_rows_with_grade_context(rows: list[ClassificationRow]) -> list[ClassificationRow]:
    rows_by_key = {_row_key(row): row for row in rows}
    donor_keys: set[tuple[str, ...]] = set()

    for short_key, short_row in list(rows_by_key.items()):
        if not short_row.recommended_grade or not short_row.grade_evidence_quote:
            continue

        candidates = [
            long_row
            for long_key, long_row in rows_by_key.items()
            if _is_suffix_path(short_key, long_key)
            and not long_row.recommended_grade
            and _has_shared_source_chunk(long_row, short_row)
        ]
        if len(candidates) != 1:
            continue

        _merge_suffix_grade_context(candidates[0], short_row)
        donor_keys.add(short_key)

    return [row for row in rows if _row_key(row) not in donor_keys]


def normalize_classification_rows(state: AgentState) -> AgentState:
    before_count = len(state.classification_rows)
    deduped: dict[tuple[str, ...], ClassificationRow] = {}
    skipped_empty_path_rows = 0

    for raw_row in state.classification_rows:
        row = _normalize_row(raw_row)
        if row is None:
            skipped_empty_path_rows += 1
            continue

        key = _row_key(row)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue
        deduped[key] = _merge_duplicate_rows(existing, row, state)

    state.classification_rows = _merge_suffix_rows_with_grade_context(list(deduped.values()))
    excluded_non_classification_rows = filter_non_classification_rows(state)
    normalized = state.classification_rows
    for row in normalized:
        row.row_id = stable_id(
            "row",
            " / ".join(row.path_levels) + "|" + str(row.recommended_grade or ""),
        )

    max_depth = max((len(row.path_levels) for row in normalized), default=0)
    evidence_quote = "；".join(" / ".join(row.path_levels) for row in normalized[:3])
    state.classification_rows = normalized
    state.classification_schema = ClassificationSchema(
        max_depth=max_depth,
        source="inferred_from_rows" if max_depth else "insufficient_evidence",
        evidence_quote=evidence_quote,
        confidence=0.7 if max_depth else 0.0,
        needs_review=True,
        review_reason=(
            "未找到明确层级表头，按抽取出的路径最大深度推断。"
            if max_depth
            else "证据不足，无法从当前文档确定分类层级。"
        ),
    )
    append_step_trace(
        state.step_traces,
        "normalize_classification_rows",
        "success",
        "",
        {
            "classification_rows_before": before_count,
            "skipped_empty_path_rows": skipped_empty_path_rows,
        },
        {
            "classification_rows_after": len(state.classification_rows),
            "excluded_non_classification_rows": excluded_non_classification_rows,
            "max_depth": max_depth,
        },
    )
    return state
