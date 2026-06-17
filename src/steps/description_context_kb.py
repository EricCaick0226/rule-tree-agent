from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..core.agent_state import AgentState, EvidenceRef
from ..io.data_classification_profile import build_description_query_terms
from ..io.description_evidence_policy import (
    should_force_insufficient_description,
    should_reject_label_only_description,
    should_reject_summarized_description,
)
from ..llm.task_utils import append_step_trace, call_llm_json, env_int, stable_id
from .description_context_index import build_description_context_index, retrieve_description_context_pack


INSUFFICIENT_DESCRIPTION = "证据不足，无法从当前文档确定"
ALLOWED_GENERATED_DESCRIPTION_SOURCES = {"summarized", "insufficient"}
DESCRIPTION_GENERATION_PROMPT = "generate_classification_descriptions_prompt.md"
DESCRIPTION_CONTEXT_REPORT = "description_context_report.json"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    _load_dotenv_if_available()
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_context_mode() -> str:
    _load_dotenv_if_available()
    mode = os.getenv("DESCRIPTION_CONTEXT_MODE", "v1").strip().lower()
    return mode if mode in {"v1", "v2"} else "v1"


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _insufficient_description_candidate(row_id: str, reason: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "proposed_description": INSUFFICIENT_DESCRIPTION,
        "description_source": "insufficient",
        "description_evidence_quote": "",
        "needs_review": True,
        "review_reason": reason,
    }


def _apply_insufficient_description(row: ClassificationRow, reason: str) -> None:
    row.description = INSUFFICIENT_DESCRIPTION
    row.description_source = "insufficient"
    row.description_evidence_quote = ""
    row.needs_review = True
    _append_review_reason(row, reason or "缺少可支撑分类说明的行级证据。")


def _apply_label_only_description_policy(state: AgentState) -> int:
    changed = 0
    for row in state.classification_rows:
        if row.description_source == "reference_library":
            continue
        if row.description_source == "insufficient" or not str(row.description or "").strip():
            continue
        description_quote = row.description_evidence_quote or row.evidence_quote
        decision = should_reject_label_only_description(row, row.description, description_quote)
        if not decision.force:
            continue
        _apply_insufficient_description(row, decision.reason)
        changed += 1
    return changed


def _should_skip_description_generation(row: Any) -> bool:
    return bool(
        getattr(row, "description_source", "") == "reference_library"
        or getattr(row, "evidence_status", "") == "reference_only"
        or getattr(row, "inclusion_status", "") == "review_candidate"
    )


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


def _flatten_context_pack(context_pack: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_pack = context_pack.get("row_evidence_pack") or {}
    description_sources = evidence_pack.get("description_sources") or []
    if description_sources:
        sources = [source for source in description_sources if isinstance(source, dict)]

        def to_context(source: dict[str, Any]) -> dict[str, Any]:
            return {
                "unit_id": source.get("unit_id", ""),
                "kind": source.get("source_type", ""),
                "context_group": "description_sources",
                "line_start": source.get("line_start"),
                "line_end": source.get("line_end"),
                "score": source.get("score", 0),
                "matched_terms": [],
                "text": source.get("text", ""),
            }

        def add_source(target: list[dict[str, Any]], source: dict[str, Any] | None) -> None:
            if not source or not source.get("text"):
                return
            text_key = _clean_text(source.get("text"))
            if any(_clean_text(item.get("text")) == text_key for item in target):
                return
            target.append(to_context(source))

        row_paths = [
            source
            for source in sources
            if source.get("source_type") == "row_field" and source.get("role") == "classification_path"
        ]
        row_descriptions = [
            source
            for source in sources
            if source.get("source_type") == "row_field" and source.get("role") == "description_evidence"
        ]
        definitions = [source for source in sources if source.get("source_type") == "definition_unit"]
        selected: list[dict[str, Any]] = []

        if row_paths:
            add_source(selected, row_paths[0])
            add_source(selected, row_paths[-1])
        if row_descriptions:
            add_source(selected, row_descriptions[0])
        if definitions:
            add_source(selected, definitions[0])
        for source in sources:
            if len(selected) >= 5:
                break
            add_source(selected, source)
        return selected[:5]

    contexts = []
    for group in ["primary_contexts", "definition_contexts", "sibling_contexts"]:
        for context in context_pack.get(group) or []:
            if not isinstance(context, dict):
                continue
            contexts.append(
                {
                    "unit_id": context.get("unit_id", ""),
                    "kind": context.get("kind", ""),
                    "context_group": group,
                    "line_start": context.get("line_start"),
                    "line_end": context.get("line_end"),
                    "score": context.get("score", 0),
                    "matched_terms": context.get("matched_terms", []),
                    "text": context.get("text", ""),
                }
            )
    return contexts


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


def generate_description_candidates_batched(
    llm_client: Any,
    rows: list[dict[str, Any]],
    batch_size: int = 20,
) -> tuple[list[dict[str, Any]], str]:
    if not rows:
        return [], ""

    size = max(1, int(batch_size or 1))
    all_candidates: list[dict[str, Any]] = []
    raw_responses: list[str] = []
    for start in range(0, len(rows), size):
        batch = rows[start : start + size]
        candidates, raw_response = generate_description_candidates(llm_client, batch)
        all_candidates.extend(candidates)
        if raw_response:
            batch_number = start // size + 1
            raw_responses.append(f"--- batch {batch_number} ---\n{raw_response}")
    return all_candidates, "\n\n".join(raw_responses)


def _state_text(state: AgentState) -> str:
    parts: list[str] = []
    for document in state.documents:
        if document.raw_text:
            parts.append(document.raw_text)
        for page in document.pages:
            if page.text:
                parts.append(page.text)
    if not parts:
        parts.extend(chunk.text for chunk in state.chunks if chunk.text)
    return "\n".join(parts)


def _row_to_report_row(row, units: list[dict[str, Any]]) -> dict[str, Any] | None:
    if _should_skip_description_generation(row):
        return None
    flags = flag_description_quality(
        {
            "path_levels": row.path_levels,
            "description": row.description,
            "data_range_examples": row.data_range_examples,
        }
    )
    if not flags:
        return None
    row_dict = {
        "path_levels": row.path_levels,
        "data_range_examples": row.data_range_examples,
        "processing_degree": row.processing_degree,
        "impact_object": row.impact_object,
        "impact_degree": row.impact_degree,
        "recommended_grade": row.recommended_grade,
    }
    query_terms = build_row_query_terms(row_dict)
    return {
        "row_id": row.row_id,
        "path": " / ".join(row.path_levels),
        "current_description": row.description,
        "description_quality_flags": flags,
        "query_terms": query_terms,
        "retrieved_contexts": retrieve_contexts(units, query_terms, top_k=5),
    }


def _row_to_v2_report_row(row, units: list[dict[str, Any]]) -> dict[str, Any] | None:
    if _should_skip_description_generation(row):
        return None
    flags = flag_description_quality(
        {
            "path_levels": row.path_levels,
            "description": row.description,
            "data_range_examples": row.data_range_examples,
        }
    )
    if not flags:
        return None
    row_dict = {
        "path_levels": row.path_levels,
        "data_range_examples": row.data_range_examples,
        "processing_degree": row.processing_degree,
        "impact_object": row.impact_object,
        "impact_degree": row.impact_degree,
        "recommended_grade": row.recommended_grade,
    }
    context_pack = retrieve_description_context_pack(row_dict, units, top_k=5)
    return {
        "row_id": row.row_id,
        "path": " / ".join(row.path_levels),
        "current_description": row.description,
        "data_range_examples": row.data_range_examples,
        "description_quality_flags": flags,
        "query_terms": build_description_query_terms(row_dict),
        "context_pack": context_pack,
        "retrieved_contexts": _flatten_context_pack(context_pack),
    }


def _row_priority(row) -> tuple[int, int, int]:
    return (
        1 if row.recommended_grade else 0,
        1 if row.data_range_examples else 0,
        len(row.path_levels),
    )


def _append_review_reason(row, reason: str) -> None:
    reasons = [item for item in [row.review_reason, reason] if item]
    row.review_reason = "；".join(dict.fromkeys(reasons))


def _context_ref_for_candidate(
    state: AgentState,
    row_id: str,
    quote: str,
    report_row: dict[str, Any],
) -> EvidenceRef | None:
    quote = str(quote or "").strip()
    if not quote:
        return None
    contexts = report_row.get("retrieved_contexts") or []
    matching_context = next(
        (
            context
            for context in contexts
            if quote in str(context.get("text") or "")
            or _clean_text(quote) in _clean_text(context.get("text") or "")
        ),
        contexts[0] if contexts else None,
    )
    if not matching_context:
        return None
    doc = state.documents[0] if state.documents else None
    return EvidenceRef(
        evidence_id=stable_id("evidence", row_id + "|" + quote),
        chunk_id=str(matching_context.get("unit_id") or "description_context"),
        doc_name=doc.doc_name if doc else "",
        section_title="description_context",
        text=str(matching_context.get("text") or quote),
        used_for="classification_description",
        relevance_score=0.85,
        page_number=None,
        source_method="text",
        source_warning="generated from description context retrieval",
    )


def _write_description_context_report(output_dir: str, report: dict[str, Any]) -> str:
    path = Path(output_dir).expanduser().resolve() / DESCRIPTION_CONTEXT_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def enhance_descriptions_with_context(
    state: AgentState,
    llm_client: Any,
    output_dir: str = "outputs",
) -> AgentState:
    _load_dotenv_if_available()
    enabled = _env_bool("DESCRIPTION_CONTEXT_ENABLED", False)
    context_mode = _env_context_mode()
    limit = env_int("DESCRIPTION_CONTEXT_LIMIT", 20)
    batch_size = env_int("DESCRIPTION_CONTEXT_BATCH_SIZE", 20)

    if not enabled:
        append_step_trace(
            state.step_traces,
            "enhance_descriptions_with_context",
            "skipped",
            "Description context enhancement is disabled.",
            {"enabled": False, "context_mode": context_mode},
            {"enhanced_rows": 0},
        )
        return state

    source_text = _state_text(state)
    units = (
        build_description_context_index(source_text)
        if context_mode == "v2"
        else build_context_units(source_text, window_lines=3)
    )
    candidates: list[tuple[tuple[int, int, int], Any, dict[str, Any]]] = []
    for row in state.classification_rows:
        report_row = _row_to_v2_report_row(row, units) if context_mode == "v2" else _row_to_report_row(row, units)
        if report_row is not None:
            candidates.append((_row_priority(row), row, report_row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: max(0, limit)]
    report_rows = [report_row for _priority, _row, report_row in selected]
    report: dict[str, Any] = {
        "enabled": True,
        "context_mode": context_mode,
        "total_row_count": len(state.classification_rows),
        "context_unit_count": len(units),
        "sampled_row_count": len(report_rows),
        "generation": {"status": "not_requested", "candidate_count": 0},
        "rows": report_rows,
    }

    if not report_rows:
        label_only_rejections = _apply_label_only_description_policy(state)
        report_path = _write_description_context_report(output_dir, report)
        append_step_trace(
            state.step_traces,
            "enhance_descriptions_with_context",
            "success",
            "No weak descriptions found.",
            {"enabled": True, "context_mode": context_mode, "limit": limit, "context_units": len(units)},
            {
                "enhanced_rows": 0,
                "label_only_rejections": label_only_rejections,
                "report_path": report_path,
            },
        )
        return state

    try:
        generated_candidates, raw_response = generate_description_candidates_batched(
            llm_client,
            report_rows,
            batch_size=batch_size,
        )
    except Exception as exc:
        label_only_rejections = _apply_label_only_description_policy(state)
        report["generation"] = {
            "status": "failed",
            "error": str(exc),
            "candidate_count": 0,
            "raw_response_excerpt": "",
        }
        report_path = _write_description_context_report(output_dir, report)
        append_step_trace(
            state.step_traces,
            "enhance_descriptions_with_context",
            "error",
            str(exc),
            {
                "enabled": True,
                "context_mode": context_mode,
                "limit": limit,
                "batch_size": batch_size,
                "context_units": len(units),
                "sampled_rows": len(report_rows),
            },
            {
                "enhanced_rows": 0,
                "label_only_rejections": label_only_rejections,
                "report_path": report_path,
            },
        )
        return state

    generated_by_row_id = {
        candidate.get("row_id", ""): candidate
        for candidate in generated_candidates
        if candidate.get("row_id")
    }
    for _priority, row, _report_row in selected:
        decision = should_force_insufficient_description(row)
        if decision.force:
            generated_by_row_id[row.row_id] = _insufficient_description_candidate(
                row.row_id,
                decision.reason,
            )
    report_by_row_id = {
        report_row.get("row_id", ""): report_row
        for report_row in report_rows
    }
    enhanced_rows = 0
    for _priority, row, _report_row in selected:
        candidate = generated_by_row_id.get(row.row_id)
        if not candidate:
            continue
        if candidate.get("description_source") == "insufficient":
            _apply_insufficient_description(
                row,
                str(candidate.get("review_reason") or "缺少可支撑分类说明的行级证据。"),
            )
            continue
        if candidate.get("description_source") != "summarized":
            continue
        proposed_description = str(candidate.get("proposed_description") or "").strip()
        if not proposed_description or proposed_description == INSUFFICIENT_DESCRIPTION:
            continue
        description_evidence_quote = str(candidate.get("description_evidence_quote") or "").strip()
        rejection = should_reject_summarized_description(
            row,
            proposed_description,
            description_evidence_quote,
        )
        if rejection.force:
            generated_by_row_id[row.row_id] = _insufficient_description_candidate(
                row.row_id,
                rejection.reason,
            )
            _apply_insufficient_description(row, rejection.reason)
            continue

        row.description = proposed_description
        row.description_source = "summarized"
        row.description_evidence_quote = description_evidence_quote
        row.needs_review = True
        _append_review_reason(
            row,
            str(candidate.get("review_reason") or "基于检索上下文总结生成，需要人工确认。"),
        )
        context_ref = _context_ref_for_candidate(
            state,
            row.row_id,
            row.description_evidence_quote,
            report_by_row_id.get(row.row_id, {}),
        )
        if context_ref is not None:
            existing_keys = {(ref.evidence_id, ref.text) for ref in row.evidence_refs}
            if (context_ref.evidence_id, context_ref.text) not in existing_keys:
                row.evidence_refs.append(context_ref)
        enhanced_rows += 1

    label_only_rejections = _apply_label_only_description_policy(state)

    for report_row in report_rows:
        report_row["generated_description"] = generated_by_row_id.get(report_row.get("row_id", ""), {})

    report["generation"] = {
        "status": "success",
        "candidate_count": len(generated_candidates),
        "batch_size": batch_size,
        "batch_count": (len(report_rows) + max(1, batch_size) - 1) // max(1, batch_size),
        "raw_response_excerpt": raw_response[:2000],
    }
    report_path = _write_description_context_report(output_dir, report)
    append_step_trace(
        state.step_traces,
        "enhance_descriptions_with_context",
        "success",
        "",
        {
            "enabled": True,
            "context_mode": context_mode,
            "limit": limit,
            "batch_size": batch_size,
            "context_units": len(units),
            "sampled_rows": len(report_rows),
        },
        {
            "enhanced_rows": enhanced_rows,
            "label_only_rejections": label_only_rejections,
            "report_path": report_path,
        },
    )
    return state
