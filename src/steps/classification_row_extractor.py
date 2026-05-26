from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState, ClassificationRow
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    chunk_payload,
    clamp_confidence,
    parse_bool,
    refs_from_chunk_ids,
    stable_id,
    string_list,
)


ALLOWED_SUPPORT_LEVELS = {"explicit", "structural", "weak"}
ALLOWED_DESCRIPTION_SOURCES = {"quoted", "summarized", "insufficient"}
INSUFFICIENT_DESCRIPTION = "证据不足，无法从当前文档确定"
SUPPORT_RANK = {"weak": 0, "structural": 1, "explicit": 2}
DESCRIPTION_SOURCE_RANK = {"insufficient": 0, "summarized": 1, "quoted": 2}


def _payload(state: AgentState) -> dict[str, Any]:
    chunks = chunk_payload(state.chunks)
    for item in chunks:
        signal = state.block_signals.get(item["chunk_id"], {})
        item["block_signal"] = signal.get("block_signal", item.get("chunk_signal", "normal"))
        item["block_signal_reason"] = signal.get("reason", "")

    return {
        "task": "从文档 chunk 中抽取候选分类分级明细行。",
        "document_chunks": chunks,
        "allowed_support_levels": sorted(ALLOWED_SUPPORT_LEVELS),
        "description_policy": {
            "insufficient_text": INSUFFICIENT_DESCRIPTION,
            "allowed_description_sources": sorted(ALLOWED_DESCRIPTION_SOURCES),
        },
        "output_schema": {
            "classification_rows": [
                {
                    "path_levels": [],
                    "recommended_grade": None,
                    "description": "",
                    "description_source": "quoted | summarized | insufficient",
                    "description_evidence_quote": "",
                    "evidence_quote": "",
                    "evidence_chunk_ids": [],
                    "support_level": "explicit | structural | weak",
                    "confidence": 0.0,
                    "needs_review": True,
                    "review_reason": "",
                    "status": "evidence_supported | proposed | insufficient_evidence",
                }
            ]
        },
    }


def _to_rows(data: dict[str, Any], state: AgentState) -> list[ClassificationRow]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in state.chunks}
    rows_by_fingerprint: dict[str, ClassificationRow] = {}

    for item in data.get("classification_rows") or []:
        if not isinstance(item, dict):
            continue

        path_levels = string_list(item.get("path_levels"))
        if not path_levels:
            continue

        recommended_grade_value = item.get("recommended_grade")
        recommended_grade = str(recommended_grade_value).strip() if recommended_grade_value else None

        raw_description = str(item.get("description") or "").strip()
        description = raw_description or INSUFFICIENT_DESCRIPTION
        description_source = str(item.get("description_source") or "").strip()
        if not raw_description or description == INSUFFICIENT_DESCRIPTION:
            description_source = "insufficient"
        if description_source not in ALLOWED_DESCRIPTION_SOURCES:
            description_source = "insufficient" if description == INSUFFICIENT_DESCRIPTION else "summarized"
        if description_source == "insufficient":
            description = INSUFFICIENT_DESCRIPTION

        support_level = str(item.get("support_level") or "").strip()
        if support_level not in ALLOWED_SUPPORT_LEVELS:
            support_level = "weak"

        fingerprint = " / ".join(path_levels) + "|" + str(recommended_grade or "")
        refs = refs_from_chunk_ids(
            chunk_by_id,
            item.get("evidence_chunk_ids") or [],
            "classification_row:" + " / ".join(path_levels),
            0.9,
        )
        needs_review = (
            description_source == "insufficient"
            or not refs
            or support_level in {"structural", "weak"}
            or recommended_grade is None
            or parse_bool(item.get("needs_review"), False)
        )
        review_reason = str(item.get("review_reason") or "").strip()
        if description_source == "insufficient" and not review_reason:
            review_reason = "当前文档未提供该分类项的说明或范围描述。"
        if not refs and not review_reason:
            review_reason = "分类行缺少有效证据引用。"
        status = str(item.get("status") or ("proposed" if needs_review else "evidence_supported"))
        if needs_review and status != "insufficient_evidence":
            status = "proposed"

        row = ClassificationRow(
            row_id=stable_id("row", fingerprint),
            path_levels=path_levels,
            recommended_grade=recommended_grade,
            description=description,
            description_source=description_source,
            description_evidence_quote=str(item.get("description_evidence_quote") or "").strip(),
            evidence_quote=str(item.get("evidence_quote") or "").strip(),
            evidence_refs=refs,
            support_level=support_level,
            confidence=clamp_confidence(item.get("confidence"), 0.65),
            needs_review=needs_review,
            review_reason=review_reason,
            status=status,
        )
        existing = rows_by_fingerprint.get(fingerprint)
        if existing is None or _row_rank(row) > _row_rank(existing):
            rows_by_fingerprint[fingerprint] = row

    return list(rows_by_fingerprint.values())


def _row_rank(row: ClassificationRow) -> tuple[int, int, int, float]:
    return (
        1 if row.evidence_refs else 0,
        DESCRIPTION_SOURCE_RANK.get(row.description_source, 0),
        SUPPORT_RANK.get(row.support_level, 0),
        row.confidence,
    )


def extract_classification_rows_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    if not state.chunks:
        state.classification_rows = []
        append_step_trace(
            state.step_traces,
            "extract_classification_rows_with_llm",
            "success",
            "No chunks available for row extraction.",
            {"chunks": 0},
            {"classification_rows": 0},
        )
        return state

    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="extract_classification_rows_with_llm",
        prompt_file="extract_classification_rows_prompt.md",
        payload=_payload(state),
        required_keys={"classification_rows": list},
        temperature=0.0,
        disable_thinking=True,
    )
    state.classification_rows = _to_rows(data, state)
    append_step_trace(
        state.step_traces,
        "extract_classification_rows_with_llm",
        "success",
        "",
        {"chunks": len(state.chunks)},
        {"classification_rows": len(state.classification_rows)},
        raw_response,
    )
    return state
