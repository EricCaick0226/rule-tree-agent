from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState, GradeDefinition
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


def _payload(state: AgentState) -> dict[str, Any]:
    chunks = chunk_payload(state.chunks)
    for item in chunks:
        signal = state.block_signals.get(item["chunk_id"], {})
        item["block_signal"] = signal.get("block_signal", item.get("chunk_signal", "normal"))
        item["block_signal_reason"] = signal.get("reason", "")

    return {
        "task": "抽取文档明确给出的分级定义。",
        "document_chunks": chunks,
        "output_schema": {
            "grade_definitions": [
                {
                    "grade_name": "...",
                    "definition": "...",
                    "criteria": ["..."],
                    "evidence_quote": "...",
                    "evidence_chunk_ids": ["doc_1_chunk_1"],
                    "confidence": 0.0,
                    "needs_review": True,
                    "review_reason": "",
                    "status": "evidence_supported | proposed",
                }
            ]
        },
    }


def _grade_rank(grade: GradeDefinition) -> tuple[int, int, int, float]:
    return (
        1 if grade.evidence_refs else 0,
        1 if grade.definition else 0,
        len(grade.criteria),
        grade.confidence,
    )


def _to_grade_definitions(data: dict[str, Any], state: AgentState) -> list[GradeDefinition]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in state.chunks}
    grades_by_name: dict[str, GradeDefinition] = {}

    for item in data.get("grade_definitions") or []:
        if not isinstance(item, dict):
            continue

        grade_name = str(item.get("grade_name") or "").strip()
        definition = str(item.get("definition") or "").strip()
        if not grade_name:
            continue

        refs = refs_from_chunk_ids(
            chunk_by_id,
            item.get("evidence_chunk_ids") or [],
            f"grade_definition:{grade_name}",
            0.9,
        )
        needs_review = parse_bool(item.get("needs_review"), not bool(definition and refs))
        review_reason = str(item.get("review_reason") or "").strip()
        status = str(item.get("status") or ("proposed" if needs_review else "evidence_supported"))
        if not refs:
            needs_review = True
            if not review_reason:
                review_reason = "分级定义缺少有效证据引用。"
            if status == "evidence_supported":
                status = "proposed"
        if not definition:
            needs_review = True
            if not review_reason:
                review_reason = "分级定义缺少明确释义。"
            if status == "evidence_supported":
                status = "proposed"

        grade = GradeDefinition(
            grade_id=stable_id("grade", grade_name),
            grade_name=grade_name,
            definition=definition,
            criteria=string_list(item.get("criteria")),
            evidence_refs=refs,
            evidence_claim_ids=[],
            confidence=clamp_confidence(item.get("confidence"), 0.65),
            needs_review=needs_review,
            review_reason=review_reason,
            status=status,
        )
        existing = grades_by_name.get(grade_name)
        if existing is None or _grade_rank(grade) > _grade_rank(existing):
            grades_by_name[grade_name] = grade

    return list(grades_by_name.values())


def extract_grade_definitions_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    if not state.chunks:
        state.grade_scheme = []
        append_step_trace(
            state.step_traces,
            "extract_grade_definitions_with_llm",
            "success",
            "No chunks available for grade definition extraction.",
            {"chunks": 0},
            {"grade_definitions": 0},
        )
        return state

    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="extract_grade_definitions_with_llm",
        prompt_file="extract_grade_definitions_prompt.md",
        payload=_payload(state),
        required_keys={"grade_definitions": list},
        temperature=0.0,
        disable_thinking=True,
    )
    state.grade_scheme = _to_grade_definitions(data, state)
    append_step_trace(
        state.step_traces,
        "extract_grade_definitions_with_llm",
        "success",
        "",
        {"chunks": len(state.chunks)},
        {"grade_definitions": len(state.grade_scheme)},
        raw_response,
    )
    return state
