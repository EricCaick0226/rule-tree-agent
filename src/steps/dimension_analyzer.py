from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState, ClassificationDimension
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    claim_payload,
    clamp_confidence,
    count_claim_types,
    env_int,
    filter_claims_for_stage,
    parse_bool,
    refs_from_claim_ids,
    stable_id,
    valid_claim_ids,
)


def _concept_payload(state: AgentState) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": concept.concept_id,
            "name": concept.name,
            "aliases": concept.aliases,
            "definitions": concept.definitions,
            "included_items": concept.included_items,
            "excluded_items": concept.excluded_items,
            "related_claim_ids": concept.related_claim_ids,
        }
        for concept in state.concept_profiles
    ]


def _payload(state: AgentState, claims) -> dict[str, Any]:
    schema = {
        "classification_dimensions": [
            {
                "name": "分类维度名；必须由 claim 支持",
                "description": "证据内说明",
                "reason": "为什么认为这是分类维度",
                "evidence_claim_ids": ["claim_xxx"],
                "confidence": 0.0,
                "needs_review": False,
            }
        ],
        "selected_dimension_name": "可靠主维度名称；无法确定则为 null",
    }
    return {
        "task": "发现文档支持的分类维度。只处理分类维度，不建树。",
        "output_schema": schema,
        "evidence_claims": claim_payload(claims),
        "concept_profiles": _concept_payload(state),
    }


def discover_dimensions_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    stage_claims = filter_claims_for_stage(
        state.evidence_claims,
        "dimension",
        env_int("LLM_DIMENSION_MAX_CLAIMS", 150),
    )
    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="发现分类维度",
        prompt_file="discover_dimensions_prompt.md",
        payload=_payload(state, stage_claims),
        required_keys={"classification_dimensions": list, "selected_dimension_name": (str, type(None))},
        max_tokens=env_int("LLM_DIMENSION_MAX_TOKENS", 1600),
        temperature=0.0,
    )
    claim_by_id = {claim.claim_id: claim for claim in state.evidence_claims}
    dimensions: list[ClassificationDimension] = []

    for item in data.get("classification_dimensions") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        claim_ids = valid_claim_ids(claim_by_id, item.get("evidence_claim_ids") or [])
        refs = refs_from_claim_ids(claim_by_id, claim_ids)
        needs_review = parse_bool(item.get("needs_review"), not bool(refs))
        dimensions.append(
            ClassificationDimension(
                dimension_id=stable_id("dim", name),
                name=name,
                description=str(item.get("description") or ""),
                evidence_refs=refs,
                evidence_claim_ids=claim_ids,
                reason=str(item.get("reason") or ""),
                confidence=clamp_confidence(item.get("confidence"), 0.65),
                needs_review=needs_review,
            )
        )

    selected_name = data.get("selected_dimension_name")
    selected = None
    if selected_name:
        selected = next((dim for dim in dimensions if dim.name == selected_name), None)
    if selected is not None and selected.needs_review:
        selected = None

    state.classification_dimensions = dimensions
    state.selected_dimension = selected
    append_step_trace(
        state.step_traces,
        step_name="discover_dimensions_with_llm",
        status="success",
        input_summary={
            "claims": len(state.evidence_claims),
            "stage_claims": len(stage_claims),
            "stage_claim_types": count_claim_types(stage_claims),
            "concept_profiles": len(state.concept_profiles),
        },
        output_summary={"dimensions": len(dimensions), "selected": selected.name if selected else None},
        raw_response=raw_response,
    )
    return state
