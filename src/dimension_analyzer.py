from __future__ import annotations

import json
from typing import Any

from .agent_state import AgentState, ClassificationDimension
from .llm_task_utils import (
    append_step_trace,
    claim_payload,
    clamp_confidence,
    common_system_prompt,
    extract_json_object,
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


def _user_prompt(state: AgentState) -> str:
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
    return json.dumps(
        {
            "task": "发现文档支持的分类维度。只处理分类维度，不建树。",
            "output_schema": schema,
            "evidence_claims": claim_payload(state.evidence_claims),
            "concept_profiles": _concept_payload(state),
        },
        ensure_ascii=False,
        indent=2,
    )


def discover_dimensions_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    messages = [
        {"role": "system", "content": common_system_prompt("发现分类维度")},
        {"role": "user", "content": _user_prompt(state)},
    ]
    response = llm_client.chat(messages)
    data = extract_json_object(response.content)
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
    if selected is None and dimensions:
        selected = sorted(dimensions, key=lambda item: (item.needs_review, -item.confidence))[0]

    state.classification_dimensions = dimensions
    state.selected_dimension = selected
    append_step_trace(
        state.step_traces,
        step_name="discover_dimensions_with_llm",
        status="success",
        input_summary={"claims": len(state.evidence_claims), "concept_profiles": len(state.concept_profiles)},
        output_summary={"dimensions": len(dimensions), "selected": selected.name if selected else None},
        raw_response=response.content,
    )
    return state
