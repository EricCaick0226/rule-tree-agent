from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState, ConceptProfile
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    claim_payload,
    clamp_confidence,
    count_claim_types,
    env_int,
    filter_claims_for_stage,
    normalize_text,
    parse_bool,
    refs_from_claim_ids,
    stable_id,
    string_list,
    valid_claim_ids,
)


def _payload(state: AgentState, claims) -> dict[str, Any]:
    schema = {
        "concept_profiles": [
            {
                "name": "规范化概念名，必须来自证据",
                "aliases": ["同义或近义称呼，必须来自证据"],
                "definitions": ["定义文本，必须来自证据"],
                "included_items": ["包括项，必须来自证据"],
                "excluded_items": ["排除项，必须来自证据"],
                "related_claim_ids": ["claim_xxx"],
                "confidence": 0.0,
                "needs_review": False,
                "status": "evidence_supported | proposed | insufficient_evidence",
            }
        ]
    }
    return {
        "task": "基于 evidence_claims 形成概念画像。不要建树，不要分级。",
        "output_schema": schema,
        "evidence_claims": claim_payload(claims),
    }


def normalize_concepts_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    stage_claims = filter_claims_for_stage(
        state.evidence_claims,
        "concept",
        env_int("LLM_CONCEPT_MAX_CLAIMS", 300),
    )
    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="归一化概念并形成概念画像",
        prompt_file="normalize_concepts_prompt.md",
        payload=_payload(state, stage_claims),
        required_keys={"concept_profiles": list},
        max_tokens=env_int("LLM_CONCEPT_MAX_TOKENS", 1600),
        temperature=0.0,
        disable_thinking=True,
    )
    claim_by_id = {claim.claim_id: claim for claim in state.evidence_claims}
    profiles: list[ConceptProfile] = []
    seen: set[str] = set()

    for item in data.get("concept_profiles") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        norm = normalize_text(name)
        if norm in seen:
            continue
        seen.add(norm)
        claim_ids = valid_claim_ids(claim_by_id, item.get("related_claim_ids") or [])
        refs = refs_from_claim_ids(claim_by_id, claim_ids)
        needs_review = parse_bool(item.get("needs_review"), not bool(refs))
        confidence = clamp_confidence(item.get("confidence"), 0.65)
        profile = ConceptProfile(
            concept_id=stable_id("concept_profile", norm),
            name=name,
            aliases=string_list(item.get("aliases")),
            definitions=string_list(item.get("definitions")),
            included_items=string_list(item.get("included_items")),
            excluded_items=string_list(item.get("excluded_items")),
            related_claim_ids=claim_ids,
            evidence_refs=refs,
            confidence=confidence,
            needs_review=needs_review,
            status=str(item.get("status") or ("proposed" if needs_review else "evidence_supported")),
        )
        profiles.append(profile)

    state.concept_profiles = profiles
    append_step_trace(
        state.step_traces,
        step_name="normalize_concepts_with_llm",
        status="success",
        input_summary={
            "claims": len(state.evidence_claims),
            "stage_claims": len(stage_claims),
            "stage_claim_types": count_claim_types(stage_claims),
        },
        output_summary={"concept_profiles": len(state.concept_profiles)},
        raw_response=raw_response,
    )
    return state
