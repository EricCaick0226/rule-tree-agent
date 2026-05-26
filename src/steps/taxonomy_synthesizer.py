from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState, TreeNode
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
            "definitions": concept.definitions,
            "included_items": concept.included_items,
            "excluded_items": concept.excluded_items,
            "related_claim_ids": concept.related_claim_ids,
        }
        for concept in state.concept_profiles
    ]


def _dimension_payload(state: AgentState) -> dict[str, Any] | None:
    if not state.selected_dimension:
        return None
    dimension = state.selected_dimension
    return {
        "dimension_id": dimension.dimension_id,
        "name": dimension.name,
        "description": dimension.description,
        "reason": dimension.reason,
        "evidence_claim_ids": dimension.evidence_claim_ids,
        "confidence": dimension.confidence,
        "needs_review": dimension.needs_review,
    }


def _payload(state: AgentState, claims) -> dict[str, Any]:
    schema = {
        "nodes": [
            {
                "name": "节点名称，必须来自证据",
                "path": "父节点 / 子节点；根节点就是节点名称",
                "parent_path": "父路径；根节点为 null",
                "level": 1,
                "evidence_claim_ids": ["claim_xxx"],
                "confidence": 0.0,
                "needs_review": False,
                "status": "evidence_supported | proposed | insufficient_evidence",
            }
        ]
    }
    return {
        "task": (
            "基于 evidence_claims 和 concept_profiles 合成候选分类树。"
            "不要生成描述、分级或规则。"
        ),
        "selected_dimension": _dimension_payload(state),
        "output_schema": schema,
        "evidence_claims": claim_payload(claims),
        "concept_profiles": _concept_payload(state),
    }


def synthesize_taxonomy_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    stage_claims = filter_claims_for_stage(
        state.evidence_claims,
        "taxonomy",
        env_int("LLM_TAXONOMY_MAX_CLAIMS", 350),
    )
    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="合成候选分类树",
        prompt_file="synthesize_taxonomy_prompt.md",
        payload=_payload(state, stage_claims),
        required_keys={"nodes": list},
        max_tokens=env_int("LLM_TAXONOMY_MAX_TOKENS", 3000),
        temperature=0.0,
    )
    claim_by_id = {claim.claim_id: claim for claim in state.evidence_claims}
    nodes_by_path: dict[str, TreeNode] = {}
    parent_paths: dict[str, str | None] = {}

    for item in data.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        path = str(item.get("path") or name).strip()
        if not name or not path:
            continue
        claim_ids = valid_claim_ids(claim_by_id, item.get("evidence_claim_ids") or [])
        refs = refs_from_claim_ids(claim_by_id, claim_ids)
        needs_review = parse_bool(item.get("needs_review"), not bool(refs))
        level = int(item.get("level") or max(1, path.count(" / ") + 1))
        node = TreeNode(
            node_id=stable_id("node", path),
            name=name,
            path=path,
            level=level,
            parent_id=None,
            evidence_refs=refs,
            evidence_claim_ids=claim_ids,
            confidence=clamp_confidence(item.get("confidence"), 0.65),
            needs_review=needs_review,
            status=str(item.get("status") or ("proposed" if needs_review else "evidence_supported")),
        )
        nodes_by_path[path] = node
        raw_parent = item.get("parent_path")
        parent_paths[path] = str(raw_parent).strip() if raw_parent else None

    for path, node in nodes_by_path.items():
        parent_path = parent_paths.get(path)
        if parent_path and parent_path in nodes_by_path:
            node.parent_id = nodes_by_path[parent_path].node_id
        elif " / " in path:
            inferred = path.rsplit(" / ", 1)[0]
            if inferred in nodes_by_path:
                node.parent_id = nodes_by_path[inferred].node_id

    state.nodes = list(nodes_by_path.values())
    append_step_trace(
        state.step_traces,
        step_name="synthesize_taxonomy_with_llm",
        status="success",
        input_summary={
            "claims": len(state.evidence_claims),
            "stage_claims": len(stage_claims),
            "stage_claim_types": count_claim_types(stage_claims),
            "concept_profiles": len(state.concept_profiles),
        },
        output_summary={"nodes": len(state.nodes)},
        raw_response=raw_response,
    )
    return state
