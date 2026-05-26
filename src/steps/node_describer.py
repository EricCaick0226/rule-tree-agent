from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    claim_payload,
    merge_unique,
    parse_bool,
    refs_from_claim_ids,
    valid_claim_ids,
)


INSUFFICIENT_DESCRIPTION = (
    "当前文档未提供该分类的明确说明，建议人工确认其定义、范围和适用边界。"
)


def _node_payload(state: AgentState) -> list[dict[str, Any]]:
    return [
        {
            "node_id": node.node_id,
            "name": node.name,
            "path": node.path,
            "level": node.level,
            "evidence_claim_ids": node.evidence_claim_ids,
        }
        for node in state.nodes
    ]


def _payload(state: AgentState) -> dict[str, Any]:
    schema = {
        "node_descriptions": [
            {
                "path": "必须等于输入节点 path",
                "description": "仅基于证据的节点说明；证据不足则使用谨慎说明",
                "description_evidence_claim_ids": ["claim_xxx"],
                "description_evidence_level": "A | B | C | D",
                "needs_review": False,
            }
        ]
    }
    return {
        "task": (
            "为已有候选节点生成证据内描述。"
            "不要新增节点，不要分级，不要生成规则。"
        ),
        "output_schema": schema,
        "nodes": _node_payload(state),
        "evidence_claims": claim_payload(state.evidence_claims),
    }


def describe_nodes_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="生成节点证据内描述",
        prompt_file="describe_nodes_prompt.md",
        payload=_payload(state),
        required_keys={"node_descriptions": list},
    )
    claim_by_id = {claim.claim_id: claim for claim in state.evidence_claims}
    nodes_by_path = {node.path: node for node in state.nodes}

    for item in data.get("node_descriptions") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        node = nodes_by_path.get(path)
        if node is None:
            continue
        claim_ids = valid_claim_ids(claim_by_id, item.get("description_evidence_claim_ids") or [])
        refs = refs_from_claim_ids(claim_by_id, claim_ids)
        description = str(item.get("description") or "").strip() or INSUFFICIENT_DESCRIPTION
        level = str(item.get("description_evidence_level") or ("B" if refs else "D")).strip().upper()
        if level not in {"A", "B", "C", "D"}:
            level = "B" if refs else "D"
        node.description = description
        node.description_evidence_refs = refs
        node.description_evidence_level = level
        if claim_ids:
            node.evidence_claim_ids = merge_unique(node.evidence_claim_ids, claim_ids)
        if level in {"C", "D"} or not refs or parse_bool(item.get("needs_review"), False):
            node.needs_review = True

    for node in state.nodes:
        if not node.description:
            node.description = INSUFFICIENT_DESCRIPTION
            node.description_evidence_level = "D"
            node.needs_review = True

    append_step_trace(
        state.step_traces,
        step_name="describe_nodes_with_llm",
        status="success",
        input_summary={"nodes": len(state.nodes), "claims": len(state.evidence_claims)},
        output_summary={
            "described_nodes": sum(1 for node in state.nodes if node.description),
        },
        raw_response=raw_response,
    )
    return state
