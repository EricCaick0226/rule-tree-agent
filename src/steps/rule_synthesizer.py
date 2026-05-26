from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState, MatchingRule
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    claim_payload,
    clamp_confidence,
    merge_unique,
    parse_bool,
    refs_from_claim_ids,
    stable_id,
    string_list,
    valid_claim_ids,
)


def _node_payload(state: AgentState) -> list[dict[str, Any]]:
    return [
        {
            "node_id": node.node_id,
            "name": node.name,
            "path": node.path,
            "description": node.description,
            "grade": node.grade,
            "evidence_claim_ids": node.evidence_claim_ids,
        }
        for node in state.nodes
    ]


def _payload(state: AgentState) -> dict[str, Any]:
    schema = {
        "node_rules": [
            {
                "path": "必须等于输入节点 path",
                "rules": [
                    {
                        "rule_type": "keyword_rule | phrase_rule | context_rule | negative_rule",
                        "conditions": ["必须来自证据的关键词或短语"],
                        "negative_conditions": ["只有明确排除证据时填写"],
                        "evidence_claim_ids": ["claim_xxx"],
                        "confidence": 0.0,
                        "needs_review": False,
                        "status": "evidence_supported | proposed | insufficient_evidence",
                    }
                ],
            }
        ]
    }
    return {
        "task": (
            "只为已有节点生成匹配规则。"
            "规则词、短语、排除条件必须来自 evidence_claims。"
        ),
        "output_schema": schema,
        "nodes": _node_payload(state),
        "evidence_claims": claim_payload(state.evidence_claims),
    }


def synthesize_rules_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="生成证据内匹配规则",
        prompt_file="synthesize_rules_prompt.md",
        payload=_payload(state),
        required_keys={"node_rules": list},
    )
    claim_by_id = {claim.claim_id: claim for claim in state.evidence_claims}
    nodes_by_path = {node.path: node for node in state.nodes}

    for item in data.get("node_rules") or []:
        if not isinstance(item, dict):
            continue
        node = nodes_by_path.get(str(item.get("path") or "").strip())
        if node is None:
            continue
        rules: list[MatchingRule] = []
        for rule_item in item.get("rules") or []:
            if not isinstance(rule_item, dict):
                continue
            conditions = string_list(rule_item.get("conditions"))
            negatives = string_list(rule_item.get("negative_conditions"))
            claim_ids = valid_claim_ids(claim_by_id, rule_item.get("evidence_claim_ids") or [])
            refs = refs_from_claim_ids(claim_by_id, claim_ids)
            needs_review = parse_bool(rule_item.get("needs_review"), not bool(refs) or not conditions)
            rules.append(
                MatchingRule(
                    rule_id=stable_id("rule", f"{node.path}:{'|'.join(conditions)}:{'|'.join(negatives)}"),
                    target_node_id=node.node_id,
                    rule_type=str(rule_item.get("rule_type") or "keyword_rule"),
                    conditions=conditions,
                    negative_conditions=negatives,
                    evidence_refs=refs,
                    evidence_claim_ids=claim_ids,
                    confidence=clamp_confidence(rule_item.get("confidence"), 0.6),
                    needs_review=needs_review,
                    status=str(
                        rule_item.get("status")
                        or ("proposed" if needs_review else "evidence_supported")
                    ),
                )
            )
            if claim_ids:
                node.evidence_claim_ids = merge_unique(node.evidence_claim_ids, claim_ids)
            if needs_review:
                node.needs_review = True
        node.rules = rules

    for node in state.nodes:
        if not node.rules:
            node.needs_review = True

    append_step_trace(
        state.step_traces,
        step_name="synthesize_rules_with_llm",
        status="success",
        input_summary={"nodes": len(state.nodes), "claims": len(state.evidence_claims)},
        output_summary={"rules": sum(len(node.rules) for node in state.nodes)},
        raw_response=raw_response,
    )
    return state
