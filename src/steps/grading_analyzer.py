from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState, GradeDefinition
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    claim_payload,
    clamp_confidence,
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
            "evidence_claim_ids": node.evidence_claim_ids,
        }
        for node in state.nodes
    ]


def _payload(state: AgentState) -> dict[str, Any]:
    schema = {
        "grade_scheme": [
            {
                "grade_name": "等级名称，必须来自证据",
                "definition": "等级定义，必须来自证据",
                "criteria": ["证据支持的判定条件"],
                "evidence_claim_ids": ["claim_xxx"],
                "confidence": 0.0,
                "needs_review": False,
                "status": "evidence_supported | proposed | insufficient_evidence",
            }
        ],
        "node_grade_assignments": [
            {
                "path": "必须等于输入节点 path",
                "grade": "等级名称；无法确定则为 null",
                "grade_reason": "证据内理由；无法确定则说明证据不足",
                "grade_evidence_claim_ids": ["claim_xxx"],
                "confidence": 0.0,
                "needs_review": False,
            }
        ],
    }
    return {
        "task": "只处理分级方案和节点分级。不得创造默认等级，不得基于常识分级。",
        "output_schema": schema,
        "nodes": _node_payload(state),
        "evidence_claims": claim_payload(state.evidence_claims),
    }


def analyze_grading_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="抽取分级方案并分配节点等级",
        prompt_file="analyze_grading_prompt.md",
        payload=_payload(state),
        required_keys={"grade_scheme": list, "node_grade_assignments": list},
    )
    claim_by_id = {claim.claim_id: claim for claim in state.evidence_claims}
    grades: list[GradeDefinition] = []
    seen_grades: set[str] = set()

    for item in data.get("grade_scheme") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("grade_name") or "").strip()
        if not name or name in seen_grades:
            continue
        seen_grades.add(name)
        claim_ids = valid_claim_ids(claim_by_id, item.get("evidence_claim_ids") or [])
        refs = refs_from_claim_ids(claim_by_id, claim_ids)
        needs_review = parse_bool(item.get("needs_review"), not bool(refs))
        grades.append(
            GradeDefinition(
                grade_id=stable_id("grade", name),
                grade_name=name,
                definition=str(item.get("definition") or ""),
                criteria=string_list(item.get("criteria")),
                evidence_refs=refs,
                evidence_claim_ids=claim_ids,
                confidence=clamp_confidence(item.get("confidence"), 0.65),
                needs_review=needs_review,
                status=str(item.get("status") or ("proposed" if needs_review else "evidence_supported")),
            )
        )

    nodes_by_path = {node.path: node for node in state.nodes}
    for item in data.get("node_grade_assignments") or []:
        if not isinstance(item, dict):
            continue
        node = nodes_by_path.get(str(item.get("path") or "").strip())
        if node is None:
            continue
        grade = item.get("grade")
        claim_ids = valid_claim_ids(claim_by_id, item.get("grade_evidence_claim_ids") or [])
        refs = refs_from_claim_ids(claim_by_id, claim_ids)
        node.grade = str(grade).strip() if grade else None
        node.grade_reason = str(item.get("grade_reason") or "")
        node.grade_evidence_refs = refs
        if claim_ids:
            node.evidence_claim_ids = list(dict.fromkeys([*node.evidence_claim_ids, *claim_ids]))
        if not node.grade or not refs or parse_bool(item.get("needs_review"), False):
            node.needs_review = True

    state.grade_scheme = grades
    append_step_trace(
        state.step_traces,
        step_name="analyze_grading_with_llm",
        status="success",
        input_summary={"nodes": len(state.nodes), "claims": len(state.evidence_claims)},
        output_summary={"grades": len(state.grade_scheme), "graded_nodes": sum(1 for node in state.nodes if node.grade)},
        raw_response=raw_response,
    )
    return state
