from __future__ import annotations

import json
from typing import Any

from .agent_state import AgentState, EvidenceClaim
from .llm_task_utils import (
    append_step_trace,
    chunk_payload,
    clamp_confidence,
    common_system_prompt,
    extract_json_object,
    parse_bool,
    refs_from_chunk_ids,
    stable_id,
)


def _user_prompt(state: AgentState) -> str:
    schema = {
        "claims": [
            {
                "claim_type": (
                    "definition | inclusion | exclusion | hierarchy | "
                    "classification_principle | grade_definition | "
                    "grade_mapping | rule_phrase | insufficient_evidence"
                ),
                "subject": "必须来自文档原文的主体词",
                "predicate": "定义/包括/不包括/属于/分为/映射为等关系",
                "object": "关系对象，没有则为空字符串",
                "value": "原文支持的补充信息，没有则为空字符串",
                "evidence_chunk_ids": ["doc_1_chunk_1"],
                "confidence": 0.0,
                "needs_review": False,
                "status": "evidence_supported | proposed | insufficient_evidence",
            }
        ]
    }
    return json.dumps(
        {
            "task": "从文档 chunk 中抽取证据事实 claim。不要生成规则树。",
            "allowed_claim_types": [
                "definition",
                "inclusion",
                "exclusion",
                "hierarchy",
                "classification_principle",
                "grade_definition",
                "grade_mapping",
                "rule_phrase",
                "insufficient_evidence",
            ],
            "output_schema": schema,
            "document_chunks": chunk_payload(state.chunks),
        },
        ensure_ascii=False,
        indent=2,
    )


def _to_claims(data: dict[str, Any], state: AgentState) -> list[EvidenceClaim]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in state.chunks}
    claims: list[EvidenceClaim] = []
    seen: set[str] = set()

    for item in data.get("claims") or []:
        if not isinstance(item, dict):
            continue
        claim_type = str(item.get("claim_type") or "").strip()
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        value = str(item.get("value") or "").strip()
        if not claim_type or not subject:
            continue
        fingerprint = "|".join([claim_type, subject, predicate, obj, value])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        refs = refs_from_chunk_ids(
            chunk_by_id,
            item.get("evidence_chunk_ids") or [],
            f"claim:{claim_type}:{subject}",
            0.88,
        )
        needs_review = parse_bool(item.get("needs_review"), not bool(refs))
        claim_id = str(item.get("claim_id") or "").strip() or stable_id("claim", fingerprint)
        claims.append(
            EvidenceClaim(
                claim_id=claim_id,
                claim_type=claim_type,
                subject=subject,
                predicate=predicate,
                object=obj,
                value=value,
                evidence_refs=refs,
                confidence=clamp_confidence(item.get("confidence"), 0.65),
                needs_review=needs_review,
                status=str(item.get("status") or ("proposed" if needs_review else "evidence_supported")),
            )
        )

    return claims


def extract_evidence_claims_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    messages = [
        {"role": "system", "content": common_system_prompt("抽取 evidence claims")},
        {"role": "user", "content": _user_prompt(state)},
    ]
    response = llm_client.chat(messages)
    data = extract_json_object(response.content)
    state.evidence_claims = _to_claims(data, state)
    append_step_trace(
        state.step_traces,
        step_name="extract_evidence_claims_with_llm",
        status="success",
        input_summary={"chunks": len(state.chunks)},
        output_summary={"claims": len(state.evidence_claims)},
        raw_response=response.content,
    )
    return state
