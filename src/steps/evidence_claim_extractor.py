from __future__ import annotations

import os
from typing import Any

from ..core.agent_state import AgentState, EvidenceClaim
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    chunk_payload,
    clamp_confidence,
    env_int,
    parse_bool,
    refs_from_chunk_ids,
    stable_id,
)


ALLOWED_CLAIM_TYPES = {
    "definition",
    "inclusion",
    "exclusion",
    "hierarchy",
    "classification_principle",
    "grade_definition",
    "grade_mapping",
    "rule_phrase",
    "insufficient_evidence",
}

ALLOWED_SUPPORT_LEVELS = {"explicit", "structural", "inferred", "weak", "ocr"}


def _payload(chunks: list) -> dict[str, Any]:
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
                "evidence_quote": "支持该 claim 的短原文片段；必须来自 chunk 原文",
                "support_level": "explicit | structural | inferred | weak | ocr",
                "evidence_chunk_ids": ["doc_1_chunk_1"],
                "confidence": 0.0,
                "needs_review": False,
                "review_reason": "需要人工复核的原因；无需复核则为空字符串",
                "status": "evidence_supported | proposed | insufficient_evidence",
            }
        ]
    }
    return {
        "task": "从文档 chunk 中抽取证据事实 claim。不要生成规则树。",
        "allowed_claim_types": sorted(ALLOWED_CLAIM_TYPES),
        "allowed_support_levels": sorted(ALLOWED_SUPPORT_LEVELS),
        "output_schema": schema,
        "document_chunks": chunk_payload(chunks),
    }


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
        evidence_quote = str(item.get("evidence_quote") or "").strip()
        support_level = str(item.get("support_level") or "").strip().lower()
        if not claim_type or not subject:
            continue
        if claim_type not in ALLOWED_CLAIM_TYPES:
            continue
        if support_level not in ALLOWED_SUPPORT_LEVELS:
            support_level = "weak"
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
        has_ocr_evidence = any(ref.source_method == "ocr" for ref in refs)
        if has_ocr_evidence:
            support_level = "ocr"
        review_reason = str(item.get("review_reason") or "").strip()
        if has_ocr_evidence and not review_reason:
            review_reason = "证据来自 OCR，需要人工核验识别结果。"
        needs_review = (
            has_ocr_evidence
            or support_level in {"inferred", "weak", "ocr"}
            or parse_bool(item.get("needs_review"), not bool(refs))
        )
        claim_id = str(item.get("claim_id") or "").strip() or stable_id("claim", fingerprint)
        claims.append(
            EvidenceClaim(
                claim_id=claim_id,
                claim_type=claim_type,
                subject=subject,
                predicate=predicate,
                object=obj,
                value=value,
                evidence_quote=evidence_quote,
                support_level=support_level,
                evidence_refs=refs,
                confidence=clamp_confidence(item.get("confidence"), 0.55 if has_ocr_evidence else 0.65),
                needs_review=needs_review,
                review_reason=review_reason,
                status=str(item.get("status") or ("proposed" if needs_review else "evidence_supported")),
            )
        )

    return claims


def extract_evidence_claims_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    batch_size = max(1, int(os.getenv("CLAIM_BATCH_SIZE", "8")))
    all_claims: list[EvidenceClaim] = []
    raw_batches: list[str] = []
    seen: set[str] = set()

    if not state.chunks:
        state.evidence_claims = []
        append_step_trace(
            state.step_traces,
            step_name="extract_evidence_claims_with_llm",
            status="success",
            message="No chunks available for claim extraction.",
            input_summary={"chunks": 0, "batch_size": batch_size, "batches": 0},
            output_summary={"claims": 0},
        )
        return state

    batches = [
        state.chunks[index : index + batch_size]
        for index in range(0, len(state.chunks), batch_size)
    ]
    for batch_index, batch_chunks in enumerate(batches, start=1):
        data, raw_response = call_llm_json(
            llm_client=llm_client,
            task_name=f"抽取 evidence claims batch {batch_index}/{len(batches)}",
            prompt_file="extract_evidence_claims_prompt.md",
            payload={
                **_payload(batch_chunks),
                "batch_index": batch_index,
                "batch_count": len(batches),
            },
            required_keys={"claims": list},
            max_tokens=env_int("LLM_CLAIM_MAX_TOKENS", 2000),
            temperature=0.0,
            disable_thinking=True,
        )
        raw_batches.append(f"batch={batch_index}/{len(batches)}\n{raw_response}")
        for claim in _to_claims(data, state):
            fingerprint = "|".join(
                [claim.claim_type, claim.subject, claim.predicate, claim.object, claim.value]
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            all_claims.append(claim)

    state.evidence_claims = all_claims
    append_step_trace(
        state.step_traces,
        step_name="extract_evidence_claims_with_llm",
        status="success",
        input_summary={"chunks": len(state.chunks), "batch_size": batch_size, "batches": len(batches)},
        output_summary={"claims": len(state.evidence_claims)},
        raw_response="\n\n===\n\n".join(raw_batches),
    )
    return state
