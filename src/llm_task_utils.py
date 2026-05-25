from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .agent_state import DocumentChunk, EvidenceClaim, EvidenceRef, StepTrace
from .evidence_store import create_evidence_ref, dedupe_evidence_refs


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object.")
    return data


def clamp_confidence(value: Any, default: float) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return default


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def chunk_payload(chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_name": chunk.doc_name,
            "section_title": chunk.section_title,
            "position": chunk.position,
            "text": chunk.text,
        }
        for chunk in chunks
    ]


def claim_payload(claims: list[EvidenceClaim]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "object": claim.object,
            "value": claim.value,
            "confidence": claim.confidence,
            "needs_review": claim.needs_review,
            "evidence_ids": [ref.evidence_id for ref in claim.evidence_refs],
            "evidence_texts": [ref.text for ref in claim.evidence_refs[:2]],
        }
        for claim in claims
    ]


def refs_from_chunk_ids(
    chunk_by_id: dict[str, DocumentChunk],
    chunk_ids: list[Any],
    used_for: str,
    score: float,
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for raw_id in chunk_ids or []:
        chunk = chunk_by_id.get(str(raw_id))
        if chunk is None:
            continue
        refs.append(create_evidence_ref(chunk, used_for, score))
    return dedupe_evidence_refs(refs)


def refs_from_claim_ids(
    claim_by_id: dict[str, EvidenceClaim],
    claim_ids: list[Any],
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for raw_id in claim_ids or []:
        claim = claim_by_id.get(str(raw_id))
        if claim is None:
            continue
        refs.extend(claim.evidence_refs)
    return dedupe_evidence_refs(refs)


def valid_claim_ids(claim_by_id: dict[str, EvidenceClaim], claim_ids: list[Any]) -> list[str]:
    return [str(claim_id) for claim_id in claim_ids or [] if str(claim_id) in claim_by_id]


def common_system_prompt(task_name: str) -> str:
    return f"""你是企业文档证据驱动规则树生成 Agent 的一个受控步骤：{task_name}。

硬性规则：
- 只能使用本次输入里的 document_chunks、evidence_claims、concept_profiles 或已有候选结果。
- 不得使用行业常识、默认分类、默认等级、默认风险规则或文档外示例。
- 所有业务名称、等级名称、描述、规则词、层级关系都必须能追溯到输入证据。
- 如果证据不足，必须输出 needs_review=true 或 insufficient_evidence，不得补全。
- 输出必须是 JSON object，不要 Markdown，不要解释文字。
"""


def append_step_trace(
    traces: list[StepTrace],
    step_name: str,
    status: str,
    message: str = "",
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    raw_response: str = "",
) -> None:
    traces.append(
        StepTrace(
            step_name=step_name,
            status=status,
            message=message,
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            raw_response=raw_response,
        )
    )
