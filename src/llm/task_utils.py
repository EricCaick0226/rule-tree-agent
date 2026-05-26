from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..core.agent_state import DocumentChunk, EvidenceClaim, EvidenceRef, StepTrace
from ..io.evidence_store import create_evidence_ref, dedupe_evidence_refs


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


def load_prompt(prompt_file: str) -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / prompt_file
    return prompt_path.read_text(encoding="utf-8").strip()


def _format_required_keys(required_keys: dict[str, Any] | list[str] | tuple[str, ...]) -> dict[str, Any]:
    if isinstance(required_keys, dict):
        return required_keys
    return {key: None for key in required_keys}


def _type_name(expected_type: Any) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(_type_name(item) for item in expected_type)
    if hasattr(expected_type, "__name__"):
        return expected_type.__name__
    return str(expected_type)


def env_int(name: str, default: int) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def validate_json_shape(
    data: dict[str, Any],
    required_keys: dict[str, Any] | list[str] | tuple[str, ...],
) -> None:
    expected = _format_required_keys(required_keys)
    missing = [key for key in expected if key not in data]
    if missing:
        raise ValueError(f"Missing required top-level keys: {', '.join(missing)}")
    type_errors: list[str] = []
    for key, expected_type in expected.items():
        if expected_type is None:
            continue
        value = data.get(key)
        if not isinstance(value, expected_type):
            type_errors.append(f"{key} must be {_type_name(expected_type)}")
    if type_errors:
        raise ValueError("; ".join(type_errors))


def _llm_user_message(
    task_prompt: str,
    payload: dict[str, Any],
    retry_error: str = "",
    previous_response: str = "",
) -> str:
    body: dict[str, Any] = {
        "task_prompt": task_prompt,
        "input_payload": payload,
    }
    if retry_error:
        body["retry_instruction"] = (
            "Your previous answer could not be parsed or failed schema validation. "
            "Return one complete valid JSON object only, using the required schema and grounded evidence IDs. "
            "Do not truncate strings, do not omit closing brackets, and keep text fields concise."
        )
        body["previous_error"] = retry_error
        body["previous_response_excerpt"] = previous_response[:4000]
    return json.dumps(body, ensure_ascii=False, indent=2)


def call_llm_json(
    llm_client: Any,
    task_name: str,
    prompt_file: str,
    payload: dict[str, Any],
    required_keys: dict[str, Any] | list[str] | tuple[str, ...],
    max_attempts: int = 2,
    max_tokens: int | None = None,
    temperature: float | None = None,
    disable_thinking: bool = False,
) -> tuple[dict[str, Any], str]:
    task_prompt = load_prompt(prompt_file)
    raw_responses: list[str] = []
    last_error = ""
    previous_response = ""

    for attempt in range(max(1, max_attempts)):
        attempt_max_tokens = max_tokens
        if max_tokens is not None and attempt > 0:
            attempt_max_tokens = int(max_tokens * 2)
        messages = [
            {"role": "system", "content": common_system_prompt(task_name)},
            {
                "role": "user",
                "content": _llm_user_message(
                    task_prompt=task_prompt,
                    payload=payload,
                    retry_error=last_error if attempt > 0 else "",
                    previous_response=previous_response if attempt > 0 else "",
                ),
            },
        ]
        response = llm_client.chat(
            messages,
            max_tokens=attempt_max_tokens,
            temperature=temperature,
            disable_thinking=disable_thinking,
        )
        previous_response = response.content
        raw_responses.append(f"attempt={attempt + 1}\n{response.content}")
        try:
            data = extract_json_object(response.content)
            validate_json_shape(data, required_keys)
            return data, "\n\n---\n\n".join(raw_responses)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)

    raise ValueError(f"LLM JSON output failed validation after {max_attempts} attempt(s): {last_error}")


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
            "page_number": chunk.page_number,
            "source_method": chunk.source_method,
            "source_warning": chunk.source_warning,
            "text": chunk.text,
        }
        for chunk in chunks
    ]


STAGE_CLAIM_TYPES = {
    "concept": {
        "definition",
        "inclusion",
        "exclusion",
        "hierarchy",
        "classification_principle",
        "grade_definition",
        "grade_mapping",
        "rule_phrase",
    },
    "dimension": {"classification_principle", "definition", "hierarchy", "inclusion"},
    "taxonomy": {"classification_principle", "definition", "hierarchy", "inclusion"},
    "description": {"classification_principle", "definition", "hierarchy", "inclusion", "exclusion"},
    "grading": {"grade_definition", "grade_mapping", "definition", "hierarchy"},
    "rules": {"rule_phrase", "definition", "inclusion", "exclusion", "hierarchy"},
}

SUPPORT_PRIORITY = {
    "explicit": 0,
    "structural": 1,
    "inferred": 2,
    "weak": 3,
    "ocr": 4,
}


def filter_claims_for_stage(
    claims: list[EvidenceClaim],
    stage: str,
    max_claims: int | None = None,
) -> list[EvidenceClaim]:
    allowed_types = STAGE_CLAIM_TYPES.get(stage)
    candidates = [
        claim
        for claim in claims
        if claim.claim_type != "insufficient_evidence"
        and (allowed_types is None or claim.claim_type in allowed_types)
    ]
    if not candidates and allowed_types is not None:
        candidates = [claim for claim in claims if claim.claim_type != "insufficient_evidence"]

    candidates.sort(
        key=lambda claim: (
            SUPPORT_PRIORITY.get(claim.support_level, 9),
            claim.needs_review,
            -claim.confidence,
            claim.claim_type,
            claim.subject,
            claim.object,
        )
    )
    if max_claims is not None and max_claims > 0:
        return candidates[:max_claims]
    return candidates


def count_claim_types(claims: list[EvidenceClaim]) -> dict[str, int]:
    return dict(Counter(claim.claim_type for claim in claims))


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def claim_payload(
    claims: list[EvidenceClaim],
    include_evidence_texts: bool = False,
    max_quote_chars: int = 500,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for claim in claims:
        item = {
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "object": claim.object,
            "value": claim.value,
            "evidence_quote": _truncate_text(claim.evidence_quote, max_quote_chars),
            "support_level": claim.support_level,
            "confidence": claim.confidence,
            "needs_review": claim.needs_review,
            "review_reason": claim.review_reason,
            "evidence_ids": [ref.evidence_id for ref in claim.evidence_refs],
            "evidence_page_numbers": [ref.page_number for ref in claim.evidence_refs],
            "evidence_source_methods": [ref.source_method for ref in claim.evidence_refs],
        }
        if include_evidence_texts:
            item["evidence_texts"] = [
                _truncate_text(ref.text, max_quote_chars)
                for ref in claim.evidence_refs[:2]
            ]
        payload.append(item)
    return payload


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
        ref_score = min(score, 0.68) if chunk.source_method == "ocr" else score
        refs.append(create_evidence_ref(chunk, used_for, ref_score))
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


def merge_unique(existing: list[str], additions: list[str]) -> list[str]:
    return list(dict.fromkeys([*existing, *additions]))


def common_system_prompt(task_name: str) -> str:
    return f"""你是企业文档证据驱动规则树生成 Agent 的一个受控步骤：{task_name}。

硬性规则：
- 只能使用本次输入里的 document_chunks、evidence_claims、concept_profiles 或已有候选结果。
- 不得使用行业常识、默认分类、默认等级、默认风险规则或文档外示例。
- 所有业务名称、等级名称、描述、规则词、层级关系都必须能追溯到输入证据。
- 如果证据不足，必须输出 needs_review=true 或 insufficient_evidence，不得补全。
- 如果证据来自 OCR，必须保持谨慎并标记 needs_review=true。
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
    raw_response_path: str = "",
) -> None:
    traces.append(
        StepTrace(
            step_name=step_name,
            status=status,
            message=message,
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            raw_response=raw_response,
            raw_response_path=raw_response_path,
        )
    )
