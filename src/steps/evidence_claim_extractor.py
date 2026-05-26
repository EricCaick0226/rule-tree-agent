from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ..core.agent_state import AgentState, EvidenceClaim, EvidenceRef
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


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir).expanduser().resolve() / "checkpoints" / "evidence_claim_batches.jsonl"


def _chunk_signature(chunks: list) -> str:
    digest = hashlib.sha1()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.doc_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _make_dataclass(cls, data: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in allowed})


def _claims_from_checkpoint(record: dict[str, Any]) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    for item in record.get("claims") or []:
        claim = _make_dataclass(EvidenceClaim, {**item, "evidence_refs": []})
        claim.evidence_refs = [
            _make_dataclass(EvidenceRef, ref)
            for ref in item.get("evidence_refs") or []
            if isinstance(ref, dict)
        ]
        claims.append(claim)
    return claims


def _load_checkpoint_records(path: Path, signature: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("signature") != signature:
            continue
        batch_index = int(record.get("batch_index") or 0)
        if batch_index > 0:
            records[batch_index] = record
    return records


def _write_checkpoint_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_claim_batches(chunks: list, max_chunks: int, max_chars: int) -> list[list]:
    max_chunks = max(1, int(max_chunks or 1))
    max_chars = max(0, int(max_chars or 0))
    if max_chars <= 0:
        return [
            chunks[index : index + max_chunks]
            for index in range(0, len(chunks), max_chunks)
        ]

    batches: list[list] = []
    current: list = []
    current_chars = 0

    for chunk in chunks:
        chunk_chars = len(chunk.text or "")
        oversized = chunk_chars > max_chars
        if current and (
            len(current) >= max_chunks
            or current_chars + chunk_chars > max_chars
            or oversized
        ):
            batches.append(current)
            current = []
            current_chars = 0

        if oversized:
            batches.append([chunk])
            continue

        current.append(chunk)
        current_chars += chunk_chars

    if current:
        batches.append(current)
    return batches


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


def extract_evidence_claims_with_llm(
    state: AgentState,
    llm_client: Any,
    output_dir: str = "outputs",
) -> AgentState:
    _load_dotenv_if_available()
    batch_size = max(1, env_int("CLAIM_BATCH_SIZE", 8))
    batch_max_chars = max(0, env_int("CLAIM_BATCH_MAX_CHARS", 6000))
    batching_mode = "char_budget" if batch_max_chars > 0 else "fixed_chunk_count"
    checkpoint_enabled = _env_bool("CLAIM_CHECKPOINT_ENABLED", True)
    resume_enabled = _env_bool("CLAIM_RESUME", True)
    checkpoint_path = _checkpoint_path(output_dir)
    signature = _chunk_signature(state.chunks)
    cached_records = (
        _load_checkpoint_records(checkpoint_path, signature)
        if checkpoint_enabled and resume_enabled
        else {}
    )
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
            input_summary={
                "chunks": 0,
                "batch_size": batch_size,
                "batch_max_chars": batch_max_chars,
                "batching_mode": batching_mode,
                "batches": 0,
            },
            output_summary={"claims": 0},
        )
        return state

    batches = _build_claim_batches(state.chunks, batch_size, batch_max_chars)
    print(
        "Claim extraction:",
        f"chunks={len(state.chunks)}",
        f"batch_size={batch_size}",
        f"batch_max_chars={batch_max_chars}",
        f"mode={batching_mode}",
        f"batches={len(batches)}",
        f"resume={'on' if resume_enabled else 'off'}",
    )
    if checkpoint_enabled:
        print(f"Claim checkpoint: {checkpoint_path}")
    step_start = time.monotonic()

    for batch_index, batch_chunks in enumerate(batches, start=1):
        chunk_ids = [chunk.chunk_id for chunk in batch_chunks]
        batch_chars = sum(len(chunk.text) for chunk in batch_chunks)
        cached = cached_records.get(batch_index)
        if cached and cached.get("chunk_ids") == chunk_ids:
            batch_claims = _claims_from_checkpoint(cached)
            raw_response = cached.get("raw_response") or "checkpoint hit; raw response not recorded"
            print(
                f"  - claims batch {batch_index}/{len(batches)} cached",
                f"chunks={len(batch_chunks)}",
                f"chars={batch_chars}",
                f"chunk_ids={chunk_ids[0]}..{chunk_ids[-1]}",
                f"claims={len(batch_claims)}",
            )
        else:
            batch_start = time.monotonic()
            print(
                f"  - claims batch {batch_index}/{len(batches)} start",
                f"chunks={len(batch_chunks)}",
                f"chars={batch_chars}",
                f"chunk_ids={chunk_ids[0]}..{chunk_ids[-1]}",
            )
            try:
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
            except Exception:
                elapsed = round(time.monotonic() - batch_start, 1)
                print(f"  - claims batch {batch_index}/{len(batches)} failed elapsed={elapsed}s")
                raise
            batch_claims = _to_claims(data, state)
            elapsed = round(time.monotonic() - batch_start, 1)
            print(
                f"  - claims batch {batch_index}/{len(batches)} done",
                f"claims={len(batch_claims)}",
                f"elapsed={elapsed}s",
            )
            if checkpoint_enabled:
                _write_checkpoint_record(
                    checkpoint_path,
                    {
                        "signature": signature,
                        "batch_index": batch_index,
                        "batch_count": len(batches),
                        "chunk_ids": chunk_ids,
                        "claims": [asdict(claim) for claim in batch_claims],
                        "raw_response": raw_response,
                        "elapsed_seconds": elapsed,
                    },
                )

        raw_batches.append(f"batch={batch_index}/{len(batches)}\n{raw_response}")
        for claim in batch_claims:
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
        input_summary={
            "chunks": len(state.chunks),
            "batch_size": batch_size,
            "batch_max_chars": batch_max_chars,
            "batching_mode": batching_mode,
            "batches": len(batches),
            "cached_batches": sum(
                1
                for index, batch_chunks in enumerate(batches, start=1)
                if cached_records.get(index)
                and cached_records[index].get("chunk_ids") == [chunk.chunk_id for chunk in batch_chunks]
            ),
            "checkpoint_path": str(checkpoint_path) if checkpoint_enabled else "",
            "elapsed_seconds": round(time.monotonic() - step_start, 1),
        },
        output_summary={"claims": len(state.evidence_claims)},
        raw_response="\n\n===\n\n".join(raw_batches),
    )
    return state
