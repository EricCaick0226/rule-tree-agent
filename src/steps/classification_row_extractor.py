from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ..core.agent_state import AgentState, ClassificationRow, EvidenceRef
from ..io.table_segmenter import TableSegment, segment_table_chunks_for_row_extraction
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    clamp_confidence,
    env_int,
    load_prompt,
    parse_bool,
    refs_from_chunk_ids,
    stable_id,
    string_list,
)


ALLOWED_SUPPORT_LEVELS = {"explicit", "structural", "weak"}
ALLOWED_DESCRIPTION_SOURCES = {"quoted", "summarized", "insufficient"}
INSUFFICIENT_DESCRIPTION = "证据不足，无法从当前文档确定"
SUPPORT_RANK = {"weak": 0, "structural": 1, "explicit": 2}
DESCRIPTION_SOURCE_RANK = {"insufficient": 0, "summarized": 1, "quoted": 2}
ROW_CHECKPOINT_SCHEMA_VERSION = "classification_rows_v2"
ROW_PROMPT_FILE = "extract_classification_rows_prompt.md"


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
    return Path(output_dir).expanduser().resolve() / "checkpoints" / "classification_row_batches.jsonl"


def _segment_signature(
    segments: list[TableSegment],
    prompt_text: str,
    cache_schema_version: str = ROW_CHECKPOINT_SCHEMA_VERSION,
) -> str:
    digest = hashlib.sha1()
    digest.update(cache_schema_version.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt_text.encode("utf-8"))
    digest.update(b"\0")
    for segment in segments:
        digest.update(segment.segment_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(segment.source_chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(segment.header_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(segment.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _make_dataclass(cls, data: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in allowed})


def _rows_from_checkpoint(record: dict[str, Any]) -> list[ClassificationRow]:
    rows: list[ClassificationRow] = []
    for item in record.get("classification_rows") or []:
        if not isinstance(item, dict):
            continue
        row = _make_dataclass(ClassificationRow, {**item, "evidence_refs": []})
        row.evidence_refs = [
            _make_dataclass(EvidenceRef, ref)
            for ref in item.get("evidence_refs") or []
            if isinstance(ref, dict)
        ]
        rows.append(row)
    return rows


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
        if not isinstance(record.get("segment_ids"), list):
            continue
        try:
            batch_index = int(record.get("batch_index") or 0)
        except (TypeError, ValueError):
            continue
        if batch_index > 0:
            records[batch_index] = record
    return records


def _write_checkpoint_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_segment_batches(segments: list[TableSegment], max_chars: int) -> list[list[TableSegment]]:
    max_chars = max(1, int(max_chars or 1))
    batches: list[list[TableSegment]] = []
    current: list[TableSegment] = []
    current_chars = 0

    for segment in segments:
        chars = _segment_payload_chars(segment)
        if current and current_chars + chars > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += chars

    if current:
        batches.append(current)
    return batches


def _segment_payload_chars(segment: TableSegment) -> int:
    return len(json.dumps(_segment_payload([segment]), ensure_ascii=False))


def _segment_payload(segments: list[TableSegment]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": segment.segment_id,
            "source_chunk_id": segment.source_chunk_id,
            "doc_name": segment.doc_name,
            "section_title": segment.section_title,
            "position": segment.position,
            "page_number": segment.page_number,
            "line_start": segment.line_start,
            "line_end": segment.line_end,
            "source_method": segment.source_method,
            "source_warning": segment.source_warning,
            "block_signal": segment.block_signal,
            "header_text": segment.header_text,
            "text": segment.text,
        }
        for segment in segments
    ]


def _payload(segments: list[TableSegment]) -> dict[str, Any]:
    return {
        "task": "从表格/层级文本 segment 中抽取候选分类分级明细行。",
        "table_segments": _segment_payload(segments),
        "allowed_support_levels": sorted(ALLOWED_SUPPORT_LEVELS),
        "description_policy": {
            "insufficient_text": INSUFFICIENT_DESCRIPTION,
            "allowed_description_sources": sorted(ALLOWED_DESCRIPTION_SOURCES),
        },
        "output_schema": {
            "classification_rows": [
                {
                    "path_levels": [],
                    "recommended_grade": None,
                    "description": "",
                    "description_source": "quoted | summarized | insufficient",
                    "description_evidence_quote": "",
                    "evidence_quote": "",
                    "data_range_examples": [],
                    "processing_degree": "",
                    "impact_object": "",
                    "impact_degree": "",
                    "grade_evidence_quote": "",
                    "evidence_chunk_ids": [],
                    "support_level": "explicit | structural | weak",
                    "confidence": 0.0,
                    "needs_review": True,
                    "review_reason": "",
                    "status": "evidence_supported | proposed | insufficient_evidence",
                }
            ]
        },
    }


def _to_rows(data: dict[str, Any], state: AgentState) -> list[ClassificationRow]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in state.chunks}
    rows_by_fingerprint: dict[str, ClassificationRow] = {}

    for item in data.get("classification_rows") or []:
        if not isinstance(item, dict):
            continue

        path_levels = string_list(item.get("path_levels"))
        if not path_levels:
            continue

        recommended_grade_value = item.get("recommended_grade")
        recommended_grade = str(recommended_grade_value).strip() if recommended_grade_value else None

        raw_description = str(item.get("description") or "").strip()
        description = raw_description or INSUFFICIENT_DESCRIPTION
        description_source = str(item.get("description_source") or "").strip()
        if not raw_description or description == INSUFFICIENT_DESCRIPTION:
            description_source = "insufficient"
        if description_source not in ALLOWED_DESCRIPTION_SOURCES:
            description_source = "insufficient" if description == INSUFFICIENT_DESCRIPTION else "summarized"
        if description_source == "insufficient":
            description = INSUFFICIENT_DESCRIPTION

        support_level = str(item.get("support_level") or "").strip()
        if support_level not in ALLOWED_SUPPORT_LEVELS:
            support_level = "weak"

        fingerprint = " / ".join(path_levels) + "|" + str(recommended_grade or "")
        refs = refs_from_chunk_ids(
            chunk_by_id,
            item.get("evidence_chunk_ids") or [],
            "classification_row:" + " / ".join(path_levels),
            0.9,
        )
        needs_review = (
            description_source == "insufficient"
            or not refs
            or support_level in {"structural", "weak"}
            or recommended_grade is None
            or parse_bool(item.get("needs_review"), False)
        )
        review_reason = str(item.get("review_reason") or "").strip()
        if description_source == "insufficient" and not review_reason:
            review_reason = "当前文档未提供该分类项的说明或范围描述。"
        if not refs and not review_reason:
            review_reason = "分类行缺少有效证据引用。"
        status = str(item.get("status") or ("proposed" if needs_review else "evidence_supported"))
        if needs_review and status != "insufficient_evidence":
            status = "proposed"

        row = ClassificationRow(
            row_id=stable_id("row", fingerprint),
            path_levels=path_levels,
            recommended_grade=recommended_grade,
            description=description,
            description_source=description_source,
            description_evidence_quote=str(item.get("description_evidence_quote") or "").strip(),
            evidence_quote=str(item.get("evidence_quote") or "").strip(),
            evidence_refs=refs,
            data_range_examples=string_list(item.get("data_range_examples")),
            processing_degree=str(item.get("processing_degree") or "").strip(),
            impact_object=str(item.get("impact_object") or "").strip(),
            impact_degree=str(item.get("impact_degree") or "").strip(),
            grade_evidence_quote=str(item.get("grade_evidence_quote") or "").strip(),
            support_level=support_level,
            confidence=clamp_confidence(item.get("confidence"), 0.65),
            needs_review=needs_review,
            review_reason=review_reason,
            status=status,
        )
        existing = rows_by_fingerprint.get(fingerprint)
        if existing is None or _row_rank(row) > _row_rank(existing):
            rows_by_fingerprint[fingerprint] = row

    return list(rows_by_fingerprint.values())


def _row_rank(row: ClassificationRow) -> tuple[int, int, int, float]:
    return (
        1 if row.evidence_refs else 0,
        DESCRIPTION_SOURCE_RANK.get(row.description_source, 0),
        SUPPORT_RANK.get(row.support_level, 0),
        row.confidence,
    )


def extract_classification_rows_with_llm(
    state: AgentState,
    llm_client: Any,
    output_dir: str = "outputs",
    segment_max_chars: int | None = None,
) -> AgentState:
    _load_dotenv_if_available()
    max_chars = segment_max_chars if segment_max_chars is not None else env_int("ROW_SEGMENT_MAX_CHARS", 5000)
    batch_max_chars = env_int("ROW_BATCH_MAX_CHARS", 7000)
    checkpoint_enabled = _env_bool("ROW_CHECKPOINT_ENABLED", True)
    resume_enabled = _env_bool("ROW_RESUME", True)
    checkpoint_path = _checkpoint_path(output_dir)

    if not state.chunks:
        state.classification_rows = []
        append_step_trace(
            state.step_traces,
            "extract_classification_rows_with_llm",
            "success",
            "No chunks available for row extraction.",
            {
                "chunks": 0,
                "segments": 0,
                "segment_max_chars": max_chars,
                "batch_max_chars": batch_max_chars,
                "batches": 0,
                "cached_batches": 0,
                "checkpoint_path": str(checkpoint_path) if checkpoint_enabled else "",
                "elapsed_seconds": 0.0,
            },
            {"classification_rows": 0},
        )
        return state

    segments = segment_table_chunks_for_row_extraction(
        state.chunks,
        state.block_signals,
        max_chars=max_chars,
    )
    batches = _build_segment_batches(segments, batch_max_chars)
    prompt_text = load_prompt(ROW_PROMPT_FILE)
    signature = _segment_signature(segments, prompt_text=prompt_text)
    cached_records = (
        _load_checkpoint_records(checkpoint_path, signature)
        if checkpoint_enabled and resume_enabled
        else {}
    )

    all_rows: list[ClassificationRow] = []
    raw_batches: list[str] = []
    seen: set[str] = set()
    step_start = time.monotonic()

    print(
        "Row extraction:",
        f"chunks={len(state.chunks)}",
        f"segments={len(segments)}",
        f"segment_max_chars={max_chars}",
        f"batch_max_chars={batch_max_chars}",
        f"batches={len(batches)}",
        f"resume={'on' if resume_enabled else 'off'}",
    )
    if checkpoint_enabled:
        print(f"Row checkpoint: {checkpoint_path}")

    cached_batch_count = 0
    for batch_index, batch_segments in enumerate(batches, start=1):
        segment_ids = [segment.segment_id for segment in batch_segments]
        batch_chars = sum(_segment_payload_chars(segment) for segment in batch_segments)
        cached = cached_records.get(batch_index)
        if cached and cached.get("segment_ids") == segment_ids:
            cached_batch_count += 1
            batch_rows = _rows_from_checkpoint(cached)
            raw_response = cached.get("raw_response") or "checkpoint hit; raw response not recorded"
            print(
                f"  - row batch {batch_index}/{len(batches)} cached "
                f"segments={len(batch_segments)} rows={len(batch_rows)}"
            )
        else:
            batch_start = time.monotonic()
            print(
                f"  - row batch {batch_index}/{len(batches)} start "
                f"segments={len(batch_segments)} chars={batch_chars}"
            )
            data, raw_response = call_llm_json(
                llm_client=llm_client,
                task_name=f"抽取 classification rows batch {batch_index}/{len(batches)}",
                prompt_file=ROW_PROMPT_FILE,
                payload={
                    **_payload(batch_segments),
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                },
                required_keys={"classification_rows": list},
                max_tokens=env_int("LLM_ROW_MAX_TOKENS", 6000),
                temperature=0.0,
                disable_thinking=True,
            )
            batch_rows = _to_rows(data, state)
            elapsed = round(time.monotonic() - batch_start, 1)
            print(f"  - row batch {batch_index}/{len(batches)} done rows={len(batch_rows)} elapsed={elapsed}s")
            if checkpoint_enabled:
                _write_checkpoint_record(
                    checkpoint_path,
                    {
                        "signature": signature,
                        "cache_schema_version": ROW_CHECKPOINT_SCHEMA_VERSION,
                        "batch_index": batch_index,
                        "batch_count": len(batches),
                        "segment_ids": segment_ids,
                        "source_chunk_ids": [segment.source_chunk_id for segment in batch_segments],
                        "classification_rows": [asdict(row) for row in batch_rows],
                        "raw_response": raw_response,
                        "elapsed_seconds": elapsed,
                    },
                )

        raw_batches.append(f"batch={batch_index}/{len(batches)}\n{raw_response}")
        for row in batch_rows:
            fingerprint = " / ".join(row.path_levels) + "|" + str(row.recommended_grade or "") + "|" + row.evidence_quote
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            all_rows.append(row)

    state.classification_rows = all_rows
    append_step_trace(
        state.step_traces,
        "extract_classification_rows_with_llm",
        "success",
        "",
        {
            "chunks": len(state.chunks),
            "segments": len(segments),
            "segment_max_chars": max_chars,
            "batch_max_chars": batch_max_chars,
            "batches": len(batches),
            "cached_batches": cached_batch_count,
            "checkpoint_path": str(checkpoint_path) if checkpoint_enabled else "",
            "elapsed_seconds": round(time.monotonic() - step_start, 1),
        },
        {"classification_rows": len(state.classification_rows)},
        "\n\n===\n\n".join(raw_batches),
    )
    return state
