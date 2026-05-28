from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from ..core.agent_state import AgentState
from ..llm.task_utils import (
    LLMJSONValidationError,
    append_step_trace,
    call_llm_json,
    chunk_payload,
    clamp_confidence,
    env_int,
    load_prompt,
    parse_bool,
)


ALLOWED_BLOCK_SIGNALS = {
    "table_like",
    "hierarchy_like",
    "grade_legend",
    "prose_rule",
    "normal",
    "possible_noise",
}

BLOCK_CHECKPOINT_SCHEMA_VERSION = "block_signals_v1"
BLOCK_PROMPT_FILE = "classify_document_blocks_prompt.md"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _env_bool(name: str, default: bool = True) -> bool:
    _load_dotenv_if_available()
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir).expanduser().resolve() / "checkpoints" / "block_signal_batches.jsonl"


def _chunk_batch_signature(chunks: list[Any], prompt_text: str) -> str:
    digest = hashlib.sha1()
    digest.update(BLOCK_CHECKPOINT_SCHEMA_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt_text.encode("utf-8"))
    digest.update(b"\0")
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.position).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


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
        try:
            batch_index = int(record.get("batch_index") or 0)
        except (TypeError, ValueError):
            continue
        if batch_index > 0 and isinstance(record.get("chunk_ids"), list):
            records[batch_index] = record
    return records


def _write_checkpoint_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _chunk_payload_chars(chunk: Any) -> int:
    return len(json.dumps(chunk_payload([chunk]), ensure_ascii=False))


def _build_chunk_batches(chunks: list[Any], max_chars: int) -> list[list[Any]]:
    max_chars = max(1, int(max_chars or 1))
    batches: list[list[Any]] = []
    current: list[Any] = []
    current_chars = 0

    for chunk in chunks:
        chars = _chunk_payload_chars(chunk)
        if current and current_chars + chars > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += chars

    if current:
        batches.append(current)
    return batches


def _payload(chunks: list[Any]) -> dict[str, Any]:
    return {
        "task": "为每个文档 chunk 标注非破坏性的 block_signal，供后续抽取步骤参考。",
        "allowed_block_signals": sorted(ALLOWED_BLOCK_SIGNALS),
        "document_chunks": chunk_payload(chunks),
        "output_schema": {
            "block_signals": [
                {
                    "chunk_id": "必须来自输入 document_chunks",
                    "block_signal": "table_like | hierarchy_like | grade_legend | prose_rule | normal | possible_noise",
                    "reason": "只基于当前 chunk 文本的简短理由",
                    "confidence": 0.0,
                    "needs_review": False,
                    "review_reason": "",
                }
            ]
        },
    }


def _debug_failure_path(output_dir: str, batch_label: str) -> Path:
    safe_label = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in batch_label)
    return Path(output_dir).expanduser().resolve() / "debug" / f"failed_block_batch_{safe_label}.txt"


def _write_failed_block_batch_debug(
    output_dir: str,
    batch_label: str,
    batch_chunks: list[Any],
    exc: Exception,
) -> Path:
    path = _debug_failure_path(output_dir, batch_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_response = getattr(exc, "raw_response", "")
    path.write_text(
        "\n".join(
            [
                f"batch_label={batch_label}",
                f"error={exc}",
                "chunk_ids=" + json.dumps([chunk.chunk_id for chunk in batch_chunks], ensure_ascii=False),
                "",
                "raw_response:",
                str(raw_response or ""),
            ]
        ),
        encoding="utf-8",
    )
    return path


def _normal_signal(reason: str, needs_review: bool = True) -> dict[str, Any]:
    return {
        "block_signal": "normal",
        "reason": reason,
        "confidence": 0.0,
        "needs_review": needs_review,
        "review_reason": "LLM 未返回该 chunk 的可靠 block_signal。" if needs_review else "",
    }


def _fallback_review_reason(existing_reason: str) -> str:
    fallback_reason = "LLM 返回了不允许的 block_signal，已回退为 normal，需要人工复核。"
    existing_reason = existing_reason.strip()
    if not existing_reason:
        return fallback_reason
    if fallback_reason in existing_reason:
        return existing_reason
    return f"{fallback_reason} {existing_reason}"


def _call_block_batch(
    llm_client: Any,
    batch_chunks: list[Any],
    batch_label: str,
    batch_count: int,
) -> tuple[dict[str, Any], str]:
    return call_llm_json(
        llm_client=llm_client,
        task_name=f"classify_document_blocks_with_llm batch {batch_label}/{batch_count}",
        prompt_file=BLOCK_PROMPT_FILE,
        payload={
            **_payload(batch_chunks),
            "batch_index": batch_label,
            "batch_count": batch_count,
        },
        required_keys={"block_signals": list},
        max_tokens=env_int("LLM_BLOCK_MAX_TOKENS", 4000),
        temperature=0.0,
        disable_thinking=True,
    )


def _signals_from_data(data: dict[str, Any], chunks: list[Any]) -> dict[str, dict[str, Any]]:
    known_chunk_ids = {chunk.chunk_id for chunk in chunks}
    block_signals: dict[str, dict[str, Any]] = {}

    for item in data.get("block_signals") or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id not in known_chunk_ids:
            continue
        block_signal = str(item.get("block_signal") or "").strip()
        invalid_signal = block_signal not in ALLOWED_BLOCK_SIGNALS
        if block_signal not in ALLOWED_BLOCK_SIGNALS:
            block_signal = "normal"
        review_reason = str(item.get("review_reason") or "")
        if invalid_signal:
            review_reason = _fallback_review_reason(review_reason)
        needs_review = (
            True
            if invalid_signal
            else parse_bool(item.get("needs_review"), block_signal == "normal")
        )
        block_signals[chunk_id] = {
            "block_signal": block_signal,
            "reason": str(item.get("reason") or ""),
            "confidence": clamp_confidence(item.get("confidence"), 0.0),
            "needs_review": needs_review,
            "review_reason": review_reason,
        }

    for chunk in chunks:
        if chunk.chunk_id not in block_signals:
            block_signals[chunk.chunk_id] = _normal_signal("LLM 未返回该 chunk 的 block_signal。")
    return block_signals


def _classify_batch_with_split_retry(
    llm_client: Any,
    output_dir: str,
    batch_chunks: list[Any],
    batch_label: str,
    batch_count: int,
) -> tuple[dict[str, dict[str, Any]], str, list[str]]:
    try:
        data, raw_response = _call_block_batch(llm_client, batch_chunks, batch_label, batch_count)
        return _signals_from_data(data, batch_chunks), raw_response, []
    except LLMJSONValidationError as exc:
        debug_path = _write_failed_block_batch_debug(output_dir, batch_label, batch_chunks, exc)
        if len(batch_chunks) <= 1:
            fallback = {
                batch_chunks[0].chunk_id: _normal_signal(
                    f"单 chunk block_signal JSON 解析失败，已回退为 normal；debug={debug_path}。"
                )
            }
            return fallback, f"batch={batch_label} failed JSON validation; debug={debug_path}\n{exc}", [str(debug_path)]

        merged_signals: dict[str, dict[str, Any]] = {}
        raw_parts = [
            f"batch={batch_label} failed JSON validation; debug={debug_path}; split_retry=per_chunk",
            str(exc),
        ]
        debug_paths = [str(debug_path)]
        for split_index, chunk in enumerate(batch_chunks, start=1):
            split_label = f"{batch_label}_{split_index}"
            split_signals, raw_response, split_debug_paths = _classify_batch_with_split_retry(
                llm_client=llm_client,
                output_dir=output_dir,
                batch_chunks=[chunk],
                batch_label=split_label,
                batch_count=batch_count,
            )
            merged_signals.update(split_signals)
            debug_paths.extend(split_debug_paths)
            raw_parts.append(f"split_batch={split_label}\n{raw_response}")
        return merged_signals, "\n\n--- split retry ---\n\n".join(raw_parts), debug_paths


def classify_document_blocks_with_llm(
    state: AgentState,
    llm_client: Any,
    output_dir: str = "outputs",
) -> AgentState:
    _load_dotenv_if_available()
    batch_max_chars = env_int("BLOCK_BATCH_MAX_CHARS", 6000)
    checkpoint_enabled = _env_bool("BLOCK_CHECKPOINT_ENABLED", True)
    resume_enabled = _env_bool("BLOCK_RESUME", True)
    checkpoint_path = _checkpoint_path(output_dir)

    if not state.chunks:
        state.block_signals = {}
        append_step_trace(
            state.step_traces,
            step_name="classify_document_blocks_with_llm",
            status="success",
            input_summary={
                "chunks": 0,
                "batch_max_chars": batch_max_chars,
                "batches": 0,
                "cached_batches": 0,
                "checkpoint_path": str(checkpoint_path) if checkpoint_enabled else "",
            },
            output_summary={"block_signals": 0},
        )
        return state

    batches = _build_chunk_batches(state.chunks, batch_max_chars)
    prompt_text = load_prompt(BLOCK_PROMPT_FILE)
    signature = _chunk_batch_signature(state.chunks, prompt_text=prompt_text)
    cached_records = (
        _load_checkpoint_records(checkpoint_path, signature)
        if checkpoint_enabled and resume_enabled
        else {}
    )

    print(
        "Block classification:",
        f"chunks={len(state.chunks)}",
        f"batch_max_chars={batch_max_chars}",
        f"batches={len(batches)}",
        f"resume={'on' if resume_enabled else 'off'}",
    )
    if checkpoint_enabled:
        print(f"Block checkpoint: {checkpoint_path}")

    block_signals: dict[str, dict[str, Any]] = {}
    raw_batches: list[str] = []
    cached_batch_count = 0
    step_start = time.monotonic()

    for batch_index, batch_chunks in enumerate(batches, start=1):
        chunk_ids = [chunk.chunk_id for chunk in batch_chunks]
        batch_chars = sum(_chunk_payload_chars(chunk) for chunk in batch_chunks)
        cached = cached_records.get(batch_index)
        if cached and cached.get("chunk_ids") == chunk_ids:
            cached_batch_count += 1
            batch_signals = {
                str(chunk_id): signal
                for chunk_id, signal in (cached.get("block_signals") or {}).items()
                if isinstance(signal, dict)
            }
            raw_response = cached.get("raw_response") or "checkpoint hit; raw response not recorded"
            print(
                f"  - block batch {batch_index}/{len(batches)} cached "
                f"chunks={len(batch_chunks)} signals={len(batch_signals)}"
            )
        else:
            batch_start = time.monotonic()
            print(
                f"  - block batch {batch_index}/{len(batches)} start "
                f"chunks={len(batch_chunks)} chars={batch_chars}"
            )
            batch_signals, raw_response, debug_paths = _classify_batch_with_split_retry(
                llm_client=llm_client,
                output_dir=output_dir,
                batch_chunks=batch_chunks,
                batch_label=str(batch_index),
                batch_count=len(batches),
            )
            elapsed = round(time.monotonic() - batch_start, 1)
            split_note = f" split_debug={len(debug_paths)}" if debug_paths else ""
            print(
                f"  - block batch {batch_index}/{len(batches)} done "
                f"signals={len(batch_signals)} elapsed={elapsed}s{split_note}"
            )
            if checkpoint_enabled:
                _write_checkpoint_record(
                    checkpoint_path,
                    {
                        "signature": signature,
                        "cache_schema_version": BLOCK_CHECKPOINT_SCHEMA_VERSION,
                        "batch_index": batch_index,
                        "batch_count": len(batches),
                        "chunk_ids": chunk_ids,
                        "block_signals": batch_signals,
                        "raw_response": raw_response,
                        "elapsed_seconds": elapsed,
                        "split_retry_debug_paths": debug_paths,
                    },
                )

        block_signals.update(batch_signals)
        raw_batches.append(f"batch={batch_index}/{len(batches)}\n{raw_response}")

    for chunk in state.chunks:
        if chunk.chunk_id not in block_signals:
            block_signals[chunk.chunk_id] = _normal_signal("LLM 未返回该 chunk 的 block_signal。")

    state.block_signals = block_signals
    append_step_trace(
        state.step_traces,
        step_name="classify_document_blocks_with_llm",
        status="success",
        input_summary={
            "chunks": len(state.chunks),
            "allowed_block_signals": sorted(ALLOWED_BLOCK_SIGNALS),
            "batch_max_chars": batch_max_chars,
            "batches": len(batches),
            "cached_batches": cached_batch_count,
            "checkpoint_path": str(checkpoint_path) if checkpoint_enabled else "",
            "elapsed_seconds": round(time.monotonic() - step_start, 1),
        },
        output_summary={"block_signals": len(block_signals)},
        raw_response="\n\n===\n\n".join(raw_batches),
    )
    return state
