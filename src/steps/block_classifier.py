from __future__ import annotations

from typing import Any

from ..core.agent_state import AgentState
from ..llm.task_utils import (
    append_step_trace,
    call_llm_json,
    chunk_payload,
    clamp_confidence,
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


def _payload(state: AgentState) -> dict[str, Any]:
    return {
        "task": "为每个文档 chunk 标注非破坏性的 block_signal，供后续抽取步骤参考。",
        "allowed_block_signals": sorted(ALLOWED_BLOCK_SIGNALS),
        "document_chunks": chunk_payload(state.chunks),
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


def _normal_signal(reason: str, needs_review: bool = True) -> dict[str, Any]:
    return {
        "block_signal": "normal",
        "reason": reason,
        "confidence": 0.0,
        "needs_review": needs_review,
        "review_reason": "LLM 未返回该 chunk 的可靠 block_signal。" if needs_review else "",
    }


def classify_document_blocks_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    if not state.chunks:
        state.block_signals = {}
        append_step_trace(
            state.step_traces,
            step_name="classify_document_blocks_with_llm",
            status="success",
            input_summary={"chunks": 0},
            output_summary={"block_signals": 0},
        )
        return state

    data, raw_response = call_llm_json(
        llm_client=llm_client,
        task_name="classify_document_blocks_with_llm",
        prompt_file="classify_document_blocks_prompt.md",
        payload=_payload(state),
        required_keys={"block_signals": list},
        temperature=0.0,
        disable_thinking=True,
    )

    known_chunk_ids = {chunk.chunk_id for chunk in state.chunks}
    block_signals: dict[str, dict[str, Any]] = {}

    for item in data.get("block_signals") or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id not in known_chunk_ids:
            continue
        block_signal = str(item.get("block_signal") or "").strip()
        if block_signal not in ALLOWED_BLOCK_SIGNALS:
            block_signal = "normal"
        block_signals[chunk_id] = {
            "block_signal": block_signal,
            "reason": str(item.get("reason") or ""),
            "confidence": clamp_confidence(item.get("confidence"), 0.0),
            "needs_review": parse_bool(item.get("needs_review"), block_signal == "normal"),
            "review_reason": str(item.get("review_reason") or ""),
        }

    for chunk in state.chunks:
        if chunk.chunk_id not in block_signals:
            block_signals[chunk.chunk_id] = _normal_signal("LLM 未返回该 chunk 的 block_signal。")

    state.block_signals = block_signals
    append_step_trace(
        state.step_traces,
        step_name="classify_document_blocks_with_llm",
        status="success",
        input_summary={"chunks": len(state.chunks), "allowed_block_signals": sorted(ALLOWED_BLOCK_SIGNALS)},
        output_summary={"block_signals": len(block_signals)},
        raw_response=raw_response,
    )
    return state
