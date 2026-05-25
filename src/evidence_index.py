from __future__ import annotations

from collections import defaultdict
from typing import Any

from .agent_state import AgentState


def build_evidence_index(state: AgentState) -> AgentState:
    by_doc: dict[str, list[str]] = defaultdict(list)
    by_section: dict[str, list[str]] = defaultdict(list)
    for chunk in state.chunks:
        by_doc[chunk.doc_name].append(chunk.chunk_id)
        by_section[f"{chunk.doc_name}::{chunk.section_title}"].append(chunk.chunk_id)

    state.evidence_index = {
        "chunk_count": len(state.chunks),
        "document_count": len(state.documents),
        "chunk_ids": [chunk.chunk_id for chunk in state.chunks],
        "by_doc": dict(by_doc),
        "by_section": dict(by_section),
    }
    return state
