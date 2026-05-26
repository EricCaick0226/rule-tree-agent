from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core.agent_state import AgentState


def build_evidence_index(state: AgentState) -> AgentState:
    by_doc: dict[str, list[str]] = defaultdict(list)
    by_section: dict[str, list[str]] = defaultdict(list)
    for chunk in state.chunks:
        by_doc[chunk.doc_name].append(chunk.chunk_id)
        by_section[f"{chunk.doc_name}::{chunk.section_title}"].append(chunk.chunk_id)

    state.evidence_index = {
        "chunk_count": len(state.chunks),
        "document_count": len(state.documents),
        "pdf_ocr_enabled": state.pdf_ocr_enabled,
        "source_method_counts": {
            method: sum(1 for chunk in state.chunks if chunk.source_method == method)
            for method in sorted({chunk.source_method for chunk in state.chunks})
        },
        "chunk_ids": [chunk.chunk_id for chunk in state.chunks],
        "by_doc": dict(by_doc),
        "by_section": dict(by_section),
    }
    return state
