from __future__ import annotations

import hashlib
from difflib import SequenceMatcher

from ..core.agent_state import DocumentChunk, EvidenceRef


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def create_evidence_ref(
    chunk: DocumentChunk,
    used_for: str,
    relevance_score: float,
    text: str | None = None,
) -> EvidenceRef:
    evidence_text = (text or chunk.text).strip()
    return EvidenceRef(
        evidence_id=_stable_id("ev", chunk.chunk_id, used_for, evidence_text),
        chunk_id=chunk.chunk_id,
        doc_name=chunk.doc_name,
        section_title=chunk.section_title,
        text=evidence_text,
        used_for=used_for,
        relevance_score=round(float(relevance_score), 3),
        page_number=chunk.page_number,
        source_method=chunk.source_method,
        source_warning=chunk.source_warning,
    )


def dedupe_evidence_refs(refs: list[EvidenceRef]) -> list[EvidenceRef]:
    seen: set[str] = set()
    result: list[EvidenceRef] = []
    for ref in refs:
        if ref.evidence_id in seen:
            continue
        seen.add(ref.evidence_id)
        result.append(ref)
    return result


def search_chunks_by_terms(chunks: list[DocumentChunk], terms: list[str]) -> list[DocumentChunk]:
    normalized_terms = [term.strip().lower() for term in terms if term and term.strip()]
    if not normalized_terms:
        return []

    scored: list[tuple[float, DocumentChunk]] = []
    for chunk in chunks:
        text = chunk.text.lower()
        title = chunk.section_title.lower()
        score = 0.0
        for term in normalized_terms:
            if term in text:
                score += 1.0
            if term in title:
                score += 0.6
            if term not in text:
                score += max(
                    SequenceMatcher(None, term, line.lower()).ratio()
                    for line in chunk.text.splitlines() or [chunk.text]
                ) * 0.25
        if score >= 0.45:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].position))
    return [chunk for _, chunk in scored]


def collect_unique_evidence(*groups: list[EvidenceRef]) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for group in groups:
        refs.extend(group)
    return dedupe_evidence_refs(refs)
