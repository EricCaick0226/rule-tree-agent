from __future__ import annotations

import hashlib
import re

from .agent_state import CandidateConcept, ClassificationDimension, DocumentChunk
from .evidence_store import create_evidence_ref


EXPLICIT_PATTERNS = [
    r"按照(.{2,40}?)(?:分类|进行分类|划分)",
    r"根据(.{2,40}?)(?:划分|分类)",
    r"以(.{2,40}?)(?:作为分类依据|为分类依据)",
    r"分类原则[:：]?\s*(.{2,80})",
    r"分类维度[:：]?\s*(.{2,80})",
    r"分类依据[:：]?\s*(.{2,80})",
]


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _clean_dimension(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.split(r"[，,。.;；]", cleaned, maxsplit=1)[0]
    cleaned = cleaned.strip(" ：:；;，,。.!！?？")
    return cleaned


def discover_classification_dimensions(
    concepts: list[CandidateConcept], chunks: list[DocumentChunk]
) -> list[ClassificationDimension]:
    dimensions: list[ClassificationDimension] = []
    seen: set[str] = set()

    for chunk in chunks:
        text = chunk.text
        if not any(term in text for term in ["分类", "划分", "类别", "类型"]):
            continue
        for pattern in EXPLICIT_PATTERNS:
            for match in re.finditer(pattern, text):
                name = _clean_dimension(match.group(1))
                if len(name) < 2 or name in seen:
                    continue
                seen.add(name)
                dimensions.append(
                    ClassificationDimension(
                        dimension_id=_stable_id("dim", name),
                        name=name,
                        description=f"文档明确提到以“{name}”作为分类相关依据。",
                        evidence_refs=[create_evidence_ref(chunk, f"dimension:{name}", 0.92)],
                        reason="发现明确的分类原则、分类依据或划分表述。",
                        confidence=0.9,
                        needs_review=False,
                    )
                )

    if dimensions:
        return dimensions

    top_concepts = [
        concept
        for concept in concepts
        if ("heading" in concept.concept_type or "list_item" in concept.concept_type)
        and concept.confidence >= 0.55
    ][:5]
    if len(top_concepts) >= 2:
        evidence = top_concepts[0].evidence_refs[:1]
        dimensions.append(
            ClassificationDimension(
                dimension_id=_stable_id("dim", "inferred_from_repeated_structures"),
                name="由文档重复结构推断的候选分类维度",
                description="当前文档未明确说明分类依据，仅能从重复出现的标题或列表结构提出候选维度。",
                evidence_refs=evidence,
                reason="未发现明确分类原则；根据多个结构化概念提出弱候选，必须人工确认。",
                confidence=0.45,
                needs_review=True,
            )
        )

    return dimensions


def select_primary_dimension(
    dimensions: list[ClassificationDimension],
) -> ClassificationDimension | None:
    if not dimensions:
        return None
    return sorted(dimensions, key=lambda dim: (dim.needs_review, -dim.confidence))[0]

