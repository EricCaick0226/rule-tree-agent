from __future__ import annotations

import hashlib
import re

from .agent_state import DocumentChunk, GradeDefinition
from .evidence_store import create_evidence_ref


GRADE_CONTEXT_TERMS = ["分级", "等级", "级别", "等级名称", "grade", "评级"]


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _clean_bullet(line: str) -> str:
    return re.sub(r"^\s*(?:[-*+]|\d+[.)、.．])\s*", "", line).strip()


def _extract_label_definition(line: str) -> tuple[str, str] | None:
    cleaned = _clean_bullet(line)
    if "：" not in cleaned and ":" not in cleaned:
        return None
    label, definition = re.split(r"[:：]", cleaned, maxsplit=1)
    label = label.strip(" ：:；;，,。.!！?？")
    definition = definition.strip()
    if len(label) < 1 or len(label) > 24 or not definition:
        return None
    if re.search(r"[。！？!?]", label):
        return None
    return label, definition


def extract_grade_scheme(chunks: list[DocumentChunk]) -> list[GradeDefinition]:
    grade_defs: dict[str, GradeDefinition] = {}
    for chunk in chunks:
        if any(term in chunk.section_title for term in ["对应", "映射", "关系"]):
            continue
        context_text = f"{chunk.section_title}\n{chunk.text}".lower()
        if not any(term.lower() in context_text for term in GRADE_CONTEXT_TERMS):
            continue
        for line in chunk.text.splitlines() or [chunk.text]:
            parsed = _extract_label_definition(line)
            if not parsed:
                continue
            label, definition = parsed
            if label in grade_defs:
                continue
            evidence = create_evidence_ref(chunk, f"grade_definition:{label}", 0.9, text=line.strip())
            grade_defs[label] = GradeDefinition(
                grade_id=_stable_id("grade", label),
                grade_name=label,
                definition=definition,
                criteria=[definition],
                evidence_refs=[evidence],
                confidence=0.88,
                needs_review=False,
                status="evidence_supported",
            )
    return list(grade_defs.values())
