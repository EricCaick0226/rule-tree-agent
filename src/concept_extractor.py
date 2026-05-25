from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from .agent_state import CandidateConcept, DocumentChunk, EvidenceRef
from .evidence_store import create_evidence_ref, dedupe_evidence_refs


STRUCTURAL_TRIGGERS = [
    "包括",
    "包含",
    "分为",
    "划分为",
    "类别",
    "类型",
    "等级",
    "范围",
    "定义",
    "规则",
    "信息",
    "数据",
    "资源",
    "资料",
]


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _clean_candidate(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)、.．])\s*", "", cleaned)
    cleaned = cleaned.strip(" ：:；;，,。.!！?？()（）[]【】")
    cleaned = re.sub(r"^(应|可|可以|必须|不得|不应)", "", cleaned).strip()
    return cleaned


def _is_candidate(text: str) -> bool:
    if not text or len(text) < 2 or len(text) > 40:
        return False
    if re.search(r"[。！？!?]$", text):
        return False
    if text in STRUCTURAL_TRIGGERS:
        return False
    if re.fullmatch(r"[\W_]+", text):
        return False
    return True


def _split_items(text: str) -> list[str]:
    parts = re.split(r"[、,，;；/]|以及|和|及", text)
    return [_clean_candidate(part) for part in parts if _is_candidate(_clean_candidate(part))]


def _extract_from_line(line: str, section_title: str) -> list[tuple[str, str]]:
    stripped = line.strip()
    results: list[tuple[str, str]] = []
    if not stripped:
        return results

    if re.match(r"^#{1,6}\s+\S+", stripped):
        results.append((_clean_candidate(re.sub(r"^#{1,6}\s+", "", stripped)), "heading"))
        return results

    if re.match(r"^\s*(?:[-*+]|\d+[.)、.．])\s+\S+", line):
        item = _clean_candidate(line)
        label = re.split(r"[:：]", item, maxsplit=1)[0].strip()
        if _is_candidate(label):
            results.append((label, "list_item"))

    if section_title and section_title != "未命名章节":
        title = _clean_candidate(section_title)
        if _is_candidate(title):
            results.append((title, "section_title"))

    for trigger in ["包括", "包含", "分为", "划分为"]:
        if trigger in stripped:
            right = stripped.split(trigger, 1)[1]
            right = re.split(r"[。.!！?？]", right, maxsplit=1)[0]
            for item in _split_items(right):
                results.append((item, "trigger_phrase"))

    quoted = re.findall(r"[“\"']([^”\"']{2,40})[”\"']", stripped)
    for item in quoted:
        cleaned = _clean_candidate(item)
        if _is_candidate(cleaned):
            results.append((cleaned, "quoted_term"))

    if any(trigger in stripped for trigger in STRUCTURAL_TRIGGERS):
        left = re.split(r"[:：]", stripped, maxsplit=1)[0].strip()
        left = _clean_candidate(left)
        if _is_candidate(left):
            results.append((left, "structural_line"))

    return results


def extract_concepts(chunks: list[DocumentChunk]) -> list[CandidateConcept]:
    evidence_by_norm: dict[str, list[EvidenceRef]] = defaultdict(list)
    text_by_norm: dict[str, str] = {}
    types_by_norm: dict[str, set[str]] = defaultdict(set)
    frequency: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        lines = chunk.text.splitlines() or [chunk.text]
        for line in lines:
            for text, concept_type in _extract_from_line(line, chunk.section_title):
                cleaned = _clean_candidate(text)
                if not _is_candidate(cleaned):
                    continue
                norm = _normalize(cleaned)
                text_by_norm.setdefault(norm, cleaned)
                types_by_norm[norm].add(concept_type)
                frequency[norm] += 1
                evidence_by_norm[norm].append(
                    create_evidence_ref(chunk, f"concept:{cleaned}", 0.8, text=line.strip())
                )

    concepts: list[CandidateConcept] = []
    for norm, text in sorted(text_by_norm.items(), key=lambda item: item[1]):
        types = types_by_norm[norm]
        refs = dedupe_evidence_refs(evidence_by_norm[norm])
        structural_bonus = 0.2 if {"heading", "section_title", "list_item"} & types else 0.0
        confidence = min(0.95, 0.35 + min(frequency[norm], 5) * 0.08 + structural_bonus)
        concepts.append(
            CandidateConcept(
                concept_id=_stable_id("concept", norm),
                text=text,
                normalized_text=norm,
                concept_type="+".join(sorted(types)),
                evidence_refs=refs,
                confidence=round(confidence, 3),
                needs_review=confidence < 0.7,
            )
        )

    return concepts

