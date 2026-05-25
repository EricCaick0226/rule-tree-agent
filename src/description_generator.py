from __future__ import annotations

import re

from .agent_state import DocumentChunk, TreeNode
from .evidence_store import create_evidence_ref, search_chunks_by_terms


INSUFFICIENT_DESCRIPTION = "当前文档未提供该分类的明确说明，建议人工确认其定义、范围和适用边界。"


def _line_label(line: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)、.．])\s*", "", line).strip()
    return re.split(r"[:：]", cleaned, maxsplit=1)[0].strip()


def _sentences_with_term(text: str, term: str) -> list[str]:
    lines = []
    for line in text.splitlines() or [text]:
        stripped = line.strip()
        if term in stripped:
            lines.append(stripped)
    if lines:
        return lines
    pieces = re.split(r"(?<=[。.!！?？])", text)
    return [piece.strip() for piece in pieces if term in piece and piece.strip()]


def _description_from_line(line: str, node_name: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)、.．])\s*", "", line).strip()
    if "：" in cleaned or ":" in cleaned:
        left, right = re.split(r"[:：]", cleaned, maxsplit=1)
        if node_name in left and right.strip():
            return right.strip()
    return cleaned


def generate_node_descriptions(nodes: list[TreeNode], chunks: list[DocumentChunk]) -> list[TreeNode]:
    for node in nodes:
        related_chunks = [
            chunk
            for chunk in search_chunks_by_terms(chunks, [node.name])
            if not any(term in chunk.section_title for term in ["分级", "等级", "对应", "映射"])
        ]
        evidence_lines: list[tuple[DocumentChunk, str]] = []
        for chunk in related_chunks:
            for line in _sentences_with_term(chunk.text, node.name):
                evidence_lines.append((chunk, line))

        if not evidence_lines:
            node.description = INSUFFICIENT_DESCRIPTION
            node.description_evidence_refs = []
            node.description_evidence_level = "D"
            node.needs_review = True
            continue

        evidence_lines.sort(
            key=lambda item: (
                _line_label(item[1]) != node.name,
                item[0].position,
            )
        )
        first_chunk, first_line = evidence_lines[0]
        node.description = _description_from_line(first_line, node.name)
        refs = [
            create_evidence_ref(chunk, f"description:{node.name}", 0.85, text=line)
            for chunk, line in evidence_lines[:3]
        ]
        node.description_evidence_refs = refs
        if any((":" in line or "：" in line or "定义" in line or "包括" in line) for _, line in evidence_lines):
            node.description_evidence_level = "A"
            node.confidence = max(node.confidence, 0.82)
        elif len(evidence_lines) >= 2:
            node.description_evidence_level = "B"
            node.confidence = max(node.confidence, 0.74)
        else:
            node.description_evidence_level = "C"
            node.needs_review = True
        if node.description_evidence_level in {"C", "D"}:
            node.needs_review = True
    return nodes
