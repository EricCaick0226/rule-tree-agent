from __future__ import annotations

import hashlib
import re

from .agent_state import DocumentChunk, MatchingRule, TreeNode
from .evidence_store import create_evidence_ref, search_chunks_by_terms


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _terms_from_line(line: str, node_name: str) -> list[str]:
    cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)、.．])\s*", "", line).strip()
    terms = [node_name]
    if "：" in cleaned or ":" in cleaned:
        left, right = re.split(r"[:：]", cleaned, maxsplit=1)
        if node_name in left:
            for part in re.split(r"[、,，;；/]|以及|和|及", right):
                term = part.strip(" ：:；;，,。.!！?？")
                if 2 <= len(term) <= 28 and not re.search(r"[。！？!?]", term):
                    terms.append(term)
    quoted = re.findall(r"[“\"']([^”\"']{2,40})[”\"']", cleaned)
    terms.extend(term.strip() for term in quoted)
    return list(dict.fromkeys(terms))


def _line_label(line: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)、.．])\s*", "", line).strip()
    return re.split(r"[:：]", cleaned, maxsplit=1)[0].strip()


def _negative_terms(line: str) -> list[str]:
    if not any(term in line for term in ["不包括", "除外", "排除", "不属于"]):
        return []
    tail = re.split(r"不包括|除外|排除|不属于", line, maxsplit=1)[-1]
    results: list[str] = []
    for part in re.split(r"[、,，;；/]|以及|和|及", tail):
        term = part.strip(" ：:；;，,。.!！?？")
        if 2 <= len(term) <= 28:
            results.append(term)
    return results


def generate_node_rules(nodes: list[TreeNode], chunks: list[DocumentChunk]) -> list[TreeNode]:
    for node in nodes:
        related_chunks = [
            chunk
            for chunk in search_chunks_by_terms(chunks, [node.name])
            if not any(term in chunk.section_title for term in ["分级", "等级", "对应", "映射"])
        ]
        evidence_lines: list[tuple[DocumentChunk, str]] = []
        conditions: list[str] = []
        negative_conditions: list[str] = []

        for chunk in related_chunks:
            for line in chunk.text.splitlines() or [chunk.text]:
                if node.name not in line:
                    continue
                evidence_lines.append((chunk, line.strip()))
                negative_conditions.extend(_negative_terms(line))

        exact_lines = [
            (chunk, line)
            for chunk, line in evidence_lines
            if _line_label(line) == node.name and not _negative_terms(line)
        ]
        source_lines = exact_lines or evidence_lines
        for _, line in source_lines:
            conditions.extend(_terms_from_line(line, node.name))

        conditions = list(dict.fromkeys(term for term in conditions if term))
        negative_conditions = list(dict.fromkeys(term for term in negative_conditions if term))
        if not evidence_lines or not conditions:
            node.rules = [
                MatchingRule(
                    rule_id=_stable_id("rule", f"{node.node_id}:insufficient"),
                    target_node_id=node.node_id,
                    rule_type="keyword_rule",
                    conditions=[],
                    negative_conditions=[],
                    evidence_refs=[],
                    confidence=0.0,
                    needs_review=True,
                    status="insufficient_evidence",
                )
            ]
            node.needs_review = True
            continue

        refs = [
            create_evidence_ref(chunk, f"rule:{node.name}", 0.82, text=line)
            for chunk, line in source_lines[:3]
        ]
        rule_type = "phrase_rule" if any(len(term) > 8 for term in conditions) else "keyword_rule"
        rule = MatchingRule(
            rule_id=_stable_id("rule", f"{node.node_id}:{'|'.join(conditions)}"),
            target_node_id=node.node_id,
            rule_type=rule_type,
            conditions=conditions,
            negative_conditions=negative_conditions,
            evidence_refs=refs,
            confidence=0.78 if len(conditions) > 1 else 0.62,
            needs_review=len(conditions) <= 1,
            status="proposed" if len(conditions) <= 1 else "evidence_supported",
        )
        node.rules = [rule]
        if rule.needs_review:
            node.needs_review = True
    return nodes
