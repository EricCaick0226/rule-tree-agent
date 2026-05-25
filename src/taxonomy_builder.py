from __future__ import annotations

import hashlib
import re

from .agent_state import CandidateConcept, ClassificationDimension, DocumentChunk, TreeNode
from .evidence_store import create_evidence_ref, dedupe_evidence_refs


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _clean_label(text: str) -> str:
    label = text.strip()
    label = re.sub(r"^\s*(?:[-*+]|\d+[.)、.．])\s*", "", label)
    label = re.split(r"[:：]", label, maxsplit=1)[0]
    label = label.strip(" ：:；;，,。.!！?？")
    return label


def _is_label(text: str) -> bool:
    if not text or len(text) < 2 or len(text) > 36:
        return False
    if any(term in text for term in ["不包括", "不包含", "不属于", "除外", "排除"]):
        return False
    if re.search(r"[。！？!?]", text):
        return False
    return True


def _skip_taxonomy_chunk(chunk: DocumentChunk) -> bool:
    skip_terms = ["分级", "等级", "对应", "映射", "排除", "除外"]
    return any(term in chunk.section_title for term in skip_terms)


def _bullet_level(line: str) -> int:
    indent = len(line) - len(line.lstrip(" "))
    return indent // 2 + 1


def _split_children(text: str) -> list[str]:
    text = re.split(r"[。.!！?？]", text, maxsplit=1)[0]
    parts = re.split(r"[、,，;；/]|以及|和|及", text)
    labels: list[str] = []
    for part in parts:
        label = _clean_label(part)
        if _is_label(label):
            labels.append(label)
    return labels


def _add_or_merge_node(
    nodes_by_path: dict[str, TreeNode],
    name: str,
    parent: TreeNode | None,
    level: int,
    chunk: DocumentChunk,
    evidence_text: str,
    confidence: float,
    needs_review: bool,
) -> TreeNode:
    path = f"{parent.path} / {name}" if parent else name
    evidence = create_evidence_ref(chunk, f"node:{name}", confidence, text=evidence_text)
    if path in nodes_by_path:
        node = nodes_by_path[path]
        node.evidence_refs = dedupe_evidence_refs([*node.evidence_refs, evidence])
        node.confidence = max(node.confidence, confidence)
        node.needs_review = node.needs_review or needs_review
        return node

    node = TreeNode(
        node_id=_stable_id("node", path),
        name=name,
        path=path,
        level=level,
        parent_id=parent.node_id if parent else None,
        evidence_refs=[evidence],
        confidence=round(confidence, 3),
        needs_review=needs_review,
        status="proposed" if needs_review else "evidence_supported",
    )
    nodes_by_path[path] = node
    return node


def _build_from_bullets(chunks: list[DocumentChunk]) -> list[TreeNode]:
    nodes_by_path: dict[str, TreeNode] = {}
    for chunk in chunks:
        if _skip_taxonomy_chunk(chunk):
            continue
        stack: dict[int, TreeNode] = {}
        for line in chunk.text.splitlines():
            if not re.match(r"^\s*(?:[-*+]|\d+[.)、.．])\s+\S+", line):
                continue
            label = _clean_label(line)
            if not _is_label(label):
                continue
            level = _bullet_level(line)
            parent = stack.get(level - 1)
            node = _add_or_merge_node(
                nodes_by_path,
                label,
                parent,
                level if parent else 1,
                chunk,
                line.strip(),
                0.82 if parent else 0.78,
                False,
            )
            stack[level] = node
            for deeper_level in list(stack):
                if deeper_level > level:
                    stack.pop(deeper_level, None)
    return list(nodes_by_path.values())


def _add_include_patterns(
    nodes: list[TreeNode],
    chunks: list[DocumentChunk],
) -> list[TreeNode]:
    nodes_by_path = {node.path: node for node in nodes}
    name_index = {node.name: node for node in nodes}
    for chunk in chunks:
        if _skip_taxonomy_chunk(chunk):
            continue
        for line in chunk.text.splitlines() or [chunk.text]:
            if any(term in line for term in ["不包括", "不包含", "不属于", "除外", "排除"]):
                continue
            if "包括" not in line and "包含" not in line:
                continue
            match = re.search(r"(.{2,36}?)(?:包括|包含)(.+)", line)
            if not match:
                continue
            parent_name = _clean_label(match.group(1))
            if not _is_label(parent_name):
                continue
            parent = name_index.get(parent_name)
            if parent is None:
                parent = _add_or_merge_node(
                    nodes_by_path,
                    parent_name,
                    None,
                    1,
                    chunk,
                    line.strip(),
                    0.68,
                    True,
                )
                name_index[parent.name] = parent
            for child_name in _split_children(match.group(2)):
                child = _add_or_merge_node(
                    nodes_by_path,
                    child_name,
                    parent,
                    parent.level + 1,
                    chunk,
                    line.strip(),
                    0.7,
                    True,
                )
                name_index[child.name] = child
    return list(nodes_by_path.values())


def _flat_nodes_from_concepts(
    concepts: list[CandidateConcept],
    chunks: list[DocumentChunk],
) -> list[TreeNode]:
    nodes: list[TreeNode] = []
    for concept in concepts:
        if concept.confidence < 0.68:
            continue
        if not ("list_item" in concept.concept_type or "heading" in concept.concept_type):
            continue
        refs = concept.evidence_refs[:2]
        nodes.append(
            TreeNode(
                node_id=_stable_id("node", concept.normalized_text),
                name=concept.text,
                path=concept.text,
                level=1,
                parent_id=None,
                evidence_refs=refs,
                confidence=min(concept.confidence, 0.65),
                needs_review=True,
                status="proposed",
            )
        )
        if len(nodes) >= 12:
            break
    return nodes


def build_taxonomy(
    concepts: list[CandidateConcept],
    selected_dimension: ClassificationDimension | None,
    chunks: list[DocumentChunk],
) -> list[TreeNode]:
    nodes = _build_from_bullets(chunks)
    nodes = _add_include_patterns(nodes, chunks)
    if nodes:
        return sorted(nodes, key=lambda node: (node.path.count(" / "), node.path))

    if selected_dimension is None or selected_dimension.needs_review:
        return _flat_nodes_from_concepts(concepts, chunks)

    return _flat_nodes_from_concepts(concepts, chunks)
