from __future__ import annotations

import re

from .agent_state import DocumentChunk, GradeDefinition, TreeNode
from .evidence_store import create_evidence_ref, search_chunks_by_terms


NO_GRADE_REASON = "当前文档证据不足，无法可靠推荐分级。"


def _line_has_mapping(line: str, node_name: str, grade_name: str) -> bool:
    if node_name not in line or grade_name not in line:
        return False
    separators = [":", "：", "为", "对应", "属于", "定为"]
    return any(separator in line for separator in separators)


def assign_grades_to_nodes(
    nodes: list[TreeNode],
    grade_scheme: list[GradeDefinition],
    chunks: list[DocumentChunk],
) -> list[TreeNode]:
    if not grade_scheme:
        for node in nodes:
            node.grade = None
            node.grade_reason = "当前文档未提供可用分级方案，不能为节点分级。"
            node.needs_review = True
        return nodes

    grade_names = [grade.grade_name for grade in grade_scheme]
    for node in nodes:
        assigned = False
        related_chunks = search_chunks_by_terms(chunks, [node.name, *grade_names])
        for chunk in related_chunks:
            for line in chunk.text.splitlines() or [chunk.text]:
                for grade_name in grade_names:
                    if _line_has_mapping(line, node.name, grade_name):
                        node.grade = grade_name
                        node.grade_reason = line.strip()
                        node.grade_evidence_refs = [
                            create_evidence_ref(chunk, f"grade_assignment:{node.name}", 0.9, text=line.strip())
                        ]
                        node.confidence = max(node.confidence, 0.84)
                        assigned = True
                        break
                if assigned:
                    break
            if assigned:
                break

        if not assigned:
            node.grade = None
            node.grade_reason = NO_GRADE_REASON
            node.needs_review = True
    return nodes

