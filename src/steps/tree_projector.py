from __future__ import annotations

from ..core.agent_state import AgentState, ClassificationRow, EvidenceRef, TreeNode
from ..llm.task_utils import append_step_trace, stable_id


INSUFFICIENT_GRADE_REASON = "证据不足，无法从当前文档确定推荐分级。"


def _clean_path_levels(row: ClassificationRow) -> list[str]:
    return [level.strip() for level in row.path_levels if level.strip()]


def _merge_evidence_refs(existing: list[EvidenceRef], additions: list[EvidenceRef]) -> list[EvidenceRef]:
    merged: list[EvidenceRef] = []
    seen_evidence_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()

    for ref in [*existing, *additions]:
        if ref.evidence_id in seen_evidence_ids or ref.chunk_id in seen_chunk_ids:
            continue
        if ref.evidence_id:
            seen_evidence_ids.add(ref.evidence_id)
        if ref.chunk_id:
            seen_chunk_ids.add(ref.chunk_id)
        merged.append(ref)

    return merged


def _merge_confidence(existing: float, candidate: float) -> float:
    if existing == 0:
        return candidate
    if candidate == 0:
        return existing
    return min(existing, candidate)


def _merge_status(existing: str, candidate: str) -> str:
    if existing != "evidence_supported" or candidate != "evidence_supported":
        return "proposed"
    return "evidence_supported"


def _merge_row_into_node(node: TreeNode, row: ClassificationRow) -> None:
    node.evidence_refs = _merge_evidence_refs(node.evidence_refs, row.evidence_refs)
    node.needs_review = node.needs_review or row.needs_review
    node.confidence = _merge_confidence(node.confidence, row.confidence)
    node.status = _merge_status(node.status, row.status)


def project_tree_from_rows(state: AgentState) -> AgentState:
    nodes_by_path: dict[str, TreeNode] = {}
    projected_rows = [row for row in state.classification_rows if row.inclusion_status == "accepted"]

    for row in projected_rows:
        path_levels = _clean_path_levels(row)
        parent_id: str | None = None
        path_parts: list[str] = []

        for index, level_name in enumerate(path_levels, start=1):
            path_parts.append(level_name)
            path = " / ".join(path_parts)
            node = nodes_by_path.get(path)

            if node is None:
                node = TreeNode(
                    node_id=stable_id("node", path),
                    name=level_name,
                    path=path,
                    level=index,
                    parent_id=parent_id,
                    confidence=row.confidence,
                    needs_review=row.needs_review,
                    status=row.status,
                    evidence_refs=list(row.evidence_refs),
                )
                nodes_by_path[path] = node
            else:
                _merge_row_into_node(node, row)

            parent_id = node.node_id

        if not path_parts:
            continue

        leaf = nodes_by_path[" / ".join(path_parts)]
        has_grade = bool(row.recommended_grade)
        merged_refs = _merge_evidence_refs(leaf.evidence_refs, row.evidence_refs)
        leaf.grade = row.recommended_grade
        leaf.description = row.description
        leaf.evidence_refs = merged_refs
        leaf.description_evidence_refs = _merge_evidence_refs(
            leaf.description_evidence_refs,
            row.evidence_refs,
        )
        leaf.grade_evidence_refs = (
            _merge_evidence_refs(leaf.grade_evidence_refs, row.evidence_refs)
            if has_grade
            else []
        )
        leaf.grade_reason = "" if has_grade else INSUFFICIENT_GRADE_REASON
        leaf.needs_review = leaf.needs_review or row.needs_review or not has_grade
        leaf.status = _merge_status(leaf.status, row.status)
        leaf.confidence = _merge_confidence(leaf.confidence, row.confidence)

    state.nodes = sorted(nodes_by_path.values(), key=lambda node: node.path)
    append_step_trace(
        state.step_traces,
        "project_tree_from_rows",
        "success",
        "",
        {
            "classification_rows": len(state.classification_rows),
            "projected_rows": len(projected_rows),
            "row_ids": [row.row_id for row in state.classification_rows],
        },
        {
            "nodes": len(state.nodes),
            "node_paths": [node.path for node in state.nodes],
        },
    )
    return state
