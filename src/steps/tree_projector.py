from __future__ import annotations

from ..core.agent_state import AgentState, ClassificationRow, TreeNode
from ..llm.task_utils import append_step_trace, stable_id


INSUFFICIENT_GRADE_REASON = "证据不足，无法从当前文档确定推荐分级。"


def _clean_path_levels(row: ClassificationRow) -> list[str]:
    return [level.strip() for level in row.path_levels if level.strip()]


def project_tree_from_rows(state: AgentState) -> AgentState:
    nodes_by_path: dict[str, TreeNode] = {}

    for row in state.classification_rows:
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

            parent_id = node.node_id

        if not path_parts:
            continue

        leaf = nodes_by_path[" / ".join(path_parts)]
        has_grade = bool(row.recommended_grade)
        leaf.grade = row.recommended_grade
        leaf.description = row.description
        leaf.description_evidence_refs = list(row.evidence_refs)
        leaf.grade_evidence_refs = list(row.evidence_refs) if has_grade else []
        leaf.grade_reason = "" if has_grade else INSUFFICIENT_GRADE_REASON
        leaf.needs_review = row.needs_review or not has_grade
        leaf.status = row.status
        leaf.confidence = row.confidence

    state.nodes = sorted(nodes_by_path.values(), key=lambda node: node.path)
    append_step_trace(
        state.step_traces,
        "project_tree_from_rows",
        "success",
        "",
        {
            "classification_rows": len(state.classification_rows),
            "row_ids": [row.row_id for row in state.classification_rows],
        },
        {
            "nodes": len(state.nodes),
            "node_paths": [node.path for node in state.nodes],
        },
    )
    return state
