from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .agent_state import AgentState, TreeNode


def _node_lines(nodes: list[TreeNode]) -> list[str]:
    lines: list[str] = []
    children_by_parent: dict[str | None, list[TreeNode]] = {}
    for node in nodes:
        children_by_parent.setdefault(node.parent_id, []).append(node)

    def render(node: TreeNode) -> None:
        indent = "  " * max(node.level - 1, 0)
        grade = f" | grade: {node.grade}" if node.grade else ""
        review = " | needs_review" if node.needs_review else ""
        lines.append(f"{indent}- {node.name}{grade}{review}")
        if node.description:
            lines.append(f"{indent}  - description: {node.description}")
        if node.rules:
            for rule in node.rules:
                if rule.conditions:
                    lines.append(f"{indent}  - rule: {', '.join(rule.conditions)}")
                else:
                    lines.append(f"{indent}  - rule: insufficient evidence")
        for child in sorted(children_by_parent.get(node.node_id, []), key=lambda item: item.path):
            render(child)

    for root in sorted(children_by_parent.get(None, []), key=lambda item: item.path):
        render(root)
    return lines


def _review_report(state: AgentState) -> str:
    nodes_with_evidence = sum(1 for node in state.nodes if node.evidence_refs)
    review_nodes = sum(1 for node in state.nodes if node.needs_review)
    unsupported = [
        issue
        for issue in state.validation_issues
        if issue.issue_type in {"unsupported_generation", "hardcoded_or_ungrounded_content"}
    ]
    low_confidence = [node for node in state.nodes if node.confidence < 0.6]
    missing_grading = [
        issue for issue in state.validation_issues if issue.issue_type == "missing_grade_scheme"
    ]

    lines = [
        "# Human Review Report",
        "",
        "## Summary",
        f"- Classification dimension found: {'yes' if state.selected_dimension else 'no'}",
        f"- Selected dimension: {state.selected_dimension.name if state.selected_dimension else 'cannot determine from current documents'}",
        f"- Grade scheme found: {'yes' if state.grade_scheme else 'no'}",
        f"- Number of nodes: {len(state.nodes)}",
        f"- Nodes with evidence: {nodes_with_evidence}",
        f"- Nodes requiring review: {review_nodes}",
        "",
        "## Unsupported Generation Issues",
    ]
    if unsupported:
        lines.extend(
            f"- [{issue.severity}] {issue.target}: {issue.problem}"
            for issue in unsupported
        )
    else:
        lines.append("- None detected.")

    lines.extend(["", "## Low Confidence Nodes"])
    if low_confidence:
        lines.extend(f"- {node.path} (confidence={node.confidence})" for node in low_confidence)
    else:
        lines.append("- None detected.")

    lines.extend(["", "## Missing Grading Issues"])
    if missing_grading:
        lines.extend(f"- {issue.problem}" for issue in missing_grading)
    elif not state.grade_scheme:
        lines.append("- No grade scheme was found in the documents.")
    else:
        ungraded = [node for node in state.nodes if node.grade is None]
        if ungraded:
            lines.extend(f"- {node.path}: {node.grade_reason}" for node in ungraded)
        else:
            lines.append("- None detected.")

    lines.extend(["", "## Suggested Human Review Actions"])
    if state.validation_issues:
        for issue in state.validation_issues:
            lines.append(f"- {issue.target}: {issue.suggested_action}")
    else:
        lines.append("- Review the candidate tree before adopting it as an enterprise standard.")

    return "\n".join(lines) + "\n"


def export_outputs(state: AgentState, output_dir: str) -> AgentState:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "rule_tree.json"
    md_path = out_dir / "rule_tree.md"
    report_path = out_dir / "review_report.md"
    state.output_paths = {
        "rule_tree_json": str(json_path),
        "rule_tree_md": str(md_path),
        "review_report_md": str(report_path),
    }

    json_path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tree_lines = [
        "# Candidate Rule Tree",
        "",
        "> This is a candidate output. It requires human review before use.",
        "",
    ]
    if state.selected_dimension:
        tree_lines.append(f"- Selected dimension: {state.selected_dimension.name}")
    else:
        tree_lines.append("- Selected dimension: insufficient evidence")
    tree_lines.append(f"- Grade scheme: {'found' if state.grade_scheme else 'insufficient evidence'}")
    tree_lines.extend(["", "## Tree", ""])
    if state.nodes:
        tree_lines.extend(_node_lines(state.nodes))
    else:
        tree_lines.append("Insufficient evidence to build a candidate tree.")
    tree_lines.append("")
    md_path.write_text("\n".join(tree_lines), encoding="utf-8")

    report_path.write_text(_review_report(state), encoding="utf-8")

    return state
