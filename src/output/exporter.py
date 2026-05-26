from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from ..core.agent_state import AgentState, TreeNode


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
                if rule.negative_conditions:
                    lines.append(f"{indent}  - negative: {', '.join(rule.negative_conditions)}")
                if not rule.conditions and not rule.negative_conditions:
                    lines.append(f"{indent}  - rule: insufficient evidence")
        for child in sorted(children_by_parent.get(node.node_id, []), key=lambda item: item.path):
            render(child)

    for root in sorted(children_by_parent.get(None, []), key=lambda item: item.path):
        render(root)
    return lines


def _review_report(state: AgentState) -> str:
    nodes_with_evidence = sum(1 for node in state.nodes if node.evidence_refs)
    review_nodes = sum(1 for node in state.nodes if node.needs_review)
    source_counts: dict[str, int] = {}
    for chunk in state.chunks:
        source_counts[chunk.source_method] = source_counts.get(chunk.source_method, 0) + 1
    ocr_claims = [
        claim
        for claim in state.evidence_claims
        if any(ref.source_method == "ocr" for ref in claim.evidence_refs)
    ]
    claim_support_counts: dict[str, int] = {}
    for claim in state.evidence_claims:
        claim_support_counts[claim.support_level] = claim_support_counts.get(claim.support_level, 0) + 1
    review_claims = [
        claim
        for claim in state.evidence_claims
        if claim.needs_review and claim.review_reason
    ]
    unsupported = [
        issue
        for issue in state.validation_issues
        if issue.issue_type in {"unsupported_generation", "hardcoded_or_ungrounded_content"}
    ]
    low_confidence = [node for node in state.nodes if node.confidence < 0.6]
    missing_grading = [
        issue for issue in state.validation_issues if issue.issue_type == "missing_grade_scheme"
    ]
    selected_dimension = (
        state.selected_dimension.name
        if state.selected_dimension
        else "cannot determine from current documents"
    )

    lines = [
        "# Human Review Report",
        "",
        "## Summary",
        f"- LLM enabled: {'yes' if state.llm_enabled else 'no'}",
        f"- LLM used: {'yes' if state.llm_used else 'no'}",
        f"- LLM model: {state.llm_model if state.llm_enabled else 'not used'}",
        f"- LLM base URL: {state.llm_base_url if state.llm_enabled else 'not used'}",
        f"- LLM error: {state.llm_error if state.llm_error else 'none'}",
        f"- PDF OCR enabled: {'yes' if state.pdf_ocr_enabled else 'no'}",
        f"- Source chunks by method: {source_counts if source_counts else 'none'}",
        f"- Evidence claims: {len(state.evidence_claims)}",
        f"- Evidence claims by support level: {claim_support_counts if claim_support_counts else 'none'}",
        f"- Concept profiles: {len(state.concept_profiles)}",
        f"- Classification dimension found: {'yes' if state.selected_dimension else 'no'}",
        f"- Selected dimension: {selected_dimension}",
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

    lines.extend(["", "## OCR Evidence Notes"])
    if ocr_claims:
        lines.append(
            "- OCR-derived evidence is not final evidence until manually checked against the PDF pages."
        )
        for claim in ocr_claims[:30]:
            pages = sorted(
                {
                    ref.page_number
                    for ref in claim.evidence_refs
                    if ref.source_method == "ocr" and ref.page_number is not None
                }
            )
            page_note = f"pages {pages}" if pages else "pages unknown"
            lines.append(f"- {claim.claim_id}: {claim.subject} ({page_note})")
        if len(ocr_claims) > 30:
            lines.append(f"- ... {len(ocr_claims) - 30} more OCR-backed claims omitted from this summary.")
    elif state.pdf_ocr_enabled:
        lines.append("- OCR was enabled, but no OCR-backed evidence claims were produced.")
    else:
        lines.append("- OCR was not enabled.")

    lines.extend(["", "## Evidence Claim Review Reasons"])
    if review_claims:
        for claim in review_claims[:30]:
            lines.append(f"- {claim.claim_id}: {claim.review_reason}")
        if len(review_claims) > 30:
            lines.append(f"- ... {len(review_claims) - 30} more review reasons omitted from this summary.")
    else:
        lines.append("- None recorded.")

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

    lines.extend(["", "## Step Trace"])
    if state.step_traces:
        for trace in state.step_traces:
            trace_path = f"; raw={trace.raw_response_path}" if trace.raw_response_path else ""
            lines.append(
                f"- {trace.step_name}: {trace.status}; "
                f"input={trace.input_summary}; output={trace.output_summary}{trace_path}"
            )
    else:
        lines.append("- No step traces recorded.")

    return "\n".join(lines) + "\n"


def _safe_trace_name(index: int, step_name: str) -> str:
    safe_step = re.sub(r"[^a-zA-Z0-9_.-]+", "_", step_name).strip("_") or "step"
    return f"{index:02d}_{safe_step}.txt"


def _export_raw_traces(state: AgentState, out_dir: Path) -> Path | None:
    traces_with_raw = [trace for trace in state.step_traces if trace.raw_response]
    if not traces_with_raw:
        return None

    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    for index, trace in enumerate(state.step_traces, start=1):
        if not trace.raw_response:
            continue
        trace_path = trace_dir / _safe_trace_name(index, trace.step_name)
        trace_path.write_text(trace.raw_response, encoding="utf-8")
        trace.raw_response_path = str(trace_path)
        trace.raw_response = ""

    return trace_dir


def export_outputs(state: AgentState, output_dir: str) -> AgentState:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = _export_raw_traces(state, out_dir)

    json_path = out_dir / "rule_tree.json"
    md_path = out_dir / "rule_tree.md"
    report_path = out_dir / "review_report.md"
    state.output_paths = {
        "rule_tree_json": str(json_path),
        "rule_tree_md": str(md_path),
        "review_report_md": str(report_path),
    }
    if trace_dir:
        state.output_paths["trace_dir"] = str(trace_dir)

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
    tree_lines.append(f"- LLM enabled: {'yes' if state.llm_enabled else 'no'}")
    tree_lines.append(f"- LLM used: {'yes' if state.llm_used else 'no'}")
    if state.llm_enabled:
        tree_lines.append(f"- LLM model: {state.llm_model}")
        tree_lines.append(f"- LLM base URL: {state.llm_base_url}")
    tree_lines.append(f"- PDF OCR enabled: {'yes' if state.pdf_ocr_enabled else 'no'}")
    if state.llm_error:
        tree_lines.append(f"- LLM error: {state.llm_error}")
    tree_lines.append(f"- Evidence claims: {len(state.evidence_claims)}")
    tree_lines.append(f"- Concept profiles: {len(state.concept_profiles)}")
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
