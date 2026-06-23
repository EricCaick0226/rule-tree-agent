from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from ..core.agent_state import AgentState, TreeNode


LEVEL_NAMES = "一二三四五六七八九十"
HIERARCHICAL_CODE_RE = re.compile(r"(?<!\d)\d+(?:\s*[.．]\s*\d+)+(?!\d)")
NUMERIC_LEVEL_RE = re.compile(r"^\d+(?:\s*[.．]\s*\d+)*$")


def _level_column(index: int) -> str:
    if 1 <= index <= len(LEVEL_NAMES):
        return f"{LEVEL_NAMES[index - 1]}级分类"
    return f"第{index}级分类"


def _safe_md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _rule_table_json(state: AgentState) -> dict:
    return {
        "classification_schema": (
            asdict(state.classification_schema) if state.classification_schema else None
        ),
        "grade_scheme": [asdict(grade) for grade in state.grade_scheme],
        "classification_rows": [asdict(row) for row in state.classification_rows],
        "validation_issues": [asdict(issue) for issue in state.validation_issues],
    }


def _reference_candidates_json(state: AgentState) -> dict:
    return {
        "reference_candidate_rows": [
            asdict(row) for row in state.reference_candidate_rows
        ],
    }


def _reference_candidates_md(state: AgentState) -> str:
    lines = [
        "# Reference Candidate Rows",
        "",
        "> These rows come from the reference library and were not directly matched to the current document. Review before adding them to the main rule table.",
        "",
        f"- Candidate rows: {len(state.reference_candidate_rows)}",
        "",
    ]
    if not state.reference_candidate_rows:
        lines.append("(none)")
        return "\n".join(lines) + "\n"

    headers = [
        "分类路径",
        "分类说明",
        "参考行",
        "需复核",
        "复核原因",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in state.reference_candidate_rows:
        match = row.reference_matches[0] if row.reference_matches else {}
        cells = [
            " / ".join(row.path_levels),
            row.description,
            str(match.get("reference_row_id") or ""),
            "yes" if row.needs_review else "no",
            row.review_reason,
        ]
        lines.append("| " + " | ".join(_safe_md_cell(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def _rule_table_md(state: AgentState) -> str:
    schema_depth = state.classification_schema.max_depth if state.classification_schema else 0
    row_depth = max((len(row.path_levels) for row in state.classification_rows), default=0)
    max_depth = max(schema_depth, row_depth)
    headers = [_level_column(index) for index in range(1, max_depth + 1)]
    headers.extend(
        [
            "推荐分级",
            "分类说明",
            "数据范围及示例",
            "数据加工程度",
            "影响对象",
            "影响程度",
            "证据强度",
            "行来源",
            "内容来源",
            "纳入状态",
            "证据状态",
            "原始路径",
            "预填字段",
            "需复核",
            "复核原因",
        ]
    )

    lines = [
        "# Candidate Classification Table",
        "",
        "> This is a candidate output. It requires human review before use.",
        "",
    ]
    if not state.classification_rows:
        lines.append("证据不足，无法从当前文档确定分类分级明细。")
        return "\n".join(lines) + "\n"

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in state.classification_rows:
        cells: list[object] = []
        for index in range(max_depth):
            cells.append(row.path_levels[index] if index < len(row.path_levels) else "")
        cells.extend(
            [
                row.recommended_grade or "",
                row.description,
                "；".join(row.data_range_examples),
                row.processing_degree,
                row.impact_object,
                row.impact_degree,
                row.support_level,
                row.row_source,
                row.content_source,
                row.inclusion_status,
                row.evidence_status,
                " / ".join(row.original_path_levels),
                "；".join(row.reference_prefilled_fields),
                "yes" if row.needs_review else "no",
                row.review_reason,
            ]
        )
        lines.append("| " + " | ".join(_safe_md_cell(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


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
        for child in sorted(children_by_parent.get(node.node_id, []), key=lambda item: item.path):
            render(child)

    for root in sorted(children_by_parent.get(None, []), key=lambda item: item.path):
        render(root)
    return lines


def _normalized_level(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).replace("．", ".")


def _structure_quality_metrics(state: AgentState) -> dict[str, int]:
    rows_with_numeric_only = 0
    numeric_only_levels = 0
    multi_code_levels = 0

    for row in state.classification_rows:
        row_has_numeric_only = False
        for level in row.path_levels:
            text = str(level or "")
            if len(HIERARCHICAL_CODE_RE.findall(text)) >= 2:
                multi_code_levels += 1
            if NUMERIC_LEVEL_RE.fullmatch(_normalized_level(text)):
                numeric_only_levels += 1
                row_has_numeric_only = True
        if row_has_numeric_only:
            rows_with_numeric_only += 1

    return {
        "classification_rows": len(state.classification_rows),
        "multi_code_path_levels": multi_code_levels,
        "rows_with_numeric_only_path_levels": rows_with_numeric_only,
        "numeric_only_path_levels": numeric_only_levels,
    }


def _structure_quality_lines(state: AgentState) -> list[str]:
    metrics = _structure_quality_metrics(state)
    lines = [
        "",
        "## Structure Quality Notes",
        f"- Classification rows: {metrics['classification_rows']}",
        (
            "- Path levels containing multiple hierarchical codes: "
            f"{metrics['multi_code_path_levels']}"
        ),
        (
            "- Rows with numeric-only path levels: "
            f"{metrics['rows_with_numeric_only_path_levels']}"
        ),
        f"- Numeric-only path levels: {metrics['numeric_only_path_levels']}",
    ]
    if metrics["multi_code_path_levels"]:
        lines.append("- Review rows where one path level contains multiple hierarchical codes.")
    if metrics["numeric_only_path_levels"]:
        lines.append("- Review rows where codes were separated from labels into standalone path levels.")
    if not metrics["multi_code_path_levels"] and not metrics["numeric_only_path_levels"]:
        lines.append("- No obvious path-structure fragmentation detected.")
    return lines


def _review_report(state: AgentState) -> str:
    nodes_with_evidence = sum(1 for node in state.nodes if node.evidence_refs)
    review_nodes = sum(1 for node in state.nodes if node.needs_review)
    reference_prefilled_rows = sum(
        1 for row in state.classification_rows if row.reference_prefilled_fields
    )
    reference_candidate_rows = len(state.reference_candidate_rows)
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
        f"- Grade scheme found: {'yes' if state.grade_scheme else 'no'}",
        f"- Number of nodes: {len(state.nodes)}",
        f"- Nodes with evidence: {nodes_with_evidence}",
        f"- Nodes requiring review: {review_nodes}",
        f"- Reference-prefilled rows: {reference_prefilled_rows}",
        f"- Reference candidate rows: {reference_candidate_rows}",
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

    lines.extend(_structure_quality_lines(state))

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
    table_json_path = out_dir / "rule_table.json"
    table_md_path = out_dir / "rule_table.md"
    reference_candidates_json_path = out_dir / "reference_candidates.json"
    reference_candidates_md_path = out_dir / "reference_candidates.md"
    report_path = out_dir / "review_report.md"
    state.output_paths = {
        "rule_tree_json": str(json_path),
        "rule_tree_md": str(md_path),
        "rule_table_json": str(table_json_path),
        "rule_table_md": str(table_md_path),
        "reference_candidates_json": str(reference_candidates_json_path),
        "reference_candidates_md": str(reference_candidates_md_path),
        "review_report_md": str(report_path),
    }
    if trace_dir:
        state.output_paths["trace_dir"] = str(trace_dir)

    json_path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    table_json_path.write_text(
        json.dumps(_rule_table_json(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    table_md_path.write_text(_rule_table_md(state), encoding="utf-8")
    reference_candidates_json_path.write_text(
        json.dumps(_reference_candidates_json(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reference_candidates_md_path.write_text(
        _reference_candidates_md(state),
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
