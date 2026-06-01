from __future__ import annotations

import json
from typing import Any


def render_json_report(report: dict[str, Any]) -> str:
    return (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def render_markdown_report(report: dict[str, Any]) -> str:
    recommendation = _dict(report.get("recommendation"))
    inputs = _dict(report.get("inputs"))
    row_extraction = _dict(report.get("row_extraction"))
    quality = _dict(report.get("quality"))
    structure = _dict(report.get("structure"))
    risk_signals = _list(report.get("risk_signals"))

    lines = [
        "# Eval Report",
        "",
        "## Verdict",
        f"- merge_ready: {_bool_text(recommendation.get('merge_ready', False))}",
        f"- note: {_text(recommendation.get('note'), 'n/a')}",
        "",
        "## Inputs",
    ]
    for key in (
        "rule_table_json",
        "rule_tree_json",
        "review_report_md",
        "row_checkpoint",
        "evidence_checkpoint",
        "block_checkpoint",
    ):
        if key in inputs:
            lines.append(f"- {key}: {_bool_text(inputs.get(key))}")
    lines.extend(
        [
            f"- debug_file_count: {_int_text(inputs.get('debug_file_count'))}",
            f"- trace_file_count: {_int_text(inputs.get('trace_file_count'))}",
            "",
            "## Runtime Summary",
            f"- batch_count: {_int_text(row_extraction.get('batch_count'))}",
            f"- completed_batch_indices: {_join_limited(row_extraction.get('completed_batch_indices'))}",
            f"- appears_complete: {_bool_text(row_extraction.get('appears_complete'))}",
            f"- total_checkpoint_rows: {_int_text(row_extraction.get('total_checkpoint_rows'))}",
            f"- total_elapsed_seconds: {_number_text(row_extraction.get('total_elapsed_seconds'))}",
            "",
            "## Quality Summary",
            f"- classification_row_count: {_int_text(quality.get('classification_row_count'))}",
            f"- unique_path_count: {_int_text(quality.get('unique_path_count'))}",
            f"- duplicate_path_count: {_int_text(quality.get('duplicate_path_count'))}",
            f"- needs_review_count: {_int_text(quality.get('needs_review_count'))}",
            f"- missing_evidence_quote_count: {_int_text(quality.get('missing_evidence_quote_count'))}",
            f"- missing_evidence_refs_count: {_int_text(quality.get('missing_evidence_refs_count'))}",
            "",
            "## Slowest Batches",
        ]
    )
    slowest_batches = _list(row_extraction.get("slowest_batches"))
    if slowest_batches:
        for batch in slowest_batches[:3]:
            batch_data = _dict(batch)
            lines.append(
                "- batch "
                f"{_text(batch_data.get('batch_index'), '?')}: "
                f"{_number_text(batch_data.get('elapsed_seconds'))}s"
            )
    else:
        lines.append("- none")

    severity_counts = _dict(quality.get("validation_issue_count_by_severity"))
    high_targets = _list(quality.get("high_severity_targets"))
    lines.extend(
        [
            "",
            "## Structure Risks",
            f"- header_as_path_count: {_int_text(structure.get('header_as_path_count'))}",
            f"- generic_column_path_count: {_int_text(structure.get('generic_column_path_count'))}",
            f"- stale_section_title_count: {_int_text(structure.get('stale_section_title_count'))}",
            f"- appendix_table_detected_count: {_int_text(structure.get('appendix_table_detected_count'))}",
            f"- continued_table_count: {_int_text(structure.get('continued_table_count'))}",
            "",
            "## Validation Issues",
            f"- validation_issue_count: {_int_text(quality.get('validation_issue_count'))}",
            f"- by_severity: {_format_mapping(severity_counts)}",
            f"- high_severity_targets: {_join_limited(high_targets)}",
            "",
            "## Risk Signals",
        ]
    )
    if risk_signals:
        for signal in risk_signals[:5]:
            lines.append(f"- {_format_risk_signal(signal)}")
        if len(risk_signals) > 5:
            lines.append(f"- ... {len(risk_signals) - 5} more")
    else:
        lines.append("- none")

    reasons = _list(recommendation.get("reasons"))
    lines.extend(
        [
            "",
            "## Recommended Next Action",
        ]
    )
    if reasons:
        for reason in reasons[:5]:
            lines.append(f"- {_text(reason)}")
        if len(reasons) > 5:
            lines.append(f"- ... {len(reasons) - 5} more")
    else:
        lines.append("- no blocking diagnostic findings")

    return "\n".join(lines) + "\n"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _int_text(value: Any) -> str:
    if isinstance(value, bool):
        return "0"
    if isinstance(value, int):
        return str(value)
    return "0"


def _number_text(value: Any) -> str:
    if isinstance(value, bool):
        return "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "0"


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = _one_line(str(value))
    return text if text else default


def _bounded_text(value: Any, default: str = "", limit: int = 96) -> str:
    text = _text(value, default)
    if len(text) <= limit:
        return text
    return text[: limit - 4].rstrip() + " ..."


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _join_limited(value: Any, limit: int = 8) -> str:
    items = _list(value)
    if not items:
        return "none"
    rendered = ", ".join(_text(item, "?") for item in items[:limit])
    if len(items) > limit:
        rendered += f", ... {len(items) - limit} more"
    return rendered


def _format_mapping(value: dict[str, Any]) -> str:
    if not value:
        return "none"
    return ", ".join(f"{key}={_text(item)}" for key, item in sorted(value.items()))


def _format_risk_signal(signal: Any) -> str:
    if not isinstance(signal, dict):
        return _bounded_text(signal, "unknown risk")
    signal_type = _bounded_text(signal.get("type"), "unknown", limit=32)
    severity = _bounded_text(signal.get("severity"), "review", limit=24)
    message = _bounded_text(signal.get("message"), "", limit=72)
    detail_parts = []
    for key in ("path", "count"):
        if key in signal:
            detail_parts.append(f"{key}={_bounded_text(signal.get(key), limit=36)}")
    if "path_levels" in signal:
        detail_parts.append(
            f"path_levels={_bounded_text(_join_limited(signal.get('path_levels')), limit=36)}"
        )
    details = f" ({', '.join(detail_parts)})" if detail_parts else ""
    return f"{signal_type} [{severity}]: {message}{details}"
