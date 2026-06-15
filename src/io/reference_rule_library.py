from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .rule_table_linker import (
    RuleTableReference,
    build_rule_table_links_from_references,
    links_to_dicts,
    load_rule_table_rows,
)


REVIEW_ONLY_REASON = "当前输出未找到高相似匹配；该项仅来自参考库，需人工确认当前地方文档是否应补充。"
MATCH_REASON = "当前输出未找到高相似匹配。"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata.json is malformed: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"metadata.json must contain a JSON object: {path}")
    name = str(data.get("name") or "").strip()
    source_type = str(data.get("source_type") or "").strip()
    if not name or not source_type:
        raise ValueError(f"metadata.json must contain non-empty name and source_type: {path}")
    return data


def load_reference_library(library_dir: Path) -> tuple[list[RuleTableReference], list[str]]:
    root = Path(library_dir)
    if not root.exists():
        raise FileNotFoundError(f"reference library does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"reference library must be a directory: {root}")

    references: list[RuleTableReference] = []
    warnings: list[str] = []
    for entry in sorted(item for item in root.iterdir() if item.is_dir()):
        metadata_path = entry / "metadata.json"
        rule_table_path = entry / "rule_table.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"reference entry missing metadata.json: {entry}")
        if not rule_table_path.exists():
            raise FileNotFoundError(f"reference entry missing rule_table.json: {entry}")

        metadata = _load_metadata(metadata_path)
        rows = load_rule_table_rows(rule_table_path)
        if not rows:
            warnings.append(f"reference has no usable rows and was skipped: {entry}")
            continue
        references.append(
            RuleTableReference(
                name=str(metadata["name"]).strip(),
                source_type=str(metadata["source_type"]).strip(),
                path=str(rule_table_path),
                rows=rows,
            )
        )
    return references, warnings


def _matched_reference_keys(matched_current_rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for link in matched_current_rows:
        for match in link.get("matches") or []:
            if not isinstance(match, dict):
                continue
            reference_file = str(match.get("reference_file") or "")
            reference_row_id = str(match.get("reference_row_id") or "")
            if reference_file or reference_row_id:
                keys.add((reference_file, reference_row_id))
    return keys


def _reference_row_path(row: dict[str, Any]) -> list[str]:
    return _string_list(row.get("path_levels") or row.get("path"))


def _missing_reference_suggestions(
    references: list[RuleTableReference],
    matched_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for reference in references:
        for row in reference.rows:
            row_id = str(row.get("row_id") or "")
            key = (reference.path, row_id)
            if key in matched_keys:
                continue
            suggestions.append(
                {
                    "reference_name": reference.name,
                    "reference_type": reference.source_type,
                    "reference_file": reference.path,
                    "reference_row_id": row_id,
                    "reference_path": _reference_row_path(row),
                    "reference_description": str(row.get("description") or ""),
                    "reference_grade": row.get("recommended_grade"),
                    "suggestion_type": "missing_reference_candidate",
                    "source": "reference_library",
                    "match_reason": MATCH_REASON,
                    "needs_review": True,
                    "review_reason": REVIEW_ONLY_REASON,
                }
            )
    suggestions.sort(
        key=lambda item: (
            str(item.get("reference_name") or ""),
            "/".join(_string_list(item.get("reference_path"))),
            str(item.get("reference_row_id") or ""),
        )
    )
    return suggestions


def build_reference_suggestion_report(
    current_path: str,
    current_rows: list[dict[str, Any]],
    references: list[RuleTableReference],
    warnings: list[str] | None = None,
    top_k: int = 3,
    min_score: float = 0.5,
) -> dict[str, Any]:
    links = build_rule_table_links_from_references(
        current_rows=current_rows,
        references=references,
        top_k=top_k,
        min_score=min_score,
    )
    matched_current_rows = links_to_dicts(links)
    matched_keys = _matched_reference_keys(matched_current_rows)
    return {
        "current": current_path,
        "references": [
            {
                "name": reference.name,
                "type": reference.source_type,
                "path": reference.path,
                "rows": len(reference.rows),
            }
            for reference in references
        ],
        "warnings": list(warnings or []),
        "matched_current_rows": matched_current_rows,
        "missing_reference_suggestions": _missing_reference_suggestions(references, matched_keys),
    }


def _path_text(path: object) -> str:
    levels = _string_list(path)
    return " / ".join(levels) if levels else "(empty path)"


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def render_reference_suggestions_markdown(report: dict[str, Any]) -> str:
    matched_rows = [item for item in report.get("matched_current_rows") or [] if isinstance(item, dict)]
    missing_rows = [
        item for item in report.get("missing_reference_suggestions") or [] if isinstance(item, dict)
    ]
    lines = [
        "# Reference Suggestions",
        "",
        f"- Current: `{report.get('current', '')}`",
        f"- References: {len(report.get('references') or [])}",
        f"- Matched current rows: {len(matched_rows)}",
        f"- Missing reference suggestions: {len(missing_rows)}",
        "- Reference rows are review hints only; they are not current-document evidence.",
        "- Missing suggestions are not merged into rule_table.json or rule_tree.json.",
        "",
    ]

    warnings = [str(item) for item in report.get("warnings") or [] if str(item).strip()]
    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Matched Current Rows")
    if not matched_rows:
        lines.append("")
        lines.append("(none)")
        lines.append("")
    for index, link in enumerate(matched_rows, start=1):
        lines.append("")
        lines.append(f"### {index}. {_path_text(link.get('current_path'))}")
        lines.append(f"- current_row_id: `{link.get('current_row_id', '')}`")
        lines.append(f"- current_description_source: `{link.get('current_description_source', '')}`")
        for match in link.get("matches") or []:
            shared = ", ".join(_string_list(match.get("shared_terms")))
            lines.append(
                "- match: "
                f"{_path_text(match.get('reference_path'))} "
                f"(score={match.get('score')}, reference={match.get('reference_name')}, "
                f"type={match.get('reference_type')}, row_id={match.get('reference_row_id')}, "
                f"shared: {shared})"
            )

    lines.append("")
    lines.append("## Missing Reference Suggestions")
    if not missing_rows:
        lines.append("")
        lines.append("(none)")
        lines.append("")
    for index, suggestion in enumerate(missing_rows, start=1):
        lines.append("")
        lines.append(f"### {index}. {_path_text(suggestion.get('reference_path'))}")
        lines.append(f"- reference: `{suggestion.get('reference_name', '')}`")
        lines.append(f"- reference_type: `{suggestion.get('reference_type', '')}`")
        lines.append(f"- reference_row_id: `{suggestion.get('reference_row_id', '')}`")
        lines.append(f"- reference_grade: `{suggestion.get('reference_grade', '')}`")
        lines.append(f"- needs_review: `{_bool_text(suggestion.get('needs_review'))}`")
        lines.append(f"- review_reason: {suggestion.get('review_reason', '')}")
        description = str(suggestion.get("reference_description") or "").strip()
        if description:
            lines.append(f"- reference_description: {description}")

    return "\n".join(lines).rstrip() + "\n"
