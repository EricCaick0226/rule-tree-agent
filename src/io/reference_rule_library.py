from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from .rule_table_linker import (
    RuleTableReference,
    build_rule_table_links_from_references,
    links_to_dicts,
    load_rule_table_rows,
)


REVIEW_ONLY_REASON = "当前输出未找到高相似匹配；该项仅来自参考库，需人工确认当前地方文档是否应补充。"
MATCH_REASON = "当前输出未找到高相似匹配。"
STRONG_MATCH_SCORE = 0.85
BROAD_REFERENCE_REUSE_THRESHOLD = 3
GENERIC_SINGLE_LEVEL_TERMS = {"预案", "方案", "资源管理服务"}
SUSPICIOUS_REFERENCE_TERMS = {"技资管理"}


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
        if not metadata_path.exists() and not rule_table_path.exists():
            continue
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
                reuse_policy=str(metadata.get("reuse_policy") or "assist").strip() or "assist",
                reference_trust_level=str(metadata.get("reference_trust_level") or "auxiliary").strip() or "auxiliary",
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


def _top_match(link: dict[str, Any]) -> dict[str, Any] | None:
    matches = link.get("matches") or []
    if not matches or not isinstance(matches[0], dict):
        return None
    return matches[0]


def _top_reference_key(link: dict[str, Any]) -> tuple[str, str]:
    match = _top_match(link)
    if not match:
        return "", ""
    return str(match.get("reference_file") or ""), str(match.get("reference_row_id") or "")


def _split_match_tiers(
    matched_current_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reuse_counts = Counter(_top_reference_key(link) for link in matched_current_rows)
    strong_matches: list[dict[str, Any]] = []
    broad_matches: list[dict[str, Any]] = []
    for link in matched_current_rows:
        match = _top_match(link)
        score = float(match.get("score") or 0.0) if match else 0.0
        reused = reuse_counts[_top_reference_key(link)] >= BROAD_REFERENCE_REUSE_THRESHOLD
        if score >= STRONG_MATCH_SCORE and not reused:
            strong_matches.append(link)
        else:
            broad_matches.append(link)
    return strong_matches, broad_matches


def _compact_path_text(path: object) -> str:
    return re.sub(r"\s+", "", _path_text(path))


def _is_low_quality_reference_suggestion(suggestion: dict[str, Any]) -> bool:
    levels = _string_list(suggestion.get("reference_path"))
    if not levels:
        return True
    compact = _compact_path_text(levels)
    if len(levels) == 1 and compact in GENERIC_SINGLE_LEVEL_TERMS:
        return True
    if any(term in compact for term in SUSPICIOUS_REFERENCE_TERMS):
        return True
    if re.search(r"\d+(?:\.\d+)?\s+\d+", _path_text(levels)):
        return True
    return False


def _split_missing_tiers(
    missing_reference_suggestions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing_candidates: list[dict[str, Any]] = []
    low_quality_rows: list[dict[str, Any]] = []
    for suggestion in missing_reference_suggestions:
        if _is_low_quality_reference_suggestion(suggestion):
            low_quality_rows.append(suggestion)
        else:
            missing_candidates.append(suggestion)
    return missing_candidates, low_quality_rows


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
    missing_reference_suggestions = _missing_reference_suggestions(references, matched_keys)
    strong_matches, broad_matches = _split_match_tiers(matched_current_rows)
    missing_candidates, low_quality_reference_rows = _split_missing_tiers(missing_reference_suggestions)
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
        "missing_reference_suggestions": missing_reference_suggestions,
        "strong_matches": strong_matches,
        "broad_matches": broad_matches,
        "missing_candidates": missing_candidates,
        "low_quality_reference_rows": low_quality_reference_rows,
    }


def _path_text(path: object) -> str:
    levels = _string_list(path)
    return " / ".join(levels) if levels else "(empty path)"


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def _append_match_section(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.append(title)
    if not rows:
        lines.append("")
        lines.append("(none)")
        lines.append("")
        return
    for index, link in enumerate(rows, start=1):
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


def _append_suggestion_section(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.append(title)
    if not rows:
        lines.append("")
        lines.append("(none)")
        lines.append("")
        return
    for index, suggestion in enumerate(rows, start=1):
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
    lines.append("")


def render_reference_suggestions_markdown(report: dict[str, Any]) -> str:
    matched_rows = [item for item in report.get("matched_current_rows") or [] if isinstance(item, dict)]
    missing_rows = [
        item for item in report.get("missing_reference_suggestions") or [] if isinstance(item, dict)
    ]
    strong_matches = [item for item in report.get("strong_matches") or [] if isinstance(item, dict)]
    broad_matches = [item for item in report.get("broad_matches") or [] if isinstance(item, dict)]
    if not strong_matches and not broad_matches and matched_rows:
        strong_matches, broad_matches = _split_match_tiers(matched_rows)
    missing_candidates = [item for item in report.get("missing_candidates") or [] if isinstance(item, dict)]
    low_quality_rows = [
        item for item in report.get("low_quality_reference_rows") or [] if isinstance(item, dict)
    ]
    if not missing_candidates and not low_quality_rows and missing_rows:
        missing_candidates, low_quality_rows = _split_missing_tiers(missing_rows)
    lines = [
        "# Reference Suggestions",
        "",
        f"- Current: `{report.get('current', '')}`",
        f"- References: {len(report.get('references') or [])}",
        f"- Matched current rows: {len(matched_rows)}",
        f"- Missing reference suggestions: {len(missing_rows)}",
        f"- Strong matches: {len(strong_matches)}",
        f"- Broad matches: {len(broad_matches)}",
        f"- Missing candidates: {len(missing_candidates)}",
        f"- Low quality reference rows: {len(low_quality_rows)}",
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

    _append_match_section(lines, "## Strong Matches", strong_matches)
    _append_match_section(lines, "## Broad Matches", broad_matches)
    _append_suggestion_section(lines, "## Missing Candidates", missing_candidates)
    _append_suggestion_section(lines, "## Low Quality Reference Rows", low_quality_rows)

    return "\n".join(lines).rstrip() + "\n"
