from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.agent_state import AgentState, ClassificationRow  # noqa: E402
from src.steps.reference_row_prefill import apply_reference_row_reuse  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"rule table must be a JSON object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _make_classification_row(data: dict[str, Any]) -> ClassificationRow:
    allowed = {field.name for field in fields(ClassificationRow)}
    payload = {key: value for key, value in data.items() if key in allowed}
    payload["evidence_refs"] = []
    return ClassificationRow(**payload)


def _load_classification_rows(rule_table_path: Path) -> list[ClassificationRow]:
    data = _load_json(rule_table_path)
    raw_rows = data.get("classification_rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"rule table missing classification_rows array: {rule_table_path}")
    rows: list[ClassificationRow] = []
    for item in raw_rows:
        if isinstance(item, dict):
            rows.append(_make_classification_row(deepcopy(item)))
    return rows


def _path_text(path_levels: list[str]) -> str:
    return " / ".join(path_levels)


def _first_direct_match(row: ClassificationRow) -> dict[str, Any]:
    for match in row.reference_matches:
        if isinstance(match, dict) and match.get("usage") == "direct_reuse":
            return match
    return {}


def _description_preview(description: str, limit: int = 120) -> str:
    text = " ".join(str(description or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _direct_reuse_row_payload(row: ClassificationRow) -> dict[str, Any]:
    match = _first_direct_match(row)
    old_path = row.original_path_levels or row.path_levels
    return {
        "row_id": row.row_id,
        "old_path": list(old_path),
        "new_path": list(row.path_levels),
        "reference_path": _string_list(match.get("reference_path")),
        "reference_row_id": str(match.get("reference_row_id") or ""),
        "reference_file": str(match.get("reference_file") or ""),
        "match_type": str(match.get("match_type") or ""),
        "score": match.get("score"),
        "reused_fields": list(row.reference_prefilled_fields),
        "description_source": row.description_source,
        "description_preview": _description_preview(row.description),
    }


def _review_candidate_payload(row: ClassificationRow) -> dict[str, Any]:
    match = row.reference_matches[0] if row.reference_matches else {}
    return {
        "row_id": row.row_id,
        "path": list(row.path_levels),
        "reference_row_id": str(match.get("reference_row_id") or ""),
        "reference_file": str(match.get("reference_file") or ""),
        "description_source": row.description_source,
        "description_preview": _description_preview(row.description),
        "review_reason": row.review_reason,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reference Reuse Offline Report",
        "",
        f"- rule_table: `{report['rule_table']}`",
        f"- reference_library: `{report['reference_library']}`",
        f"- original_rows: {report['original_rows']}",
        f"- final_rows: {report['final_rows']}",
        f"- direct_reused_rows: {report['direct_reused_rows']}",
        f"- review_candidates_added: {report['review_candidates_added']}",
        "",
        "## Match Types",
        "",
    ]
    if report["match_types"]:
        for match_type, count in sorted(report["match_types"].items()):
            lines.append(f"- {match_type}: {count}")
    else:
        lines.append("(none)")

    lines.extend(["", "## Direct Reuse Rows", ""])
    if not report["direct_reuse_rows"]:
        lines.append("(none)")
    for index, row in enumerate(report["direct_reuse_rows"], start=1):
        lines.extend(
            [
                f"### {index}. {_path_text(row['old_path'])}",
                f"- row_id: `{row['row_id']}`",
                f"- new_path: {_path_text(row['new_path'])}",
                f"- reference_path: {_path_text(row['reference_path'])}",
                f"- match_type: `{row['match_type']}`",
                f"- reused_fields: {', '.join(row['reused_fields'])}",
                f"- description_source: `{row['description_source']}`",
                f"- description_preview: {row['description_preview']}",
                "",
            ]
        )

    lines.extend(["## Review Candidates Added", ""])
    if not report["review_candidates"]:
        lines.append("(none)")
    for index, row in enumerate(report["review_candidates"], start=1):
        lines.extend(
            [
                f"### {index}. {_path_text(row['path'])}",
                f"- row_id: `{row['row_id']}`",
                f"- reference_row_id: `{row['reference_row_id']}`",
                f"- description_source: `{row['description_source']}`",
                f"- description_preview: {row['description_preview']}",
                f"- review_reason: {row['review_reason']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def generate_reference_reuse_report(
    rule_table_path: Path,
    reference_library_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    original_rows = _load_classification_rows(rule_table_path)
    state = AgentState(
        task="offline_reference_reuse_report",
        classification_rows=deepcopy(original_rows),
    )
    state = apply_reference_row_reuse(state, library_dir=str(reference_library_dir))

    original_count = len(original_rows)
    direct_reuse_rows = [
        _direct_reuse_row_payload(row)
        for row in state.classification_rows[:original_count]
        if row.reference_prefilled_fields and _first_direct_match(row)
    ]
    review_candidates = [
        _review_candidate_payload(row)
        for row in state.reference_candidate_rows
        if row.inclusion_status == "review_candidate"
    ]
    match_types = Counter(
        row["match_type"]
        for row in direct_reuse_rows
        if row.get("match_type")
    )
    reused_fields = Counter(
        field
        for row in direct_reuse_rows
        for field in row.get("reused_fields", [])
    )
    report: dict[str, Any] = {
        "rule_table": str(rule_table_path),
        "reference_library": str(reference_library_dir),
        "original_rows": original_count,
        "final_rows": len(state.classification_rows),
        "direct_reused_rows": len(direct_reuse_rows),
        "review_candidates_added": len(review_candidates),
        "match_types": dict(sorted(match_types.items())),
        "reused_fields": dict(sorted(reused_fields.items())),
        "direct_reuse_rows": direct_reuse_rows,
        "review_candidates": review_candidates,
        "step_traces": [
            {
                "step_name": trace.step_name,
                "status": trace.status,
                "input_summary": trace.input_summary,
                "output_summary": trace.output_summary,
            }
            for trace in state.step_traces
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "direct_reuse_report.json", report)
    (output_dir / "direct_reuse_report.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reference row reuse offline and write detailed review reports."
    )
    parser.add_argument("--rule-table", required=True, help="Existing rule_table.json path.")
    parser.add_argument(
        "--reference-library",
        default="reference_library",
        help="Reference library root directory. Default: reference_library.",
    )
    parser.add_argument("--out", required=True, help="Output directory for reports.")
    args = parser.parse_args()

    report = generate_reference_reuse_report(
        Path(args.rule_table),
        Path(args.reference_library),
        Path(args.out),
    )
    print(f"Wrote {Path(args.out) / 'direct_reuse_report.json'}")
    print(f"Wrote {Path(args.out) / 'direct_reuse_report.md'}")
    print(
        "direct_reused_rows={direct_reused_rows} review_candidates_added={review_candidates_added}".format(
            **report
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
