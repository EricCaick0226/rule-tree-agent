from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.eval_harness.loader import EvalInputs, LoadedJsonl


HEADER_PATH_LEVELS = {"类", "项", "目"}
GENERIC_COLUMN_PATH_LEVELS = {"类", "项", "目", "编码", "标识符", "数据元", "项目", "占位"}


def build_eval_report(inputs: EvalInputs) -> dict[str, Any]:
    rule_table = inputs.files["rule_table_json"]
    rows = _classification_rows(rule_table.data if rule_table.exists else None)
    validation_issues = _validation_issues(rule_table.data if rule_table.exists else None)
    quality = _quality_metrics(rows, validation_issues)
    structure = _structure_metrics(rows)
    local_metrics = _local_metrics(rows)
    row_extraction = _row_checkpoint_metrics(inputs.row_checkpoint)
    run_completeness = _run_completeness(inputs, row_extraction)
    risk_signals = _risk_signals(rows, inputs, quality, structure)
    recommendation = _recommendation(inputs, quality, run_completeness, risk_signals)

    return {
        "output_dir": str(inputs.output_dir),
        "inputs": {
            "rule_table_json": inputs.files["rule_table_json"].exists
            and not inputs.files["rule_table_json"].error,
            "rule_tree_json": inputs.files["rule_tree_json"].exists
            and not inputs.files["rule_tree_json"].error,
            "review_report_md": inputs.files["review_report_md"].exists
            and not inputs.files["review_report_md"].error,
            "row_checkpoint": inputs.row_checkpoint.exists,
            "evidence_checkpoint": inputs.evidence_checkpoint.exists,
            "block_checkpoint": inputs.block_checkpoint.exists,
            "debug_file_count": len(inputs.debug_files),
            "trace_file_count": len(inputs.trace_files),
        },
        "run_completeness": run_completeness,
        "row_extraction": row_extraction,
        "quality": quality,
        "structure": structure,
        "local_metrics": local_metrics,
        "risk_signals": risk_signals,
        "recommendation": recommendation,
    }


def _classification_rows(rule_table: Any) -> list[dict[str, Any]]:
    if not isinstance(rule_table, dict):
        return []
    rows = rule_table.get("classification_rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _validation_issues(rule_table: Any) -> list[dict[str, Any]]:
    if not isinstance(rule_table, dict):
        return []
    issues = rule_table.get("validation_issues")
    if not isinstance(issues, list):
        return []
    return [issue for issue in issues if isinstance(issue, dict)]


def _quality_metrics(
    rows: list[dict[str, Any]],
    validation_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = [_path_key(row) for row in rows]
    path_counts = Counter(path for path in paths if path)
    severity_counts = Counter(
        str(issue.get("severity", "unknown")).lower() or "unknown"
        for issue in validation_issues
    )

    return {
        "classification_row_count": len(rows),
        "unique_path_count": len(path_counts),
        "duplicate_path_count": sum(count - 1 for count in path_counts.values()),
        "needs_review_count": sum(1 for row in rows if _needs_review(row)),
        "missing_evidence_quote_count": sum(
            1 for row in rows if not _present(row.get("evidence_quote"))
        ),
        "missing_evidence_refs_count": sum(
            1 for row in rows if not _present(row.get("evidence_refs"))
        ),
        "validation_issue_count": len(validation_issues),
        "validation_issue_count_by_severity": dict(sorted(severity_counts.items())),
        "high_severity_targets": [
            str(issue.get("target", ""))
            for issue in validation_issues
            if str(issue.get("severity", "")).lower() == "high"
        ],
    }


def _row_checkpoint_metrics(checkpoint: LoadedJsonl) -> dict[str, Any]:
    completed_indices: list[int] = []
    rows_per_batch: dict[str, int] = {}
    elapsed_seconds_per_batch: dict[str, float] = {}
    batches_with_debug_paths: list[int] = []
    batch_count = 0

    for record in checkpoint.records:
        batch_index = _int_or_none(record.get("batch_index"))
        if batch_index is None:
            continue

        completed_indices.append(batch_index)
        batch_count = max(batch_count, _int_or_none(record.get("batch_count")) or 0)
        row_count = _checkpoint_row_count(record)
        rows_per_batch[str(batch_index)] = row_count

        elapsed_seconds = _float_or_zero(record.get("elapsed_seconds"))
        elapsed_seconds_per_batch[str(batch_index)] = elapsed_seconds

        if _present(record.get("split_retry_debug_paths")):
            batches_with_debug_paths.append(batch_index)

    completed_batch_indices = sorted(set(completed_indices))
    total_elapsed_seconds = round(sum(elapsed_seconds_per_batch.values()), 3)
    appears_complete = _appears_complete(completed_batch_indices, batch_count)

    return {
        "checkpoint_exists": checkpoint.exists,
        "checkpoint_path": str(checkpoint.path),
        "batch_count": batch_count,
        "completed_batch_indices": completed_batch_indices,
        "rows_per_batch": dict(sorted(rows_per_batch.items(), key=lambda item: int(item[0]))),
        "elapsed_seconds_per_batch": dict(
            sorted(elapsed_seconds_per_batch.items(), key=lambda item: int(item[0]))
        ),
        "total_elapsed_seconds": total_elapsed_seconds,
        "slowest_batches": _slowest_batches(elapsed_seconds_per_batch),
        "batches_with_debug_paths": sorted(set(batches_with_debug_paths)),
        "total_checkpoint_rows": sum(rows_per_batch.values()),
        "corrupt_record_count": checkpoint.corrupt_records,
        "errors": list(checkpoint.errors),
        "appears_complete": appears_complete,
    }


def _structure_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "header_as_path_count": sum(1 for row in rows if _has_header_as_path(row)),
        "generic_column_path_count": sum(1 for row in rows if _has_generic_path_fragment(row)),
        "stale_section_title_count": sum(1 for row in rows if _has_stale_section_title(row)),
        "appendix_table_detected_count": sum(1 for row in rows if _has_appendix_table_evidence(row)),
        "continued_table_count": sum(1 for row in rows if _has_continued_table_evidence(row)),
    }


def _local_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(_row_section_title(row), []).append(row)

    by_section = []
    for section_title, section_rows in grouped_rows.items():
        structure = _structure_metrics(section_rows)
        by_section.append(
            {
                "section_title": section_title,
                "row_count": len(section_rows),
                "needs_review_count": sum(1 for row in section_rows if _needs_review(row)),
                **structure,
            }
        )

    by_section.sort(key=lambda item: (-int(item["row_count"]), str(item["section_title"])))
    return {"by_section": by_section}


def _run_completeness(
    inputs: EvalInputs,
    row_extraction: dict[str, Any],
) -> dict[str, Any]:
    missing_final_artifacts = [
        loaded.path.name
        for loaded in inputs.files.values()
        if not loaded.exists
        and loaded.path.name in {"rule_table.json", "rule_tree.json"}
    ]
    file_errors = {
        name: loaded.error
        for name, loaded in inputs.files.items()
        if loaded.exists and loaded.error
    }
    checkpoint_corrupt_records = {
        "row": inputs.row_checkpoint.corrupt_records,
        "evidence": inputs.evidence_checkpoint.corrupt_records,
        "block": inputs.block_checkpoint.corrupt_records,
    }
    checkpoint_error_count = {
        "row": len(inputs.row_checkpoint.errors),
        "evidence": len(inputs.evidence_checkpoint.errors),
        "block": len(inputs.block_checkpoint.errors),
    }

    return {
        "missing_final_artifacts": missing_final_artifacts,
        "file_errors": file_errors,
        "checkpoint_corrupt_records": checkpoint_corrupt_records,
        "checkpoint_error_count": checkpoint_error_count,
        "row_checkpoint_appears_complete": row_extraction["appears_complete"],
    }


def _risk_signals(
    rows: list[dict[str, Any]],
    inputs: EvalInputs,
    quality: dict[str, Any],
    structure: dict[str, Any],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for debug_file in inputs.debug_files:
        debug_text = debug_file.text.lower()
        if "unterminated string" in debug_text or "failed validation" in debug_text:
            signals.append(
                {
                    "type": "debug_json_failure",
                    "severity": "review",
                    "message": "Debug file contains JSON parsing or validation failure text.",
                    "path": _relative_path(debug_file.path, inputs.output_dir),
                }
            )
            break
    generic_fragment = next(
        (row for row in rows if _has_generic_path_fragment(row)),
        None,
    )
    if generic_fragment is not None:
        signals.append(
            {
                "type": "generic_path_fragment",
                "severity": "review",
                "message": "Path levels look like table headers or placeholder fragments.",
                "path_levels": list(_path_key(generic_fragment)),
            }
        )
    header_fragment = next((row for row in rows if _has_header_as_path(row)), None)
    if header_fragment is not None:
        signals.append(
            {
                "type": "header_as_path",
                "severity": "review",
                "message": "Path levels contain hierarchy column headers such as 类/项/目.",
                "path_levels": list(_path_key(header_fragment)),
                "count": structure["header_as_path_count"],
            }
        )
    stale_section = next((row for row in rows if _has_stale_section_title(row)), None)
    if stale_section is not None:
        signals.append(
            {
                "type": "stale_section_title",
                "severity": "review",
                "message": "Evidence text contains appendix/table markers while refs keep an unrelated section title.",
                "path_levels": list(_path_key(stale_section)),
                "count": structure["stale_section_title_count"],
            }
        )
    if quality["duplicate_path_count"]:
        signals.append(
            {
                "type": "duplicate_path",
                "severity": "review",
                "message": "Multiple classification rows share the same path.",
                "count": quality["duplicate_path_count"],
            }
        )
    if quality["missing_evidence_quote_count"] or quality["missing_evidence_refs_count"]:
        signals.append(
            {
                "type": "missing_evidence",
                "severity": "review",
                "message": "Some classification rows are missing evidence quotes or refs.",
                "missing_evidence_quote_count": quality["missing_evidence_quote_count"],
                "missing_evidence_refs_count": quality["missing_evidence_refs_count"],
            }
        )
    return signals


def _recommendation(
    inputs: EvalInputs,
    quality: dict[str, Any],
    run_completeness: dict[str, Any],
    risk_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []

    for artifact in run_completeness["missing_final_artifacts"]:
        reasons.append(f"missing final artifact: {artifact}")
    for file_key, error in run_completeness["file_errors"].items():
        reasons.append(f"{_display_final_artifact_name(file_key)} read error: {error}")
    if quality["validation_issue_count_by_severity"].get("high", 0):
        reasons.append("high severity validation issues present")
    if any(signal["type"] == "debug_json_failure" for signal in risk_signals):
        reasons.append("row extraction debug failures found")
    if inputs.row_checkpoint.exists and not run_completeness["row_checkpoint_appears_complete"]:
        reasons.append("row checkpoint does not appear complete")

    return {
        "merge_ready": not reasons,
        "reasons": reasons,
        "note": (
            "Advisory-only diagnostics from saved artifacts; this is not a business "
            "semantic correctness judgment."
        ),
    }


def _path_key(row: dict[str, Any]) -> tuple[str, ...]:
    levels = row.get("path_levels")
    if isinstance(levels, list):
        return tuple(str(level).strip() for level in levels if str(level).strip())
    path = row.get("path")
    if isinstance(path, str) and path.strip():
        return tuple(part.strip() for part in path.split("/") if part.strip())
    return ()


def _has_header_as_path(row: dict[str, Any]) -> bool:
    return any(level in HEADER_PATH_LEVELS for level in _path_key(row))


def _display_final_artifact_name(file_key: str) -> str:
    return {
        "rule_table_json": "rule_table.json",
        "rule_tree_json": "rule_tree.json",
        "review_report_md": "review_report.md",
    }.get(file_key, file_key)


def _relative_path(path: Any, root: Any) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _evidence_refs(row: dict[str, Any]) -> list[dict[str, Any]]:
    refs = row.get("evidence_refs")
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, dict)]


def _row_section_title(row: dict[str, Any]) -> str:
    for ref in _evidence_refs(row):
        section_title = str(ref.get("section_title") or "").strip()
        if section_title:
            return section_title
    return "unknown"


def _evidence_ref_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for ref in _evidence_refs(row):
        parts.append(str(ref.get("section_title") or ""))
        parts.append(str(ref.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _has_appendix_table_evidence(row: dict[str, Any]) -> bool:
    text = _evidence_ref_text(row)
    return bool(("附 录" in text or "附录" in text) and "表" in text)


def _has_continued_table_evidence(row: dict[str, Any]) -> bool:
    return "（续）" in _evidence_ref_text(row)


def _has_stale_section_title(row: dict[str, Any]) -> bool:
    for ref in _evidence_refs(row):
        section_title = str(ref.get("section_title") or "")
        text = str(ref.get("text") or "")
        if not section_title or not text:
            continue
        has_table_context = ("附 录" in text or "附录" in text) and "表" in text
        section_mentions_table = ("附 录" in section_title or "附录" in section_title) and "表" in section_title
        if has_table_context and not section_mentions_table:
            return True
    return False


def _needs_review(row: dict[str, Any]) -> bool:
    if bool(row.get("needs_review")):
        return True
    status = str(row.get("review_status", "")).lower()
    return "review" in status


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _checkpoint_row_count(record: dict[str, Any]) -> int:
    for key in ("classification_rows", "rows", "output_rows", "items"):
        value = record.get(key)
        if isinstance(value, list):
            return len(value)
    row_count = _int_or_none(record.get("row_count"))
    return row_count or 0


def _appears_complete(completed_indices: list[int], batch_count: int) -> bool:
    if batch_count <= 0:
        return False
    completed = set(completed_indices)
    one_based = set(range(1, batch_count + 1))
    zero_based = set(range(batch_count))
    return one_based.issubset(completed) or zero_based.issubset(completed)


def _slowest_batches(elapsed_seconds_per_batch: dict[str, float]) -> list[dict[str, Any]]:
    slowest = sorted(
        elapsed_seconds_per_batch.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"batch_index": int(batch_index), "elapsed_seconds": elapsed_seconds}
        for batch_index, elapsed_seconds in slowest[:3]
    ]


def _has_generic_path_fragment(row: dict[str, Any]) -> bool:
    levels = _path_key(row)
    if not levels:
        return False
    generic_tokens = {
        "名称",
        "代码",
        "类别",
        "序号",
        "说明",
    } | GENERIC_COLUMN_PATH_LEVELS
    short_fragment_count = sum(
        1 for level in levels if len(level) <= 1 and not level.isascii()
    )
    if short_fragment_count >= 2:
        return True
    return any(level in generic_tokens for level in levels)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _float_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return numeric_value
        return 0.0
    if isinstance(value, str):
        try:
            numeric_value = float(value)
        except ValueError:
            return 0.0
        if math.isfinite(numeric_value):
            return numeric_value
        return 0.0
    return 0.0
