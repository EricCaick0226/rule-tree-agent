from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.io.document_structure import detect_structure_signal


@dataclass(frozen=True)
class StructureSignalCase:
    line: str
    expected_kind: str | None
    expected_title: str | None = None
    line_number: int | None = None


def evaluate_structure_signal_cases(cases: list[StructureSignalCase]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    by_expected_kind: dict[str, dict[str, int]] = {}
    matched_count = 0
    false_positive_count = 0
    false_negative_count = 0
    kind_mismatch_count = 0
    title_mismatch_count = 0

    for case in cases:
        signal = detect_structure_signal(case.line, line_number=case.line_number)
        actual_kind = signal.kind if signal else None
        actual_title = signal.title if signal else None
        expected_kind = case.expected_kind
        expected_title = case.expected_title

        if expected_kind:
            bucket = by_expected_kind.setdefault(expected_kind, {"expected": 0, "matched": 0})
            bucket["expected"] += 1

        kind_matches = actual_kind == expected_kind
        title_matches = expected_title is None or actual_title == expected_title
        if kind_matches and title_matches:
            matched_count += 1
            if expected_kind:
                by_expected_kind[expected_kind]["matched"] += 1
            continue

        if expected_kind is None and actual_kind is not None:
            false_positive_count += 1
            failure_type = "false_positive"
        elif expected_kind is not None and actual_kind is None:
            false_negative_count += 1
            failure_type = "false_negative"
        elif not kind_matches:
            kind_mismatch_count += 1
            failure_type = "kind_mismatch"
        else:
            failure_type = "title_mismatch"

        if not title_matches:
            title_mismatch_count += 1

        failures.append(
            {
                "type": failure_type,
                "line": case.line,
                "line_number": case.line_number,
                "expected_kind": expected_kind,
                "expected_title": expected_title,
                "actual_kind": actual_kind,
                "actual_title": actual_title,
            }
        )

    total = len(cases)
    return {
        "total": total,
        "matched_count": matched_count,
        "accuracy": round(matched_count / total, 3) if total else 0.0,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "kind_mismatch_count": kind_mismatch_count,
        "title_mismatch_count": title_mismatch_count,
        "by_expected_kind": by_expected_kind,
        "failures": failures,
    }
