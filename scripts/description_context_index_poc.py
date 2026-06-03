from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.steps.description_context_index import (
    build_description_context_index,
    retrieve_description_context_pack,
)
from src.steps.description_context_kb import flag_description_quality


def _load_rule_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("classification_rows") if isinstance(data, dict) else None
    return [row for row in rows or [] if isinstance(row, dict)]


def _row_priority(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        1 if row.get("recommended_grade") else 0,
        1 if row.get("data_range_examples") else 0,
        len(row.get("path_levels") or []),
    )


def build_description_context_index_report(
    txt_path: Path,
    rule_table_path: Path,
    limit: int,
) -> dict[str, Any]:
    text = txt_path.read_text(encoding="utf-8")
    rows = _load_rule_rows(rule_table_path)
    units = build_description_context_index(text)
    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []

    for row in rows:
        flags = flag_description_quality(row)
        if not flags:
            continue
        candidates.append(
            (
                _row_priority(row),
                {
                    "row_id": row.get("row_id", ""),
                    "path": " / ".join(str(level) for level in row.get("path_levels") or []),
                    "current_description": row.get("description", ""),
                    "description_quality_flags": flags,
                    "context_pack": retrieve_description_context_pack(row, units, top_k=5),
                },
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    sampled = [
        report_row
        for _priority, report_row in candidates[: max(0, limit)]
    ]

    return {
        "txt_path": str(txt_path),
        "rule_table_path": str(rule_table_path),
        "total_row_count": len(rows),
        "context_unit_count": len(units),
        "sampled_row_count": len(sampled),
        "unit_kind_counts": {
            kind: sum(1 for unit in units if unit.get("kind") == kind)
            for kind in sorted({str(unit.get("kind") or "") for unit in units})
        },
        "rows": sampled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a read-only v2 description-context retrieval report for weak classification descriptions."
    )
    parser.add_argument("--txt", required=True, help="Source TXT document path.")
    parser.add_argument("--rule-table", required=True, help="rule_table.json path.")
    parser.add_argument("--out", required=True, help="Output JSON report path.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum weak-description rows to include.")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    report = build_description_context_index_report(
        txt_path=Path(args.txt).expanduser().resolve(),
        rule_table_path=Path(args.rule_table).expanduser().resolve(),
        limit=max(0, args.limit),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} rows={report['sampled_row_count']} units={report['context_unit_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
