from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.reference_rule_library import (  # noqa: E402
    build_reference_suggestion_report,
    load_reference_library,
    render_reference_suggestions_markdown,
)
from src.io.rule_table_linker import load_rule_table_rows  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build review-only reference suggestions from a curated rule-table library."
    )
    parser.add_argument("--current", required=True, help="Current rule_table.json path.")
    parser.add_argument("--library", required=True, help="Reference library directory.")
    parser.add_argument("--out", required=True, help="Output directory for reference_suggestions.json/md.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum matches per current row.")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum match score.")
    args = parser.parse_args(argv)

    current_path = Path(args.current)
    current_rows = load_rule_table_rows(current_path)
    references, warnings = load_reference_library(Path(args.library))
    report = build_reference_suggestion_report(
        current_path=str(current_path),
        current_rows=current_rows,
        references=references,
        warnings=warnings,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference_suggestions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "reference_suggestions.md").write_text(
        render_reference_suggestions_markdown(report),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'reference_suggestions.json'}")
    print(f"Wrote {output_dir / 'reference_suggestions.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
