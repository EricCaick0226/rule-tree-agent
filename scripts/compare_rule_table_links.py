from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.rule_table_linker import (  # noqa: E402
    build_rule_table_links,
    links_to_dicts,
    load_rule_table_rows,
    render_rule_table_link_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a read-only similarity report between two rule_table.json files."
    )
    parser.add_argument("--current", required=True, help="Current rule_table.json path.")
    parser.add_argument("--reference", required=True, help="Reference or historical rule_table.json path.")
    parser.add_argument("--out", required=True, help="Output directory for link_report.json/md.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum matches per current row.")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum similarity score.")
    parser.add_argument("--max-links", type=int, default=80, help="Maximum linked current rows to report.")
    args = parser.parse_args()

    current_rows = load_rule_table_rows(Path(args.current))
    reference_rows = load_rule_table_rows(Path(args.reference))
    links = build_rule_table_links(
        current_rows,
        reference_rows,
        top_k=args.top_k,
        min_score=args.min_score,
        max_links=args.max_links,
    )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "current": str(Path(args.current)),
        "reference": str(Path(args.reference)),
        "current_rows": len(current_rows),
        "reference_rows": len(reference_rows),
        "linked_current_rows": len(links),
        "links": links_to_dicts(links),
    }
    (output_dir / "link_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "link_report.md").write_text(
        render_rule_table_link_markdown(links),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'link_report.json'}")
    print(f"Wrote {output_dir / 'link_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
