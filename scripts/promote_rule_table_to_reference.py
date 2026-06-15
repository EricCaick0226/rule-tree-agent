from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.rule_table_linker import load_rule_table_rows  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote a reviewed rule_table.json into the reference library."
    )
    parser.add_argument("--rule-table", required=True, help="Reviewed rule_table.json path.")
    parser.add_argument("--name", required=True, help="Reference display name.")
    parser.add_argument("--type", required=True, dest="source_type", help="Reference source type.")
    parser.add_argument("--description", default="", help="Reference description.")
    parser.add_argument("--out", required=True, help="Reference output directory.")
    args = parser.parse_args(argv)

    rule_table_path = Path(args.rule_table)
    rows = load_rule_table_rows(rule_table_path)
    if not rows:
        raise ValueError(f"rule_table must contain at least one classification row: {rule_table_path}")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rule_table_path, output_dir / "rule_table.json")

    metadata = {
        "name": args.name,
        "source_type": args.source_type,
        "description": args.description,
        "source_rule_table": str(rule_table_path),
        "curation_status": "reviewed_seed",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'rule_table.json'}")
    print(f"Wrote {output_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
