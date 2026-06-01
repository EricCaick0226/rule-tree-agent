from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.eval_harness.loader import load_output_dir
from src.eval_harness.metrics import build_eval_report
from src.eval_harness.report import render_json_report, render_markdown_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval_harness",
        description="Read saved rule-tree-agent outputs and write eval reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="diagnose a saved output directory",
    )
    diagnose_parser.add_argument("output_dir")

    args = parser.parse_args(argv)
    if args.command == "diagnose":
        return _diagnose(args.output_dir)
    return 2


def _diagnose(output_dir: str) -> int:
    root = Path(output_dir)
    if not root.is_dir():
        print(f"error: output_dir is not a directory: {root}", file=sys.stderr)
        return 2

    json_report_path = root / "eval_report.json"
    markdown_report_path = root / "eval_report.md"

    inputs = load_output_dir(root)
    report = build_eval_report(inputs)
    json_report = render_json_report(report)
    markdown_report = render_markdown_report(report)

    try:
        json_report_path.write_text(json_report, encoding="utf-8")
        markdown_report_path.write_text(markdown_report, encoding="utf-8")
    except OSError as exc:
        print(f"error: failed to write eval reports: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {json_report_path}")
    print(f"wrote {markdown_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
