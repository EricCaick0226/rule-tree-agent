from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.wps_txt_cleaner import clean_wps_txt_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean TXT exported from WPS PDF-to-TXT before running the row-first pipeline."
    )
    parser.add_argument("--input", required=True, type=Path, help="WPS-exported TXT input path.")
    parser.add_argument("--out", required=True, type=Path, help="Cleaned TXT output path.")
    parser.add_argument("--review-out", type=Path, default=None, help="Optional concise review JSON output path.")
    args = parser.parse_args(argv)

    try:
        result = clean_wps_txt_file(args.input, args.out, args.review_out)
    except Exception as exc:
        print(f"Failed to clean WPS TXT: {exc}", file=sys.stderr)
        return 1

    print(f"Cleaned TXT written: {args.out}")
    stats_text = ", ".join(f"{key}={value}" for key, value in result.stats.items())
    print(f"Stats: {stats_text}")
    if args.review_out is not None:
        print(f"Review JSON written: {args.review_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
