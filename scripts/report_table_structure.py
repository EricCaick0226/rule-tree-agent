from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.document_parser import chunk_documents, parse_documents
from src.io.table_segmenter import segment_table_chunks_for_row_extraction
from src.io.table_structure_report import (
    build_table_structure_report,
    filtered_report_to_dict,
    render_filtered_table_structure_markdown,
    render_table_structure_markdown,
    report_to_dict,
)


def write_table_structure_report(txt_path: Path, out_dir: Path) -> None:
    documents = parse_documents([str(txt_path)])
    chunks = chunk_documents(documents)
    block_signals = {
        chunk.chunk_id: {"block_signal": "table_like"}
        for chunk in chunks
        if chunk.text.strip()
    }
    segments = segment_table_chunks_for_row_extraction(
        chunks,
        block_signals=block_signals,
        max_chars=5000,
    )
    report = build_table_structure_report(segments)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table_structure_report.json").write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "table_structure_report.md").write_text(
        render_table_structure_markdown(report),
        encoding="utf-8",
    )
    (out_dir / "table_structure_filtered_report.json").write_text(
        json.dumps(filtered_report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "table_structure_filtered_report.md").write_text(
        render_filtered_table_structure_markdown(report),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a table structure diagnostic report.")
    parser.add_argument("--txt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        write_table_structure_report(args.txt, args.out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.out}/table_structure_report.json")
    print(f"Wrote {args.out}/table_structure_report.md")
    print(f"Wrote {args.out}/table_structure_filtered_report.json")
    print(f"Wrote {args.out}/table_structure_filtered_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
