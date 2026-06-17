from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.wst363_data_elements import write_wst363_payload  # noqa: E402


INPUTS = {
    "part_02.json": Path("data/input_docs/1733821987071_29917_副本.txt"),
    "part_03.json": Path("data/input_docs/1739782590964_77827_副本.txt"),
    "part_10.json": Path("data/input_docs/1733821986179_91444_副本.txt"),
}
OUTPUT_DIR = Path("reference_library/data_elements/wst363")


def main() -> None:
    total = 0
    parts = []
    for output_name, input_path in INPUTS.items():
        output_path = OUTPUT_DIR / output_name
        payload = write_wst363_payload(input_path, output_path)
        total += int(payload["element_count"])
        parts.append(
            {
                "part": payload["part"],
                "standard": payload["standard"],
                "title": payload["title"],
                "source_path": str(input_path),
                "output_path": str(output_path),
                "element_count": payload["element_count"],
                "status": "parsed",
            }
        )
        print(
            output_path,
            payload["standard"],
            payload["title"],
            f"elements={payload['element_count']}",
        )
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = {
        "standard_family": "WS/T 363—2023",
        "library_type": "data_element_library",
        "parts": sorted(parts, key=lambda item: item["part"]),
        "total_elements": total,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path, f"parts={len(parts)}")
    print(f"total_elements={total}")


if __name__ == "__main__":
    main()
