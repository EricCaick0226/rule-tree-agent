from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.wst363_data_elements import build_wst363_payload, write_wst363_payload  # noqa: E402


INPUT_DIR = Path("data/input_docs")
OUTPUT_DIR = Path("reference_library/data_elements/wst363")


def discover_wst363_inputs(input_dir: Path = INPUT_DIR) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for input_path in sorted(input_dir.glob("*.txt")):
        payload = build_wst363_payload(
            input_path.read_text(encoding="utf-8"),
            source_path=str(input_path),
        )
        part = str(payload.get("part") or "").strip()
        standard = str(payload.get("standard") or "").strip()
        if not part or not standard.startswith("WS/T 363."):
            continue
        if part in inputs:
            raise ValueError(f"duplicate WS/T 363 part {part}: {inputs[part]} and {input_path}")
        inputs[part] = input_path
    return inputs


def build_library(inputs: dict[str, Path], output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("part_*.json"):
        stale_path.unlink()

    total = 0
    parts = []
    for part, input_path in sorted(inputs.items()):
        output_path = output_dir / f"part_{part}.json"
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

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "standard_family": "WS/T 363—2023",
        "library_type": "data_element_library",
        "parts": sorted(parts, key=lambda item: item["part"]),
        "total_elements": total,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path, f"parts={len(parts)}")
    print(f"total_elements={total}")
    return manifest


def main() -> None:
    build_library(discover_wst363_inputs(INPUT_DIR), OUTPUT_DIR)


if __name__ == "__main__":
    main()
