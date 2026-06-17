from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MAPPINGS: dict[str, dict[str, Any]] = {
    "row_b96c15c23f6c": {
        "aliases": ["患者信息", "病人信息", "就诊人信息"],
        "element_codes": [
            "DE02.01.039.01",
            "DE02.01.030.02",
            "DE01.00.021.00",
            "DE02.01.005.00",
            "DE02.01.040.00",
            "DE02.01.026.00",
        ],
    },
    "row_9bdc5f3ff50e": {
        "aliases": ["临床服务信息", "诊疗服务"],
        "element_codes": [
            "DE05.01.024.00",
            "DE05.01.024.01",
            "DE05.01.024.02",
            "DE05.01.024.03",
            "DE05.01.024.10",
        ],
    },
    "row_dbe5e197369a": {
        "aliases": ["电子病历信息", "病历信息"],
        "element_codes": [
            "DE01.00.004.00",
            "DE05.01.024.00",
            "DE05.01.024.01",
            "DE05.01.024.02",
            "DE05.01.024.03",
            "DE05.01.024.10",
        ],
    },
    "row_a4911580fc4a": {
        "aliases": ["电子病历临床诊疗", "临床诊疗信息"],
        "element_codes": [
            "DE05.01.024.00",
            "DE05.01.024.01",
            "DE05.01.024.02",
            "DE05.01.024.03",
            "DE05.01.024.10",
        ],
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _load_data_elements(data_elements_dir: Path) -> dict[str, dict[str, Any]]:
    elements: dict[str, dict[str, Any]] = {}
    for path in sorted(data_elements_dir.glob("part_*.json")):
        data = _load_json(path)
        for element in data.get("elements") or []:
            if not isinstance(element, dict):
                continue
            code = str(element.get("element_code") or "").strip()
            if code:
                elements[code] = element
    return elements


def _element_ref(element: dict[str, Any]) -> str:
    source_part = str(element.get("source_part") or "").strip()
    code = str(element.get("element_code") or "").strip()
    return f"{source_part}:{code}" if source_part else code


def apply_mappings(
    rule_table_path: Path,
    data_elements_dir: Path,
    mappings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mappings = mappings or DEFAULT_MAPPINGS
    data = _load_json(rule_table_path)
    rows = data.get("classification_rows") or []
    elements = _load_data_elements(data_elements_dir)
    updated_rows = 0
    missing_rows: list[str] = []
    missing_elements: list[str] = []

    rows_by_id = {
        str(row.get("row_id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("row_id") or "").strip()
    }
    for row_id, mapping in mappings.items():
        row = rows_by_id.get(row_id)
        if row is None:
            missing_rows.append(row_id)
            continue

        selected_elements: list[dict[str, Any]] = []
        for code in _string_list(mapping.get("element_codes")):
            element = elements.get(code)
            if element is None:
                missing_elements.append(code)
                continue
            selected_elements.append(element)

        examples = _dedupe(
            _string_list(row.get("data_range_examples"))
            + [
                str(element.get("element_name") or "").strip()
                for element in selected_elements
                if str(element.get("element_name") or "").strip()
            ]
        )
        refs = _dedupe(
            _string_list(row.get("data_element_refs"))
            + [_element_ref(element) for element in selected_elements if _element_ref(element)]
        )
        aliases = _dedupe(_string_list(row.get("aliases")) + _string_list(mapping.get("aliases")))
        reuse_allowed = _dedupe(
            _string_list(row.get("reuse_allowed_fields"))
            + ["path_levels", "data_range_examples", "data_element_refs"]
        )

        row["data_range_examples"] = examples
        row["data_element_refs"] = refs
        if aliases:
            row["aliases"] = aliases
        row["reuse_allowed_fields"] = reuse_allowed
        row["curation_status"] = "data_elements_poc"
        updated_rows += 1

    _write_json(rule_table_path, data)
    return {
        "rule_table": str(rule_table_path),
        "data_elements_dir": str(data_elements_dir),
        "updated_rows": updated_rows,
        "missing_rows": missing_rows,
        "missing_elements": sorted(set(missing_elements)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply curated WS/T 363 data-element mappings to reference rows."
    )
    parser.add_argument(
        "--rule-table",
        default="reference_library/wst787_2021/rule_table.json",
        help="Reference rule_table.json to update.",
    )
    parser.add_argument(
        "--data-elements",
        default="reference_library/data_elements/wst363",
        help="Directory containing WST363 part_*.json files.",
    )
    args = parser.parse_args(argv)
    summary = apply_mappings(Path(args.rule_table), Path(args.data_elements))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
