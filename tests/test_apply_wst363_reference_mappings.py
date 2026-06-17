from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.apply_wst363_reference_mappings import apply_mappings


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ApplyWst363ReferenceMappingsTests(unittest.TestCase):
    def test_applies_curated_data_element_examples_to_reference_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rule_table = root / "rule_table.json"
            data_dir = root / "data_elements"
            _write_json(
                rule_table,
                {
                    "classification_rows": [
                        {
                            "row_id": "ref_patient",
                            "path_levels": ["患者"],
                            "recommended_grade": "3级",
                            "data_range_examples": [],
                        },
                        {
                            "row_id": "ref_other",
                            "path_levels": ["其他"],
                            "data_range_examples": [],
                        },
                    ]
                },
            )
            _write_json(
                data_dir / "part_03.json",
                {
                    "elements": [
                        {
                            "element_code": "DE02.01.039.01",
                            "element_name": "患者姓名",
                            "source_part": "WS/T 363.3—2023",
                        },
                        {
                            "element_code": "DE02.01.030.02",
                            "element_name": "患者身份证件号码",
                            "source_part": "WS/T 363.3—2023",
                        },
                    ]
                },
            )

            summary = apply_mappings(
                rule_table_path=rule_table,
                data_elements_dir=data_dir,
                mappings={
                    "ref_patient": {
                        "element_codes": ["DE02.01.039.01", "DE02.01.030.02"],
                        "aliases": ["患者信息"],
                    }
                },
            )

            data = json.loads(rule_table.read_text(encoding="utf-8"))

        patient = data["classification_rows"][0]
        other = data["classification_rows"][1]
        self.assertEqual(summary["updated_rows"], 1)
        self.assertEqual(patient["recommended_grade"], "3级")
        self.assertEqual(patient["data_range_examples"], ["患者姓名", "患者身份证件号码"])
        self.assertEqual(
            patient["data_element_refs"],
            ["WS/T 363.3—2023:DE02.01.039.01", "WS/T 363.3—2023:DE02.01.030.02"],
        )
        self.assertEqual(patient["aliases"], ["患者信息"])
        self.assertIn("data_range_examples", patient["reuse_allowed_fields"])
        self.assertEqual(patient["curation_status"], "data_elements_poc")
        self.assertEqual(other["data_range_examples"], [])


if __name__ == "__main__":
    unittest.main()
