from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.split_reference_candidates_from_output import split_reference_candidates


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SplitReferenceCandidatesFromOutputTests(unittest.TestCase):
    def test_splits_reference_candidates_out_of_rule_table(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "old_output"
            output_dir = root / "cleaned_output"
            _write_json(
                input_dir / "rule_table.json",
                {
                    "classification_schema": {"max_depth": 3, "source": "test"},
                    "grade_scheme": [],
                    "classification_rows": [
                        {
                            "row_id": "current",
                            "path_levels": ["基础资源", "设备资源", "硬件设备"],
                            "description": "硬件设备相关信息。",
                            "description_source": "quoted",
                            "reference_prefilled_fields": ["description"],
                        },
                        {
                            "row_id": "candidate",
                            "path_levels": ["基础资源", "设备资源", "软件设备"],
                            "description": "软件设备相关信息。",
                            "description_source": "reference_library",
                            "row_source": "reference_library",
                            "content_source": "reference_library",
                            "inclusion_status": "review_candidate",
                            "evidence_status": "reference_only",
                            "reference_matches": [
                                {"reference_row_id": "ref_software"}
                            ],
                        },
                    ],
                    "validation_issues": [],
                },
            )

            summary = split_reference_candidates(input_dir, output_dir)
            table = json.loads((output_dir / "rule_table.json").read_text(encoding="utf-8"))
            candidates = json.loads(
                (output_dir / "reference_candidates.json").read_text(encoding="utf-8")
            )
            report = (output_dir / "review_report.md").read_text(encoding="utf-8")
            candidate_md = (output_dir / "reference_candidates.md").read_text(encoding="utf-8")

        self.assertEqual(summary["original_rows"], 2)
        self.assertEqual(summary["classification_rows"], 1)
        self.assertEqual(summary["reference_candidate_rows"], 1)
        self.assertEqual(table["classification_rows"][0]["row_id"], "current")
        self.assertEqual(candidates["reference_candidate_rows"][0]["row_id"], "candidate")
        self.assertIn("- Reference candidate rows: 1", report)
        self.assertIn("ref_software", candidate_md)


if __name__ == "__main__":
    unittest.main()
