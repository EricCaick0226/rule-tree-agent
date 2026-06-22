from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.report_reference_reuse_offline import generate_reference_reuse_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": "direct",
            "reference_trust_level": "authoritative",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_patient_info",
                    "path_levels": ["基础资源", "服务范围与对象", "患者", "患者信息"],
                    "description": "患者信息包括患者身份识别和联系方式等基本资料。",
                    "description_source": "classification_standard_excel",
                    "source_confidence": "curated_answer",
                }
            ]
        },
    )
    return library


class ReportReferenceReuseOfflineTests(unittest.TestCase):
    def test_reports_direct_reuse_deltas_and_match_details(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rule_table = root / "rule_table.json"
            library = _reference_library(root)
            out_dir = root / "reuse_report"
            _write_json(
                rule_table,
                {
                    "classification_rows": [
                        {
                            "row_id": "row_patient_info",
                            "path_levels": ["服务范围与对象", "患者", "患者信息"],
                            "description": "证据不足，无法从当前文档确定",
                            "description_source": "insufficient",
                            "status": "proposed",
                        }
                    ]
                },
            )

            report = generate_reference_reuse_report(rule_table, library, out_dir)
            json_report = json.loads((out_dir / "direct_reuse_report.json").read_text(encoding="utf-8"))
            markdown_report = (out_dir / "direct_reuse_report.md").read_text(encoding="utf-8")

        self.assertEqual(report["original_rows"], 1)
        self.assertEqual(report["direct_reused_rows"], 1)
        self.assertEqual(report["review_candidates_added"], 0)
        self.assertEqual(report["match_types"], {"parent_and_leaf": 1})
        self.assertEqual(len(json_report["direct_reuse_rows"]), 1)
        row = json_report["direct_reuse_rows"][0]
        self.assertEqual(row["row_id"], "row_patient_info")
        self.assertEqual(row["old_path"], ["服务范围与对象", "患者", "患者信息"])
        self.assertEqual(row["new_path"], ["基础资源", "服务范围与对象", "患者", "患者信息"])
        self.assertEqual(row["reference_path"], ["基础资源", "服务范围与对象", "患者", "患者信息"])
        self.assertEqual(row["match_type"], "parent_and_leaf")
        self.assertEqual(row["reused_fields"], ["path_levels", "description"])
        self.assertEqual(row["description_source"], "classification_standard_excel")
        self.assertIn("parent_and_leaf", markdown_report)
        self.assertIn("服务范围与对象 / 患者 / 患者信息", markdown_report)


if __name__ == "__main__":
    unittest.main()
