from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from scripts.build_reference_suggestions import main


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class BuildReferenceSuggestionsCliTests(unittest.TestCase):
    def test_main_writes_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "outputs_current" / "rule_table.json"
            library = root / "reference_library"
            out = root / "outputs_current"
            write_json(
                current,
                {
                    "classification_rows": [
                        {
                            "row_id": "cur_1",
                            "path_levels": ["个人信息", "身份证号"],
                            "description": "证据不足，无法从当前文档确定",
                            "description_source": "insufficient",
                            "data_range_examples": ["身份证号"],
                        }
                    ]
                },
            )
            write_json(
                library / "wst787_2021" / "metadata.json",
                {
                    "name": "WST 787-2021",
                    "source_type": "national_standard",
                    "description": "国家卫生信息资源分类与编码管理规范",
                },
            )
            write_json(
                library / "wst787_2021" / "rule_table.json",
                {
                    "classification_rows": [
                        {
                            "row_id": "ref_1",
                            "path_levels": ["个人敏感信息", "身份证号"],
                            "description": "证据不足，无法从当前文档确定",
                            "description_source": "insufficient",
                            "data_range_examples": ["身份证号"],
                        },
                        {
                            "row_id": "ref_2",
                            "path_levels": ["个人敏感信息", "医保卡号"],
                            "description": "医保卡号码。",
                            "description_source": "quoted",
                            "data_range_examples": ["医保卡号"],
                        },
                    ]
                },
            )

            exit_code = main(
                [
                    "--current",
                    str(current),
                    "--library",
                    str(library),
                    "--out",
                    str(out),
                    "--min-score",
                    "0.2",
                ]
            )

            self.assertEqual(exit_code, 0)
            report_json = out / "reference_suggestions.json"
            report_md = out / "reference_suggestions.md"
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["matched_current_rows"]), 1)
            self.assertEqual(len(payload["missing_reference_suggestions"]), 1)
            markdown = report_md.read_text(encoding="utf-8")
            self.assertIn("# Reference Suggestions", markdown)
            self.assertIn("not current-document evidence", markdown)


if __name__ == "__main__":
    unittest.main()
