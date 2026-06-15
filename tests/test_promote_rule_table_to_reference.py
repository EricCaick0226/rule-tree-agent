from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from scripts.promote_rule_table_to_reference import main


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class PromoteRuleTableToReferenceTests(unittest.TestCase):
    def test_main_copies_rule_table_and_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs_233" / "rule_table.json"
            out = root / "reference_library" / "wst787_2021"
            write_json(
                source,
                {
                    "classification_rows": [
                        {
                            "row_id": "ref_1",
                            "path_levels": ["个人敏感信息", "身份证号"],
                            "description": "证据不足，无法从当前文档确定",
                            "description_source": "insufficient",
                            "data_range_examples": ["身份证号"],
                        }
                    ]
                },
            )

            exit_code = main(
                [
                    "--rule-table",
                    str(source),
                    "--name",
                    "WST 787-2021",
                    "--type",
                    "national_standard",
                    "--description",
                    "国家卫生信息资源分类与编码管理规范",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(exit_code, 0)
            copied = json.loads((out / "rule_table.json").read_text(encoding="utf-8"))
            metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(copied["classification_rows"]), 1)
            self.assertEqual(metadata["name"], "WST 787-2021")
            self.assertEqual(metadata["source_type"], "national_standard")
            self.assertEqual(metadata["description"], "国家卫生信息资源分类与编码管理规范")
            self.assertEqual(metadata["source_rule_table"], str(source))
            self.assertEqual(metadata["curation_status"], "reviewed_seed")

    def test_main_rejects_empty_rule_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "empty" / "rule_table.json"
            out = root / "reference_library" / "empty"
            write_json(source, {"classification_rows": []})

            with self.assertRaisesRegex(ValueError, "must contain at least one classification row"):
                main(
                    [
                        "--rule-table",
                        str(source),
                        "--name",
                        "Empty",
                        "--type",
                        "national_standard",
                        "--out",
                        str(out),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
