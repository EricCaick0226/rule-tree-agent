from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from src.io.reference_rule_library import (
    REVIEW_ONLY_REASON,
    build_reference_suggestion_report,
    load_reference_library,
    render_reference_suggestions_markdown,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ReferenceRuleLibraryTests(unittest.TestCase):
    def test_load_reference_library_reads_metadata_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "wst787_2021" / "metadata.json",
                {
                    "name": "WST 787-2021",
                    "source_type": "national_standard",
                    "description": "国家卫生信息资源分类与编码管理规范",
                },
            )
            write_json(
                root / "wst787_2021" / "rule_table.json",
                {
                    "classification_rows": [
                        {
                            "row_id": "ref_1",
                            "path_levels": ["个人信息", "身份证号"],
                            "description": "身份证件号码。",
                            "description_source": "quoted",
                            "data_range_examples": ["身份证号"],
                        }
                    ]
                },
            )

            references, warnings = load_reference_library(root)

        self.assertEqual(warnings, [])
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].name, "WST 787-2021")
        self.assertEqual(references[0].source_type, "national_standard")
        self.assertTrue(references[0].path.endswith("wst787_2021/rule_table.json"))
        self.assertEqual(len(references[0].rows), 1)

    def test_load_reference_library_rejects_malformed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "bad_ref" / "metadata.json", {"name": "missing type"})
            write_json(root / "bad_ref" / "rule_table.json", {"classification_rows": []})

            with self.assertRaisesRegex(ValueError, "metadata.json must contain non-empty name and source_type"):
                load_reference_library(root)

    def test_build_report_separates_matches_and_missing_reference_suggestions(self) -> None:
        current_rows = [
            {
                "row_id": "cur_1",
                "path_levels": ["个人信息", "身份证号"],
                "description": "证据不足，无法从当前文档确定",
                "description_source": "insufficient",
                "data_range_examples": ["身份证号"],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "wst787_2021" / "metadata.json",
                {
                    "name": "WST 787-2021",
                    "source_type": "national_standard",
                    "description": "国家卫生信息资源分类与编码管理规范",
                },
            )
            write_json(
                root / "wst787_2021" / "rule_table.json",
                {
                    "classification_rows": [
                        {
                            "row_id": "ref_match",
                            "path_levels": ["个人敏感信息", "身份证号"],
                            "description": "证据不足，无法从当前文档确定",
                            "description_source": "insufficient",
                            "data_range_examples": ["身份证号"],
                            "recommended_grade": "3级",
                        },
                        {
                            "row_id": "ref_missing",
                            "path_levels": ["个人敏感信息", "医保卡号"],
                            "description": "医保卡号码。",
                            "description_source": "quoted",
                            "data_range_examples": ["医保卡号"],
                            "recommended_grade": "3级",
                        },
                    ]
                },
            )
            references, warnings = load_reference_library(root)

        report = build_reference_suggestion_report(
            current_path="outputs_current/rule_table.json",
            current_rows=current_rows,
            references=references,
            warnings=warnings,
            top_k=2,
            min_score=0.2,
        )

        self.assertEqual(report["current"], "outputs_current/rule_table.json")
        self.assertEqual(report["warnings"], [])
        self.assertEqual(len(report["references"]), 1)
        self.assertEqual(len(report["matched_current_rows"]), 1)
        self.assertEqual(
            report["matched_current_rows"][0]["matches"][0]["reference_row_id"],
            "ref_match",
        )
        self.assertEqual(len(report["missing_reference_suggestions"]), 1)
        missing = report["missing_reference_suggestions"][0]
        self.assertEqual(missing["reference_row_id"], "ref_missing")
        self.assertEqual(missing["reference_path"], ["个人敏感信息", "医保卡号"])
        self.assertEqual(missing["reference_grade"], "3级")
        self.assertEqual(missing["suggestion_type"], "missing_reference_candidate")
        self.assertEqual(missing["source"], "reference_library")
        self.assertTrue(missing["needs_review"])
        self.assertEqual(missing["review_reason"], REVIEW_ONLY_REASON)

    def test_markdown_states_reference_boundary(self) -> None:
        report = {
            "current": "outputs_current/rule_table.json",
            "references": [
                {
                    "name": "WST 787-2021",
                    "type": "national_standard",
                    "path": "reference_library/wst787_2021/rule_table.json",
                    "rows": 2,
                }
            ],
            "warnings": [],
            "matched_current_rows": [
                {
                    "current_row_id": "cur_1",
                    "current_path": ["个人信息", "身份证号"],
                    "current_description_source": "insufficient",
                    "matches": [
                        {
                            "reference_name": "WST 787-2021",
                            "reference_type": "national_standard",
                            "reference_file": "reference_library/wst787_2021/rule_table.json",
                            "reference_row_id": "ref_match",
                            "reference_path": ["个人敏感信息", "身份证号"],
                            "score": 1.0,
                            "shared_terms": ["身份证号"],
                            "reference_description_source": "insufficient",
                        }
                    ],
                }
            ],
            "missing_reference_suggestions": [
                {
                    "reference_name": "WST 787-2021",
                    "reference_type": "national_standard",
                    "reference_file": "reference_library/wst787_2021/rule_table.json",
                    "reference_row_id": "ref_missing",
                    "reference_path": ["个人敏感信息", "医保卡号"],
                    "reference_description": "医保卡号码。",
                    "reference_grade": "3级",
                    "suggestion_type": "missing_reference_candidate",
                    "source": "reference_library",
                    "match_reason": "当前输出未找到高相似匹配。",
                    "needs_review": True,
                    "review_reason": REVIEW_ONLY_REASON,
                }
            ],
        }

        markdown = render_reference_suggestions_markdown(report)

        self.assertIn("# Reference Suggestions", markdown)
        self.assertIn("Reference rows are review hints only; they are not current-document evidence.", markdown)
        self.assertIn("## Matched Current Rows", markdown)
        self.assertIn("个人信息 / 身份证号", markdown)
        self.assertIn("## Missing Reference Suggestions", markdown)
        self.assertIn("个人敏感信息 / 医保卡号", markdown)
        self.assertIn("needs_review: `true`", markdown)


if __name__ == "__main__":
    unittest.main()
