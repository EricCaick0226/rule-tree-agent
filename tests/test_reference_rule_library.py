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

    def test_load_reference_library_skips_auxiliary_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "wst787_2021" / "metadata.json",
                {
                    "name": "WST 787-2021",
                    "source_type": "national_standard",
                },
            )
            write_json(
                root / "wst787_2021" / "rule_table.json",
                {
                    "classification_rows": [
                        {
                            "row_id": "ref_1",
                            "path_levels": ["患者"],
                        }
                    ]
                },
            )
            write_json(
                root / "data_elements" / "wst363" / "part_02.json",
                {"elements": []},
            )

            references, warnings = load_reference_library(root)

        self.assertEqual(warnings, [])
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].name, "WST 787-2021")

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
        self.assertIn("## Strong Matches", markdown)
        self.assertIn("个人信息 / 身份证号", markdown)
        self.assertIn("## Missing Candidates", markdown)
        self.assertIn("个人敏感信息 / 医保卡号", markdown)
        self.assertIn("needs_review: `true`", markdown)

    def test_report_adds_match_and_missing_tiers_without_removing_raw_fields(self) -> None:
        current_rows = [
            {
                "row_id": "cur_strong",
                "path_levels": ["单位法人"],
                "description": "单位法人。",
                "description_source": "quoted",
                "data_range_examples": ["单位法人"],
            },
            {
                "row_id": "cur_broad_1",
                "path_levels": ["医疗服务(基层)", "门诊服务"],
                "description": "证据不足，无法从当前文档确定",
                "description_source": "insufficient",
                "data_range_examples": ["门诊服务"],
            },
            {
                "row_id": "cur_broad_2",
                "path_levels": ["医疗服务(基层)", "住院服务"],
                "description": "证据不足，无法从当前文档确定",
                "description_source": "insufficient",
                "data_range_examples": ["住院服务"],
            },
            {
                "row_id": "cur_broad_3",
                "path_levels": ["医疗服务(基层)", "体检服务"],
                "description": "证据不足，无法从当前文档确定",
                "description_source": "insufficient",
                "data_range_examples": ["体检服务"],
            },
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
                            "row_id": "ref_strong",
                            "path_levels": ["单位法人"],
                            "description": "单位法人。",
                            "description_source": "quoted",
                            "data_range_examples": ["单位法人"],
                        },
                        {
                            "row_id": "ref_broad",
                            "path_levels": ["医疗服务(基层)"],
                            "description": "证据不足，无法从当前文档确定",
                            "description_source": "insufficient",
                            "data_range_examples": ["门诊服务", "住院服务", "体检服务"],
                        },
                        {
                            "row_id": "ref_missing_good",
                            "path_levels": ["药品供应", "供应管理"],
                            "description": "供应管理。",
                            "description_source": "quoted",
                            "data_range_examples": ["供应管理"],
                        },
                        {
                            "row_id": "ref_missing_bad",
                            "path_levels": ["3.5 16 职业健康管理"],
                            "description": "证据不足，无法从当前文档确定",
                            "description_source": "insufficient",
                            "data_range_examples": [],
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
            top_k=1,
            min_score=0.2,
        )

        self.assertIn("matched_current_rows", report)
        self.assertIn("missing_reference_suggestions", report)
        self.assertEqual(len(report["strong_matches"]), 1)
        self.assertEqual(report["strong_matches"][0]["current_row_id"], "cur_strong")
        self.assertEqual(len(report["broad_matches"]), 3)
        self.assertEqual(
            {item["current_row_id"] for item in report["broad_matches"]},
            {"cur_broad_1", "cur_broad_2", "cur_broad_3"},
        )
        self.assertEqual(len(report["missing_candidates"]), 1)
        self.assertEqual(report["missing_candidates"][0]["reference_row_id"], "ref_missing_good")
        self.assertEqual(len(report["low_quality_reference_rows"]), 1)
        self.assertEqual(report["low_quality_reference_rows"][0]["reference_row_id"], "ref_missing_bad")

    def test_markdown_renders_tiered_sections(self) -> None:
        report = {
            "current": "outputs_current/rule_table.json",
            "references": [],
            "warnings": [],
            "matched_current_rows": [],
            "missing_reference_suggestions": [],
            "strong_matches": [
                {
                    "current_row_id": "cur_1",
                    "current_path": ["单位法人"],
                    "current_description_source": "quoted",
                    "matches": [
                        {
                            "reference_name": "WST 787-2021",
                            "reference_type": "national_standard",
                            "reference_row_id": "ref_1",
                            "reference_path": ["单位法人"],
                            "score": 1.0,
                            "shared_terms": ["单位法人"],
                        }
                    ],
                }
            ],
            "broad_matches": [
                {
                    "current_row_id": "cur_2",
                    "current_path": ["医疗服务(基层)", "门诊服务"],
                    "current_description_source": "insufficient",
                    "matches": [
                        {
                            "reference_name": "WST 787-2021",
                            "reference_type": "national_standard",
                            "reference_row_id": "ref_2",
                            "reference_path": ["医疗服务(基层)"],
                            "score": 0.8,
                            "shared_terms": ["医疗服务(基层)"],
                        }
                    ],
                }
            ],
            "missing_candidates": [
                {
                    "reference_name": "WST 787-2021",
                    "reference_type": "national_standard",
                    "reference_row_id": "ref_3",
                    "reference_path": ["药品供应", "供应管理"],
                    "reference_description": "供应管理。",
                    "reference_grade": None,
                    "needs_review": True,
                    "review_reason": REVIEW_ONLY_REASON,
                }
            ],
            "low_quality_reference_rows": [
                {
                    "reference_name": "WST 787-2021",
                    "reference_type": "national_standard",
                    "reference_row_id": "ref_4",
                    "reference_path": ["3.5 16 职业健康管理"],
                    "reference_description": "证据不足，无法从当前文档确定",
                    "reference_grade": None,
                    "needs_review": True,
                    "review_reason": REVIEW_ONLY_REASON,
                }
            ],
        }

        markdown = render_reference_suggestions_markdown(report)

        self.assertIn("## Strong Matches", markdown)
        self.assertIn("## Broad Matches", markdown)
        self.assertIn("## Missing Candidates", markdown)
        self.assertIn("## Low Quality Reference Rows", markdown)
        self.assertIn("单位法人", markdown)
        self.assertIn("药品供应 / 供应管理", markdown)
        self.assertIn("3.5 16 职业健康管理", markdown)


if __name__ == "__main__":
    unittest.main()
