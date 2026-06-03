import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.description_context_poc import build_description_context_report


class DescriptionContextPOCTests(unittest.TestCase):
    def test_script_writes_weak_description_context_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            txt_path = tmp_path / "source.txt"
            rule_table_path = tmp_path / "rule_table.json"
            out_path = tmp_path / "description_context_poc.json"

            txt_path.write_text(
                "\n".join(
                    [
                        "业务资源",
                        "公共卫生包括免疫规划监测、疾病监测等数据。",
                        "免疫规划监测涉及疫苗接种记录。",
                    ]
                ),
                encoding="utf-8",
            )
            rule_table_path.write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "row_id": "row_0",
                                "path_levels": ["业务资源"],
                                "description": "业务资源",
                                "data_range_examples": [],
                                "recommended_grade": None,
                            },
                            {
                                "row_id": "row_1",
                                "path_levels": ["业务资源", "公共卫生", "免疫规划监测"],
                                "description": "免疫规划监测",
                                "data_range_examples": ["疫苗接种记录"],
                                "recommended_grade": "3级",
                            },
                            {
                                "row_id": "row_2",
                                "path_levels": ["业务资源", "公共卫生", "疾病监测"],
                                "description": "公共卫生下疾病监测相关的数据分类项。",
                                "data_range_examples": [],
                                "recommended_grade": "2级",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/description_context_poc.py",
                    "--txt",
                    str(txt_path),
                    "--rule-table",
                    str(rule_table_path),
                    "--out",
                    str(out_path),
                    "--limit",
                    "1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["sampled_row_count"], 1)
            self.assertEqual(report["rows"][0]["row_id"], "row_1")
            self.assertIn("description_equals_leaf", report["rows"][0]["description_quality_flags"])
            self.assertTrue(report["rows"][0]["retrieved_contexts"])

    def test_report_can_attach_generated_description_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            txt_path = tmp_path / "source.txt"
            rule_table_path = tmp_path / "rule_table.json"
            txt_path.write_text("公共卫生包括免疫规划监测、疾病监测等数据。", encoding="utf-8")
            rule_table_path.write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "row_id": "row_1",
                                "path_levels": ["业务资源", "公共卫生", "免疫规划监测"],
                                "description": "免疫规划监测",
                                "data_range_examples": ["疫苗接种记录"],
                                "recommended_grade": "3级",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def fake_generate(_llm_client, rows, batch_size=20):
                self.assertEqual(batch_size, 20)
                self.assertEqual(rows[0]["row_id"], "row_1")
                return (
                    [
                        {
                            "row_id": "row_1",
                            "proposed_description": "公共卫生下用于免疫规划监测相关业务的数据分类项。",
                            "description_source": "summarized",
                            "description_evidence_quote": "公共卫生包括免疫规划监测、疾病监测等数据。",
                            "needs_review": True,
                            "review_reason": "基于检索上下文总结生成，需要人工确认。",
                        }
                    ],
                    "raw",
                )

            with patch("scripts.description_context_poc.generate_description_candidates_batched", side_effect=fake_generate):
                report = build_description_context_report(
                    txt_path=txt_path,
                    rule_table_path=rule_table_path,
                    limit=1,
                    generate=True,
                    llm_client=object(),
                )

            self.assertEqual(report["generation"]["status"], "success")
            self.assertEqual(report["rows"][0]["generated_description"]["description_source"], "summarized")
            self.assertIn("公共卫生下", report["rows"][0]["generated_description"]["proposed_description"])

    def test_report_generates_description_candidates_in_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            txt_path = tmp_path / "source.txt"
            rule_table_path = tmp_path / "rule_table.json"
            txt_path.write_text("公共卫生包括免疫规划监测、疾病监测、妇幼保健等数据。", encoding="utf-8")
            rule_table_path.write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "row_id": f"row_{index}",
                                "path_levels": ["业务资源", "公共卫生", f"项目{index}"],
                                "description": f"项目{index}",
                                "data_range_examples": [f"项目{index}记录"],
                                "recommended_grade": "3级",
                            }
                            for index in range(5)
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            seen_batch_sizes: list[int] = []

            def fake_generate(_llm_client, rows, batch_size=20):
                seen_batch_sizes.append(batch_size)
                return (
                    [
                        {
                            "row_id": row["row_id"],
                            "proposed_description": f"{row['path']}相关业务数据。",
                            "description_source": "summarized",
                            "description_evidence_quote": row["retrieved_contexts"][0]["text"],
                            "needs_review": True,
                            "review_reason": "基于检索上下文总结生成，需要人工确认。",
                        }
                        for row in rows
                    ],
                    "raw",
                )

            with patch("scripts.description_context_poc.generate_description_candidates_batched", side_effect=fake_generate):
                report = build_description_context_report(
                    txt_path=txt_path,
                    rule_table_path=rule_table_path,
                    limit=5,
                    generate=True,
                    llm_client=object(),
                    generation_batch_size=2,
                )

            self.assertEqual(seen_batch_sizes, [2])
            self.assertEqual(report["generation"]["status"], "success")
            self.assertEqual(report["generation"]["candidate_count"], 5)
            self.assertEqual(report["generation"]["batch_size"], 2)
            self.assertEqual(report["generation"]["batch_count"], 3)
            self.assertEqual(len([row for row in report["rows"] if row.get("generated_description")]), 5)

    def test_report_records_generation_failure_without_dropping_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            txt_path = tmp_path / "source.txt"
            rule_table_path = tmp_path / "rule_table.json"
            txt_path.write_text("公共卫生包括免疫规划监测、疾病监测等数据。", encoding="utf-8")
            rule_table_path.write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "row_id": "row_1",
                                "path_levels": ["业务资源", "公共卫生", "免疫规划监测"],
                                "description": "免疫规划监测",
                                "data_range_examples": ["疫苗接种记录"],
                                "recommended_grade": "3级",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch(
                "scripts.description_context_poc.generate_description_candidates_batched",
                side_effect=RuntimeError("LLM HTTP 401"),
            ):
                report = build_description_context_report(
                    txt_path=txt_path,
                    rule_table_path=rule_table_path,
                    limit=1,
                    generate=True,
                    llm_client=object(),
                )

            self.assertEqual(report["generation"]["status"], "failed")
            self.assertIn("LLM HTTP 401", report["generation"]["error"])
            self.assertTrue(report["rows"][0]["retrieved_contexts"])


if __name__ == "__main__":
    unittest.main()
