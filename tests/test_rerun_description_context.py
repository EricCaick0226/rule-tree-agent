from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class RerunDescriptionContextTests(unittest.TestCase):
    def test_loads_state_from_existing_rule_table_and_source_txt(self) -> None:
        module = importlib.import_module("scripts.rerun_description_context")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt_path = root / "source.txt"
            txt_path.write_text("001患者信息 患者姓名、生日、性别。", encoding="utf-8")
            rule_table_path = root / "rule_table.json"
            rule_table_path.write_text(
                json.dumps(
                    {
                        "classification_schema": {"max_depth": 3, "source": "inferred_from_rows"},
                        "classification_rows": [
                            {
                                "row_id": "row_1",
                                "path_levels": ["1服务范围与对象", "01患者", "001患者信息"],
                                "recommended_grade": "一般数据3级",
                                "description": "患者姓名、生日、性别",
                                "description_source": "quoted",
                                "data_range_examples": ["患者姓名、生日、性别"],
                                "evidence_refs": [
                                    {
                                        "evidence_id": "ev_1",
                                        "chunk_id": "chunk_1",
                                        "doc_name": "source.txt",
                                        "section_title": "",
                                        "text": "001患者信息 患者姓名、生日、性别。",
                                        "used_for": "classification_row",
                                        "relevance_score": 0.9,
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            state = module.load_state_from_rule_table(txt_path, rule_table_path)

        self.assertEqual(state.documents[0].raw_text, "001患者信息 患者姓名、生日、性别。")
        self.assertEqual(state.classification_schema.max_depth, 3)
        self.assertEqual(state.classification_rows[0].row_id, "row_1")
        self.assertEqual(state.classification_rows[0].evidence_refs[0].evidence_id, "ev_1")

    def test_reruns_description_context_without_creating_upstream_checkpoints(self) -> None:
        module = importlib.import_module("scripts.rerun_description_context")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt_path = root / "source.txt"
            txt_path.write_text("001患者信息 患者姓名、生日、性别。", encoding="utf-8")
            rule_table_path = root / "rule_table.json"
            rule_table_path.write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "row_id": "row_1",
                                "path_levels": ["1服务范围与对象", "01患者", "001患者信息"],
                                "description": "患者姓名、生日、性别",
                                "description_source": "quoted",
                                "data_range_examples": ["患者姓名、生日、性别"],
                                "evidence_quote": "001患者信息 患者姓名、生日、性别。",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            out_dir = root / "rerun_out"

            def fake_generate(_llm_client, rows, batch_size=20):
                self.assertEqual(batch_size, 1)
                self.assertEqual(rows[0]["row_id"], "row_1")
                return (
                    [
                        {
                            "row_id": "row_1",
                            "proposed_description": "用于记录患者基础身份特征的数据。",
                            "description_source": "summarized",
                            "description_evidence_quote": "001患者信息 患者姓名、生日、性别。",
                            "needs_review": True,
                            "review_reason": "基于检索上下文总结生成，需要人工确认。",
                        }
                    ],
                    "raw",
                )

            with patch(
                "src.steps.description_context_kb.generate_description_candidates_batched",
                side_effect=fake_generate,
            ):
                result = module.rerun_description_context(
                    txt_path=txt_path,
                    rule_table_path=rule_table_path,
                    output_dir=out_dir,
                    llm_client=object(),
                    mode="v2",
                    limit=5,
                    batch_size=1,
                )

            exported = json.loads((out_dir / "rule_table.json").read_text(encoding="utf-8"))
            self.assertEqual(result.classification_rows[0].description, "用于记录患者基础身份特征的数据。")
            self.assertEqual(
                exported["classification_rows"][0]["description"],
                "用于记录患者基础身份特征的数据。",
            )
            self.assertTrue((out_dir / "description_context_report.json").exists())
            self.assertFalse((out_dir / "checkpoints").exists())

    def test_rerun_refreshes_validation_issues_after_description_changes(self) -> None:
        module = importlib.import_module("scripts.rerun_description_context")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt_path = root / "source.txt"
            txt_path.write_text(
                "7人力资源\n01人力资源规划 999其他 原始数据 个人 严重危害 一般数据3级",
                encoding="utf-8",
            )
            rule_table_path = root / "rule_table.json"
            rule_table_path.write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "row_id": "row_999",
                                "path_levels": ["7人力资源", "01人力资源规划", "999其他"],
                                "recommended_grade": "一般数据3级",
                                "description": "—",
                                "description_source": "quoted",
                                "description_evidence_quote": "—",
                                "data_range_examples": ["—"],
                                "processing_degree": "原始数据",
                                "impact_object": "个人",
                                "impact_degree": "严重危害",
                                "evidence_quote": "01人力资源规划 999其他 原始数据 个人 严重危害 一般数据3级",
                                "grade_evidence_quote": "原始数据 个人 严重危害 一般数据3级",
                                "support_level": "explicit",
                                "needs_review": True,
                                "evidence_refs": [
                                    {
                                        "evidence_id": "ev_1",
                                        "chunk_id": "chunk_1",
                                        "doc_name": "source.txt",
                                        "section_title": "",
                                        "text": "01人力资源规划 999其他 原始数据 个人 严重危害 一般数据3级",
                                        "used_for": "classification_row",
                                        "relevance_score": 0.9,
                                    }
                                ],
                            }
                        ],
                        "validation_issues": [
                            {
                                "issue_id": "issue_old",
                                "issue_type": "hardcoded_or_ungrounded_content",
                                "severity": "high",
                                "target": "7人力资源 / 01人力资源规划 / 999其他",
                                "problem": "description_evidence_quote 未出现在引用证据或原文中。",
                                "suggested_action": "改为引用文档原文。",
                                "status": "open",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            out_dir = root / "rerun_out"

            def fake_generate(_llm_client, _rows, batch_size=20):
                return (
                    [
                        {
                            "row_id": "row_999",
                            "proposed_description": "未明确归类的人力资源规划相关数据。",
                            "description_source": "summarized",
                            "description_evidence_quote": "—",
                            "needs_review": True,
                            "review_reason": "基于检索上下文总结生成，需要人工确认。",
                        }
                    ],
                    "raw",
                )

            with patch(
                "src.steps.description_context_kb.generate_description_candidates_batched",
                side_effect=fake_generate,
            ):
                module.rerun_description_context(
                    txt_path=txt_path,
                    rule_table_path=rule_table_path,
                    output_dir=out_dir,
                    llm_client=object(),
                    mode="v2",
                    limit=5,
                    batch_size=1,
                )

            exported = json.loads((out_dir / "rule_table.json").read_text(encoding="utf-8"))

        problems = [issue["problem"] for issue in exported.get("validation_issues") or []]
        self.assertNotIn("description_evidence_quote 未出现在引用证据或原文中。", problems)
        self.assertEqual(exported["classification_rows"][0]["description_source"], "insufficient")


if __name__ == "__main__":
    unittest.main()
