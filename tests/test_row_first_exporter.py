from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.core.agent_state import AgentState, ClassificationRow, ClassificationSchema
from src.output.exporter import export_outputs


class RowFirstExporterTests(unittest.TestCase):
    def test_exports_dynamic_depth_table_and_tree(self) -> None:
        state = AgentState(
            task="test",
            classification_schema=ClassificationSchema(max_depth=3, source="inferred_from_rows"),
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["基础资源", "服务范围与对象", "患者"],
                    recommended_grade="3级",
                    description="证据不足，无法从当前文档确定",
                    description_source="insufficient",
                    evidence_quote="基础资源 服务范围与对象 患者 3级",
                    support_level="explicit",
                    needs_review=True,
                    review_reason="当前文档未提供该分类项的说明或范围描述。",
                )
            ],
        )

        with TemporaryDirectory() as tmp:
            result = export_outputs(state, tmp)
            table_md = Path(result.output_paths["rule_table_md"]).read_text(encoding="utf-8")
            table_json = json.loads(
                Path(result.output_paths["rule_table_json"]).read_text(encoding="utf-8")
            )
            tree_md = Path(result.output_paths["rule_tree_md"]).read_text(encoding="utf-8")
            tree_json = json.loads(
                Path(result.output_paths["rule_tree_json"]).read_text(encoding="utf-8")
            )

        self.assertIn("一级分类", table_md)
        self.assertIn("三级分类", table_md)
        self.assertIn("证据不足，无法从当前文档确定", table_md)
        self.assertEqual(table_json["classification_schema"]["max_depth"], 3)
        self.assertEqual(table_json["classification_rows"][0]["recommended_grade"], "3级")
        self.assertIn("# Candidate Rule Tree", tree_md)
        self.assertEqual(tree_json["classification_rows"][0]["path_levels"][-1], "患者")

    def test_infers_depth_from_rows_when_schema_missing(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["一级", "二级", "三级", "四级"],
                    recommended_grade="2级",
                )
            ],
        )

        with TemporaryDirectory() as tmp:
            result = export_outputs(state, tmp)
            table_md = Path(result.output_paths["rule_table_md"]).read_text(encoding="utf-8")

        self.assertIn("一级分类", table_md)
        self.assertIn("四级分类", table_md)
        self.assertIn("| 一级 | 二级 | 三级 | 四级 |", table_md)

    def test_uses_row_depth_when_schema_depth_is_too_shallow(self) -> None:
        state = AgentState(
            task="test",
            classification_schema=ClassificationSchema(max_depth=1, source="inferred_from_header"),
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["基础资源", "服务范围与对象", "患者"],
                    recommended_grade="3级",
                )
            ],
        )

        with TemporaryDirectory() as tmp:
            result = export_outputs(state, tmp)
            table_md = Path(result.output_paths["rule_table_md"]).read_text(encoding="utf-8")

        self.assertIn("三级分类", table_md)
        self.assertIn("| 基础资源 | 服务范围与对象 | 患者 |", table_md)

    def test_markdown_escapes_pipes_and_newlines_in_cells(self) -> None:
        state = AgentState(
            task="test",
            classification_schema=ClassificationSchema(max_depth=1, source="inferred_from_rows"),
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["基础|资源"],
                    recommended_grade="3级",
                    description="第一行\n第二行",
                    support_level="explicit|quoted",
                    review_reason="原因一\n原因二",
                )
            ],
        )

        with TemporaryDirectory() as tmp:
            result = export_outputs(state, tmp)
            table_md = Path(result.output_paths["rule_table_md"]).read_text(encoding="utf-8")

        self.assertIn("基础\\|资源", table_md)
        self.assertIn("第一行<br>第二行", table_md)
        self.assertIn("explicit\\|quoted", table_md)
        self.assertIn("原因一<br>原因二", table_md)

    def test_empty_rows_table_reports_insufficient_evidence(self) -> None:
        state = AgentState(task="test", classification_schema=ClassificationSchema(max_depth=3))

        with TemporaryDirectory() as tmp:
            result = export_outputs(state, tmp)
            table_md = Path(result.output_paths["rule_table_md"]).read_text(encoding="utf-8")

        self.assertIn("# Candidate Classification Table", table_md)
        self.assertIn("证据不足，无法从当前文档确定分类分级明细。", table_md)
        self.assertNotIn("| 一级分类 |", table_md)

    def test_exports_structured_row_fields(self) -> None:
        state = AgentState(task="test")
        state.classification_rows = [
            ClassificationRow(
                row_id="r1",
                path_levels=["A", "B"],
                recommended_grade="一般数据3级",
                description="姓名",
                description_source="quoted",
                data_range_examples=["姓名"],
                processing_degree="原始数据",
                impact_object="个人",
                impact_degree="严重危害",
                grade_evidence_quote="原始数据 个人 严重危害 一般数据3级",
                evidence_quote="姓名 原始数据 个人 严重危害 一般数据3级",
                support_level="explicit",
                confidence=0.9,
                needs_review=False,
            )
        ]

        with TemporaryDirectory() as tmp:
            export_outputs(state, tmp)

            table = json.loads(Path(tmp, "rule_table.json").read_text(encoding="utf-8"))
            row = table["classification_rows"][0]
            self.assertEqual(row["data_range_examples"], ["姓名"])
            self.assertEqual(row["processing_degree"], "原始数据")
            self.assertEqual(row["impact_object"], "个人")
            self.assertEqual(row["impact_degree"], "严重危害")
            self.assertEqual(row["grade_evidence_quote"], "原始数据 个人 严重危害 一般数据3级")
            markdown = Path(tmp, "rule_table.md").read_text(encoding="utf-8")
            self.assertIn("数据范围及示例", markdown)
            self.assertIn("影响程度", markdown)

    def test_default_export_profile_keeps_only_deliverable_reports(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="current",
                    path_levels=["基础资源", "设备资源", "硬件设备"],
                )
            ],
            reference_candidate_rows=[
                ClassificationRow(
                    row_id="candidate",
                    path_levels=["基础资源", "设备资源", "软件设备"],
                    description="软件设备相关信息。",
                    description_source="reference_library",
                    row_source="reference_library",
                    content_source="reference_library",
                    inclusion_status="review_candidate",
                    evidence_status="reference_only",
                )
            ],
        )

        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {}, clear=True):
                result = export_outputs(state, tmp)
            output_dir = Path(tmp)

            self.assertTrue((output_dir / "rule_table.json").exists())
            self.assertTrue((output_dir / "rule_table.md").exists())
            self.assertTrue((output_dir / "rule_tree.json").exists())
            self.assertTrue((output_dir / "rule_tree.md").exists())
            self.assertTrue((output_dir / "run_quality.json").exists())
            self.assertTrue((output_dir / "reference_candidates.json").exists())
            self.assertTrue((output_dir / "README.md").exists())
            self.assertFalse((output_dir / "review_report.md").exists())
            self.assertFalse((output_dir / "run_quality.md").exists())
            self.assertFalse((output_dir / "reference_candidates.md").exists())

        self.assertIn("rule_table_json", result.output_paths)
        self.assertIn("reference_candidates_json", result.output_paths)
        self.assertNotIn("review_report_md", result.output_paths)
        self.assertNotIn("run_quality_md", result.output_paths)
        self.assertNotIn("reference_candidates_md", result.output_paths)

    def test_default_export_profile_removes_stale_audit_reports(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="current",
                    path_levels=["基础资源", "设备资源", "硬件设备"],
                )
            ],
        )

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for name in [
                "review_report.md",
                "run_quality.md",
                "reference_candidates.md",
                "reference_candidates.json",
            ]:
                (output_dir / name).write_text("stale", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                export_outputs(state, tmp)

            self.assertFalse((output_dir / "review_report.md").exists())
            self.assertFalse((output_dir / "run_quality.md").exists())
            self.assertFalse((output_dir / "reference_candidates.md").exists())
            self.assertFalse((output_dir / "reference_candidates.json").exists())

    def test_run_quality_includes_accuracy_boundary_counters_without_extra_reports(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="current",
                    path_levels=["基础资源", "设备资源", "硬件设备"],
                    row_role="classification_detail",
                    reference_maturity="curated",
                    reference_prefilled_fields=["description"],
                ),
                ClassificationRow(
                    row_id="draft",
                    path_levels=["基础资源", "设备资源", "软件设备"],
                    row_role="classification_detail",
                    reference_maturity="",
                ),
            ],
        )

        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {}, clear=True):
                result = export_outputs(state, tmp)
            output_dir = Path(tmp)
            quality = json.loads(Path(result.output_paths["run_quality_json"]).read_text(encoding="utf-8"))

            self.assertEqual(quality["metrics"]["row_roles"], {"classification_detail": 2})
            self.assertEqual(quality["metrics"]["reference_maturity"], {"curated": 1, "none": 1})
            self.assertTrue((output_dir / "rule_table.json").exists())
            self.assertTrue((output_dir / "run_quality.json").exists())
            self.assertFalse((output_dir / "run_quality.md").exists())
            self.assertFalse((output_dir / "review_report.md").exists())

    def test_review_report_includes_structure_quality_metrics(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="r1",
                    path_levels=["3.2 电子病历数据库 3.2.2 临床诊疗"],
                ),
                ClassificationRow(
                    row_id="r2",
                    path_levels=["2", "2", "1", "临床服务"],
                ),
                ClassificationRow(
                    row_id="r3",
                    path_levels=["2.5 药品供应", "2.5.7 供应管理"],
                ),
            ],
        )

        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"EXPORT_PROFILE": "audit"}):
                result = export_outputs(state, tmp)
            report = Path(result.output_paths["review_report_md"]).read_text(encoding="utf-8")

        self.assertIn("## Structure Quality Notes", report)
        self.assertIn("- Classification rows: 3", report)
        self.assertIn("- Path levels containing multiple hierarchical codes: 1", report)
        self.assertIn("- Rows with numeric-only path levels: 1", report)
        self.assertIn("- Numeric-only path levels: 3", report)
        self.assertIn(
            "- Review rows where one path level contains multiple hierarchical codes.",
            report,
        )

    def test_exports_reference_candidates_separately_from_rule_table(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="current",
                    path_levels=["基础资源", "设备资源", "硬件设备"],
                )
            ],
            reference_candidate_rows=[
                ClassificationRow(
                    row_id="candidate",
                    path_levels=["基础资源", "设备资源", "软件设备"],
                    description="软件设备相关信息。",
                    description_source="reference_library",
                    row_source="reference_library",
                    content_source="reference_library",
                    inclusion_status="review_candidate",
                    evidence_status="reference_only",
                    needs_review=True,
                )
            ],
        )

        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"EXPORT_PROFILE": "audit"}):
                result = export_outputs(state, tmp)
            table = json.loads(Path(result.output_paths["rule_table_json"]).read_text(encoding="utf-8"))
            candidates = json.loads(
                Path(result.output_paths["reference_candidates_json"]).read_text(encoding="utf-8")
            )
            candidates_md = Path(result.output_paths["reference_candidates_md"]).read_text(
                encoding="utf-8"
            )
            report = Path(result.output_paths["review_report_md"]).read_text(encoding="utf-8")

        self.assertEqual(len(table["classification_rows"]), 1)
        self.assertEqual(len(candidates["reference_candidate_rows"]), 1)
        self.assertEqual(
            candidates["reference_candidate_rows"][0]["path_levels"],
            ["基础资源", "设备资源", "软件设备"],
        )
        self.assertIn("软件设备", candidates_md)
        self.assertIn("- Reference candidate rows: 1", report)

    def test_exports_degraded_run_quality_when_thresholds_are_missed(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="quoted",
                    path_levels=["基础资源", "患者"],
                    description="患者信息说明。",
                    description_source="quoted",
                ),
                ClassificationRow(
                    row_id="weak",
                    path_levels=["基础资源", "管理者"],
                    description_source="insufficient",
                ),
            ],
        )

        with TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ",
                {
                    "EXPORT_PROFILE": "audit",
                    "RUN_QUALITY_MIN_ROWS": "3",
                    "RUN_QUALITY_MIN_QUOTED_DESCRIPTIONS": "2",
                    "RUN_QUALITY_MAX_INSUFFICIENT_DESCRIPTIONS": "0",
                    "RUN_QUALITY_MIN_REFERENCE_PREFILLED_ROWS": "1",
                },
            ):
                result = export_outputs(state, tmp)
            quality = json.loads(
                Path(result.output_paths["run_quality_json"]).read_text(encoding="utf-8")
            )
            quality_md = Path(result.output_paths["run_quality_md"]).read_text(encoding="utf-8")
            report = Path(result.output_paths["review_report_md"]).read_text(encoding="utf-8")

        self.assertEqual(quality["status"], "degraded")
        self.assertEqual(quality["metrics"]["classification_rows"], 2)
        self.assertEqual(quality["metrics"]["description_sources"]["quoted"], 1)
        self.assertEqual(quality["metrics"]["description_sources"]["insufficient"], 1)
        self.assertEqual(quality["metrics"]["reference_prefilled_rows"], 0)
        self.assertEqual(
            [reason["code"] for reason in quality["reasons"]],
            [
                "classification_rows_below_min",
                "quoted_descriptions_below_min",
                "insufficient_descriptions_above_max",
                "reference_prefilled_rows_below_min",
            ],
        )
        self.assertIn("# Run Quality", quality_md)
        self.assertIn("- Status: degraded", quality_md)
        self.assertIn("- Run quality: degraded", report)


if __name__ == "__main__":
    unittest.main()
