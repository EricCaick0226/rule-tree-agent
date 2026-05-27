from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()
