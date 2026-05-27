from __future__ import annotations

import unittest

from src.core.agent_state import AgentState, ClassificationRow, ClassificationSchema


class RowFirstStateTests(unittest.TestCase):
    def test_classification_row_holds_structured_table_fields(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["资源", "项目"],
            recommended_grade="一般数据3级",
            data_range_examples=["姓名", "联系方式"],
            processing_degree="原始数据",
            impact_object="个人",
            impact_degree="严重危害",
            grade_evidence_quote="原始数据 个人 严重危害 一般数据3级",
        )

        self.assertEqual(row.data_range_examples, ["姓名", "联系方式"])
        self.assertEqual(row.processing_degree, "原始数据")
        self.assertEqual(row.impact_object, "个人")
        self.assertEqual(row.impact_degree, "严重危害")
        self.assertEqual(row.grade_evidence_quote, "原始数据 个人 严重危害 一般数据3级")

    def test_state_holds_classification_rows_and_schema(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="3级",
            description="证据不足，无法从当前文档确定",
            description_source="insufficient",
            description_evidence_quote="",
            evidence_quote="基础资源 服务范围与对象 患者 3级",
            evidence_refs=[],
            support_level="explicit",
            needs_review=True,
            review_reason="当前文档未提供该分类项的说明或范围描述。",
        )
        schema = ClassificationSchema(
            max_depth=3,
            source="inferred_from_rows",
            evidence_quote="基础资源 服务范围与对象 患者",
            needs_review=True,
            review_reason="未找到明确层级表头，按抽取出的路径最大深度推断。",
        )
        state = AgentState(task="test")
        state.classification_rows = [row]
        state.classification_schema = schema

        self.assertEqual(state.classification_rows[0].path_levels[-1], "患者")
        self.assertEqual(state.classification_schema.max_depth, 3)


if __name__ == "__main__":
    unittest.main()
