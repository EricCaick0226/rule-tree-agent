from __future__ import annotations

import unittest

from src.core.agent_state import AgentState, ClassificationRow, ClassificationSchema


class RowFirstStateTests(unittest.TestCase):
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
