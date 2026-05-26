from __future__ import annotations

import unittest

from src.core.agent_state import AgentState, ClassificationRow
from src.steps.classification_row_normalizer import normalize_classification_rows


class ClassificationRowNormalizerTests(unittest.TestCase):
    def test_dedupes_rows_and_computes_schema_depth(self) -> None:
        rows = [
            ClassificationRow(
                row_id="row_a",
                path_levels=["基础资源", "服务范围与对象", "患者"],
                recommended_grade="3级",
                description="证据不足，无法从当前文档确定",
                description_source="insufficient",
                evidence_quote="基础资源 服务范围与对象 患者 3级",
                support_level="explicit",
                needs_review=True,
                review_reason="当前文档未提供该分类项的说明或范围描述。",
            ),
            ClassificationRow(
                row_id="row_b",
                path_levels=["基础资源", "服务范围与对象", "患者"],
                recommended_grade="3级",
                description="重复行",
                description_source="summarized",
                evidence_quote="基础资源 服务范围与对象 患者 3级",
                support_level="weak",
                needs_review=True,
                review_reason="重复候选。",
            ),
        ]
        state = AgentState(task="test", classification_rows=rows)

        result = normalize_classification_rows(state)

        self.assertEqual(len(result.classification_rows), 1)
        self.assertEqual(result.classification_schema.max_depth, 3)
        self.assertEqual(result.classification_schema.source, "inferred_from_rows")

    def test_strips_blank_path_levels_and_removes_empty_path_rows(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_a",
                    path_levels=[" 基础资源 ", "", "  患者  "],
                    recommended_grade="3级",
                    description="患者分类项",
                    description_source="summarized",
                ),
                ClassificationRow(
                    row_id="row_b",
                    path_levels=[" ", ""],
                    recommended_grade="2级",
                    description="空路径行",
                    description_source="summarized",
                ),
            ],
        )

        result = normalize_classification_rows(state)

        self.assertEqual(len(result.classification_rows), 1)
        self.assertEqual(result.classification_rows[0].path_levels, ["基础资源", "患者"])
        self.assertEqual(result.classification_schema.max_depth, 2)

    def test_empty_or_insufficient_description_is_forced_to_review(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_a",
                    path_levels=["基础资源"],
                    recommended_grade="3级",
                    description="   ",
                    description_source="summarized",
                    needs_review=False,
                    review_reason="",
                ),
                ClassificationRow(
                    row_id="row_b",
                    path_levels=["基础资源", "患者"],
                    recommended_grade="3级",
                    description="已有文字也应被证据不足覆盖",
                    description_source="insufficient",
                    needs_review=False,
                    review_reason="",
                ),
            ],
        )

        result = normalize_classification_rows(state)

        for row in result.classification_rows:
            self.assertEqual(row.description, "证据不足，无法从当前文档确定")
            self.assertEqual(row.description_source, "insufficient")
            self.assertTrue(row.needs_review)
            self.assertTrue(row.review_reason)


if __name__ == "__main__":
    unittest.main()
