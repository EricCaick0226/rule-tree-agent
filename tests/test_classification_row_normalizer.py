from __future__ import annotations

import unittest

from src.core.agent_state import AgentState, ClassificationRow, EvidenceRef
from src.steps.classification_row_normalizer import normalize_classification_rows


class ClassificationRowNormalizerTests(unittest.TestCase):
    def _ref(self) -> EvidenceRef:
        return EvidenceRef(
            evidence_id="ev_1",
            chunk_id="chunk_1",
            doc_name="test.pdf",
            section_title="分类表",
            text="基础资源 服务范围与对象 患者 3级",
            used_for="classification_row",
            relevance_score=0.9,
        )

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

    def test_duplicate_prefers_refs_and_explicit_support_over_quoted_no_refs(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_a",
                    path_levels=["基础资源", "患者"],
                    recommended_grade="3级",
                    description="患者",
                    description_source="quoted",
                    evidence_refs=[],
                    support_level="weak",
                    confidence=0.95,
                ),
                ClassificationRow(
                    row_id="row_b",
                    path_levels=["基础资源", "患者"],
                    recommended_grade="3级",
                    description="患者分类项",
                    description_source="summarized",
                    evidence_refs=[self._ref()],
                    support_level="explicit",
                    confidence=0.7,
                ),
            ],
        )

        result = normalize_classification_rows(state)

        self.assertEqual(len(result.classification_rows), 1)
        row = result.classification_rows[0]
        self.assertEqual(row.description, "患者分类项")
        self.assertEqual(row.description_source, "summarized")
        self.assertEqual(row.support_level, "explicit")
        self.assertEqual(len(row.evidence_refs), 1)

    def test_duplicate_grade_conflict_appends_review_reason(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_a",
                    path_levels=["基础资源", "患者"],
                    recommended_grade="2级",
                    description="患者分类项",
                    description_source="summarized",
                    evidence_refs=[self._ref()],
                    support_level="explicit",
                    needs_review=False,
                    review_reason="",
                ),
                ClassificationRow(
                    row_id="row_b",
                    path_levels=["基础资源", "患者"],
                    recommended_grade="3级",
                    description="患者分类项补充",
                    description_source="summarized",
                    evidence_refs=[self._ref()],
                    support_level="explicit",
                    confidence=0.9,
                    needs_review=True,
                    review_reason="已有人工复核原因。",
                ),
            ],
        )

        result = normalize_classification_rows(state)

        row = result.classification_rows[0]
        self.assertTrue(row.needs_review)
        self.assertIn("已有人工复核原因", row.review_reason)
        self.assertRegex(row.review_reason, "不同推荐分级|分级候选")


if __name__ == "__main__":
    unittest.main()
