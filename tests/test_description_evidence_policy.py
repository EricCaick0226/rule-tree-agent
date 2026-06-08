from __future__ import annotations

import unittest

from src.core.agent_state import ClassificationRow
from src.io.description_evidence_policy import (
    EMPTY_DATA_RANGE_MARKERS,
    FALLBACK_DESCRIPTION_REASON,
    has_specific_description_evidence,
    is_empty_data_range,
    is_fallback_leaf,
    should_reject_label_only_description,
    should_force_insufficient_description,
)


class DescriptionEvidencePolicyTests(unittest.TestCase):
    def test_empty_data_range_markers_cover_common_placeholders(self) -> None:
        self.assertEqual(EMPTY_DATA_RANGE_MARKERS, {"", "-", "—", "－", "一"})
        for value in ["", "-", "—", "－", "一", "  —  "]:
            self.assertTrue(is_empty_data_range(value))
        self.assertFalse(is_empty_data_range("出生证信息"))

    def test_fallback_leaf_detects_coded_other_categories(self) -> None:
        self.assertTrue(is_fallback_leaf(["1公共卫生", "03妇幼保健", "999其他"]))
        self.assertTrue(is_fallback_leaf(["基础资源", "000其他"]))
        self.assertFalse(is_fallback_leaf(["1公共卫生", "03妇幼保健", "008计划生育技术服务"]))
        self.assertFalse(is_fallback_leaf([]))

    def test_specific_description_evidence_requires_non_empty_data_range(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["1公共卫生", "03妇幼保健", "999其他"],
            data_range_examples=["—"],
        )
        self.assertFalse(has_specific_description_evidence(row))

        row.data_range_examples = ["托幼传染病报告"]
        self.assertTrue(has_specific_description_evidence(row))

    def test_fallback_leaf_without_specific_evidence_is_forced_insufficient(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["1公共卫生", "03妇幼保健", "999其他"],
            data_range_examples=["—"],
        )
        decision = should_force_insufficient_description(row)
        self.assertTrue(decision.force)
        self.assertEqual(decision.reason, FALLBACK_DESCRIPTION_REASON)

    def test_non_fallback_leaf_is_not_forced_insufficient(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["1公共卫生", "03妇幼保健", "008计划生育技术服务"],
            data_range_examples=["—"],
        )
        decision = should_force_insufficient_description(row)
        self.assertFalse(decision.force)
        self.assertEqual(decision.reason, "")

    def test_label_only_description_is_rejected_without_rejecting_explanatory_quote(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["卫生健康数据", "基础资源类数据"],
        )

        label_only = should_reject_label_only_description(
            row,
            proposed_description="基础资源类数据",
            evidence_quote="基础资源类数据",
        )
        self.assertTrue(label_only.force)

        explanatory = should_reject_label_only_description(
            row,
            proposed_description="信息资源中最基础的数据，包括服务范围与对象、法律法规等。",
            evidence_quote="基础资源类数据：信息资源中最基础的数据，包括但不限于服务范围与对象、法律法规。",
        )
        self.assertFalse(explanatory.force)
        self.assertEqual(explanatory.reason, "")


if __name__ == "__main__":
    unittest.main()
