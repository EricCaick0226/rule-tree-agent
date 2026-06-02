from __future__ import annotations

import unittest

from src.eval_harness.structure_signals import (
    StructureSignalCase,
    evaluate_structure_signal_cases,
)


class StructureSignalEvalTests(unittest.TestCase):
    def test_scores_signal_detection_cases_and_records_failures(self) -> None:
        cases = [
            StructureSignalCase(
                line="附 录 A",
                expected_kind="appendix_heading",
                expected_title="附录 A",
                line_number=1,
            ),
            StructureSignalCase(
                line="表A.1 基础资源分类目录",
                expected_kind="table_title",
                expected_title="表A.1 基础资源分类目录",
                line_number=2,
            ),
            StructureSignalCase(
                line="1 服务范围与对象 01 患者",
                expected_kind=None,
                expected_title=None,
                line_number=3,
            ),
            StructureSignalCase(
                line="基础资源分类",
                expected_kind="appendix_heading",
                expected_title="附录 B",
                line_number=4,
            ),
            StructureSignalCase(
                line="续表B.1 业务资源分类目录",
                expected_kind=None,
                expected_title=None,
                line_number=5,
            ),
        ]

        report = evaluate_structure_signal_cases(cases)

        self.assertEqual(report["total"], 5)
        self.assertEqual(report["matched_count"], 3)
        self.assertEqual(report["false_positive_count"], 1)
        self.assertEqual(report["false_negative_count"], 0)
        self.assertEqual(report["kind_mismatch_count"], 1)
        self.assertEqual(report["title_mismatch_count"], 1)
        self.assertAlmostEqual(report["accuracy"], 0.6)
        self.assertEqual(report["by_expected_kind"]["appendix_heading"]["expected"], 2)
        self.assertEqual(report["by_expected_kind"]["appendix_heading"]["matched"], 1)
        self.assertEqual(report["failures"][0]["line_number"], 4)
        self.assertEqual(report["failures"][0]["expected_kind"], "appendix_heading")
        self.assertEqual(report["failures"][0]["actual_kind"], "classification_title")
        self.assertEqual(report["failures"][1]["line_number"], 5)
        self.assertEqual(report["failures"][1]["actual_kind"], "continued_table_title")

    def test_empty_case_list_returns_stable_zero_report(self) -> None:
        report = evaluate_structure_signal_cases([])

        self.assertEqual(report["total"], 0)
        self.assertEqual(report["accuracy"], 0.0)
        self.assertEqual(report["failures"], [])


if __name__ == "__main__":
    unittest.main()
