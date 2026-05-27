from __future__ import annotations

import unittest

from scripts.evaluate_rule_table_against_excel import (
    compare_rows,
    normalize_grade,
    normalize_path_level,
)


class RuleTableEvaluationTests(unittest.TestCase):
    def test_normalizes_codes_and_grade_names(self) -> None:
        self.assertEqual(normalize_path_level("001患者信息"), "患者信息")
        self.assertEqual(normalize_path_level("A、基础资源"), "基础资源")
        self.assertEqual(normalize_grade("一般数据3级"), "3级")

    def test_normalize_path_preserves_age_numbers(self) -> None:
        self.assertEqual(normalize_path_level("60周岁以上老年人"), "60周岁以上老年人")

    def test_compare_rows_reports_coverage_and_grade_accuracy(self) -> None:
        generated = [
            {"path_levels": ["基础资源", "患者"], "recommended_grade": "一般数据3级"},
            {"path_levels": ["基础资源", "敏感信息"], "recommended_grade": "一般数据3级"},
        ]
        reference = [
            {"path_levels": ["基础资源", "患者"], "recommended_grade": "3级"},
            {"path_levels": ["基础资源", "敏感信息"], "recommended_grade": "4级"},
            {"path_levels": ["业务资源", "门诊"], "recommended_grade": "2级"},
        ]

        result = compare_rows(generated, reference)

        self.assertEqual(result["generated_rows"], 2)
        self.assertEqual(result["reference_rows"], 3)
        self.assertEqual(result["matched_paths"], 2)
        self.assertEqual(result["missing_paths"], 1)
        self.assertEqual(result["grade_matches"], 1)
        self.assertEqual(result["grade_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
