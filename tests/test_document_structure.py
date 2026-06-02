from __future__ import annotations

import unittest

from src.io import document_structure
from src.io.document_structure import detect_structure_signal


class DocumentStructureTests(unittest.TestCase):
    def test_detects_appendix_heading(self) -> None:
        signal = detect_structure_signal("附 录 A", line_number=12)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.kind, "appendix_heading")
        self.assertEqual(signal.title, "附录 A")
        self.assertEqual(signal.line_number, 12)
        self.assertGreaterEqual(signal.confidence, 0.9)

    def test_detects_classification_title(self) -> None:
        signal = detect_structure_signal("基础资源分类", line_number=5)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.kind, "classification_title")
        self.assertEqual(signal.title, "基础资源分类")

    def test_detects_table_title(self) -> None:
        signal = detect_structure_signal("表A.1 基础资源分类目录（示例）", line_number=20)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.kind, "table_title")
        self.assertEqual(signal.title, "表A.1 基础资源分类目录（示例）")

    def test_detects_continued_table_title(self) -> None:
        signal = detect_structure_signal("续表A.1 基础资源分类目录", line_number=33)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.kind, "continued_table_title")
        self.assertEqual(signal.title, "续表A.1 基础资源分类目录")

    def test_prioritizes_continued_table_title_over_table_title(self) -> None:
        self.assertTrue(hasattr(document_structure, "detect_structure_signals"))
        signals = document_structure.detect_structure_signals("表A.1（续） 基础资源分类目录")

        self.assertGreaterEqual(len(signals), 2)
        self.assertEqual(signals[0].kind, "continued_table_title")
        self.assertGreater(signals[0].priority, signals[1].priority)

    def test_detects_numbered_main_body_heading_as_not_appendix_body(self) -> None:
        signal = detect_structure_signal("1. 范围", line_number=3)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.kind, "not_appendix_body")
        self.assertEqual(signal.title, "1. 范围")
        self.assertEqual(signal.line_number, 3)

    def test_detects_hierarchy_header(self) -> None:
        signal = detect_structure_signal("类（1 位数字） 项（2 位数字） 目（3 位数字）")

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.kind, "hierarchy_header")
        self.assertEqual(signal.title, "类（1 位数字） 项（2 位数字） 目（3 位数字）")

    def test_ignores_normal_table_row(self) -> None:
        signal = detect_structure_signal("1 服务范围与对象 01 患者")

        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()
