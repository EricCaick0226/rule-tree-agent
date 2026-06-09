from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.report_table_structure import main, write_table_structure_report


class ReportTableStructureScriptTest(unittest.TestCase):
    def test_write_table_structure_report_outputs_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            source = work_dir / "sample.txt"
            out_dir = work_dir / "report"
            source.write_text(
                "\n".join(
                    [
                        "附录 A",
                        "表 A.1 数据分类分级表",
                        "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别",
                        "基础资源 患者 个人信息 姓名 原始数据 个人 严重危害 一般数据3级",
                    ]
                ),
                encoding="utf-8",
            )

            write_table_structure_report(source, out_dir)

            json_path = out_dir / "table_structure_report.json"
            markdown_path = out_dir / "table_structure_report.md"
            filtered_json_path = out_dir / "table_structure_filtered_report.json"
            filtered_markdown_path = out_dir / "table_structure_filtered_report.md"
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(filtered_json_path.exists())
            self.assertTrue(filtered_markdown_path.exists())

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(data["total_segments"], 1)
            self.assertEqual(data["segmentation_mode"], "all_nonempty_chunks_as_table_candidates")
            filtered_data = json.loads(filtered_json_path.read_text(encoding="utf-8"))
            self.assertEqual(filtered_data["filter_mode"], "reviewable_structure_signals")
            self.assertLessEqual(filtered_data["filtered_segments"], filtered_data["total_segments"])

            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Table Structure Report", markdown)
            self.assertIn("- Segmentation mode: all_nonempty_chunks_as_table_candidates", markdown)
            self.assertIn("classification_grading_table", markdown)
            filtered_markdown = filtered_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Filtered Table Structure Report", filtered_markdown)
            self.assertIn("- Filter mode: reviewable_structure_signals", filtered_markdown)

    def test_main_returns_1_for_missing_txt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stderr = io.StringIO()
            argv = [
                "report_table_structure.py",
                "--txt",
                str(Path(tmp_dir) / "missing.txt"),
                "--out",
                str(Path(tmp_dir) / "report"),
            ]

            with patch.object(sys, "argv", argv), patch("sys.stderr", stderr):
                result = main()

            self.assertEqual(result, 1)
            self.assertIn("error:", stderr.getvalue())

    def test_main_writes_outputs_for_valid_txt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            source = work_dir / "sample.txt"
            out_dir = work_dir / "report"
            source.write_text(
                "\n".join(
                    [
                        "附录 A",
                        "表 A.1 数据分类分级表",
                        "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别",
                        "基础资源 患者 个人信息 姓名 原始数据 个人 严重危害 一般数据3级",
                    ]
                ),
                encoding="utf-8",
            )
            argv = ["report_table_structure.py", "--txt", str(source), "--out", str(out_dir)]
            stdout = io.StringIO()

            with patch.object(sys, "argv", argv), patch("sys.stdout", stdout):
                result = main()

            self.assertEqual(result, 0)
            self.assertTrue((out_dir / "table_structure_report.json").exists())
            self.assertTrue((out_dir / "table_structure_report.md").exists())
            self.assertTrue((out_dir / "table_structure_filtered_report.json").exists())
            self.assertTrue((out_dir / "table_structure_filtered_report.md").exists())


if __name__ == "__main__":
    unittest.main()
