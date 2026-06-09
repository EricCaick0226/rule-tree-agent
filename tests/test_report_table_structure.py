from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_table_structure import write_table_structure_report


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
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(data["total_segments"], 1)

            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Table Structure Report", markdown)
            self.assertIn("classification_grading_table", markdown)


if __name__ == "__main__":
    unittest.main()
