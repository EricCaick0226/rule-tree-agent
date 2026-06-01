from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from src.io.document_parser import chunk_documents, parse_documents


class DocumentParserLineTests(unittest.TestCase):
    def test_txt_chunks_include_line_span(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.txt"
            path.write_text(
                "# 分类标准\n"
                "一级分类 二级分类 推荐分级 分类说明\n"
                "基础资源 服务范围与对象 3级 患者相关资料\n"
                "\n"
                "## 分级说明\n"
                "3级 一般数据3级\n",
                encoding="utf-8",
            )
            chunks = chunk_documents(parse_documents([str(path)]))

        table_chunk = next(chunk for chunk in chunks if "基础资源" in chunk.text)
        self.assertEqual(table_chunk.line_start, 2)
        self.assertEqual(table_chunk.line_end, 3)
        self.assertEqual(table_chunk.source_method, "text")

    def test_appendix_and_table_titles_update_section_context(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "standard.txt"
            path.write_text(
                "6 多重标识符\n"
                "同一资源可以有多个标识符。\n"
                "附 录 A\n"
                "（规范性）\n"
                "基础资源分类\n"
                "基础资源分类见表A.1。\n"
                "表A.1 基础资源分类目录（示例）\n"
                "类（1 位数字） 项（2 位数字） 目（3 位数字）\n"
                "1 服务范围与对象 01 患者\n",
                encoding="utf-8",
            )
            chunks = chunk_documents(parse_documents([str(path)]))

        table_chunk = next(chunk for chunk in chunks if "1 服务范围与对象" in chunk.text)
        self.assertIn("附录 A", table_chunk.section_title)
        self.assertIn("基础资源分类", table_chunk.section_title)
        self.assertIn("表A.1 基础资源分类目录", table_chunk.section_title)
        self.assertNotIn("多重标识符", table_chunk.section_title)

    def test_continued_table_title_updates_section_context(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "standard.txt"
            path.write_text(
                "附 录 A\n"
                "基础资源分类\n"
                "表A.1 基础资源分类目录\n"
                "1 服务范围与对象 01 患者\n"
                "续表A.1 基础资源分类目录\n"
                "2 基础支撑 01 机构\n",
                encoding="utf-8",
            )
            chunks = chunk_documents(parse_documents([str(path)]))

        continued_chunk = next(chunk for chunk in chunks if "2 基础支撑" in chunk.text)
        self.assertIn("附录 A", continued_chunk.section_title)
        self.assertIn("基础资源分类", continued_chunk.section_title)
        self.assertIn("续表A.1 基础资源分类目录", continued_chunk.section_title)
        self.assertNotIn("表A.1 基础资源分类目录 / 表A.1", continued_chunk.section_title)


if __name__ == "__main__":
    unittest.main()
