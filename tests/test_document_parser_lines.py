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


if __name__ == "__main__":
    unittest.main()
