import unittest

from src.core.agent_state import DocumentChunk
from src.io.table_segmenter import segment_table_chunks_for_row_extraction


class TableSegmenterTests(unittest.TestCase):
    def _chunk(self, text: str) -> DocumentChunk:
        return DocumentChunk(
            chunk_id="doc_1_chunk_9",
            doc_id="doc_1",
            doc_name="sample.txt",
            section_title="附录B",
            text=text,
            position=9,
            line_start=100,
            line_end=100 + len(text.splitlines()) - 1,
        )

    def _chunk_with_provenance(self, text: str) -> DocumentChunk:
        return DocumentChunk(
            chunk_id="doc_1_chunk_9",
            doc_id="doc_1",
            doc_name="sample.txt",
            section_title="附录B",
            text=text,
            position=9,
            page_number=3,
            line_start=100,
            line_end=100 + len(text.splitlines()) - 1,
            source_method="ocr",
            source_warning="low confidence",
        )

    def test_splits_large_table_chunk_without_dropping_lines(self):
        header = "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"
        lines = [header]
        for index in range(1, 21):
            lines.append(f"001项目{index} 示例{index} 原始数据 个人 严重危害 一般数据3级")
        chunk = self._chunk("\n".join(lines))

        segments = segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={"doc_1_chunk_9": {"block_signal": "table_like"}},
            max_chars=180,
        )

        self.assertGreater(len(segments), 1)
        joined = "\n".join(segment.text for segment in segments)
        self.assertIn("001项目1 示例1 原始数据 个人 严重危害 一般数据3级", joined)
        self.assertIn("001项目20 示例20 原始数据 个人 严重危害 一般数据3级", joined)
        self.assertTrue(all(segment.source_chunk_id == "doc_1_chunk_9" for segment in segments))
        self.assertTrue(all(segment.segment_id.startswith("doc_1_chunk_9_seg_") for segment in segments))

    def test_keeps_small_normal_chunk_as_single_segment(self):
        chunk = self._chunk("普通说明文本")

        segments = segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={},
            max_chars=180,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].segment_id, "doc_1_chunk_9_seg_1")
        self.assertEqual(segments[0].source_chunk_id, "doc_1_chunk_9")
        self.assertEqual(segments[0].text, "普通说明文本")

    def test_repeated_headers_start_new_segment(self):
        header = "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"
        text = "\n".join(
            [
                header,
                "001项目A 示例A 原始数据 个人 严重危害 一般数据3级",
                header,
                "002项目B 示例B 原始数据 个人 特别严重危害 一般数据4级",
            ]
        )
        chunk = self._chunk(text)

        segments = segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={"doc_1_chunk_9": {"block_signal": "table_like"}},
            max_chars=1000,
        )

        self.assertEqual(len(segments), 2)
        self.assertIn("001项目A", segments[0].text)
        self.assertIn("002项目B", segments[1].text)
        self.assertEqual(segments[0].header_text, header)
        self.assertEqual(segments[1].header_text, header)

    def test_preserves_blank_lines_when_reconstructing_segments(self):
        header = "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"
        text = "\n".join(
            [
                header,
                "",
                "001项目A 示例A 原始数据 个人 严重危害 一般数据3级",
                "",
                "002项目B 示例B 原始数据 个人 特别严重危害 一般数据4级",
            ]
        )
        chunk = self._chunk(text)

        segments = segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={"doc_1_chunk_9": {"block_signal": "table_like"}},
            max_chars=95,
        )

        self.assertGreater(len(segments), 1)
        self.assertEqual("\n".join(segment.text for segment in segments), text)

    def test_preserves_source_metadata_and_real_line_spans(self):
        header = "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"
        text = "\n".join(
            [
                header,
                "001项目A 示例A 原始数据 个人 严重危害 一般数据3级",
                "002项目B 示例B 原始数据 个人 特别严重危害 一般数据4级",
            ]
        )
        chunk = self._chunk_with_provenance(text)

        segments = segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={"doc_1_chunk_9": {"block_signal": "table_like"}},
            max_chars=80,
        )

        self.assertEqual(len(segments), 2)
        self.assertTrue(all(segment.source_method == "ocr" for segment in segments))
        self.assertTrue(all(segment.source_warning == "low confidence" for segment in segments))
        self.assertTrue(all(segment.page_number == 3 for segment in segments))
        self.assertEqual(segments[0].line_start, 100)
        self.assertEqual(segments[0].line_end, 101)
        self.assertEqual(segments[1].line_start, 102)
        self.assertEqual(segments[1].line_end, 102)
        self.assertEqual(segments[1].header_text, header)
        self.assertNotIn(header, segments[1].text)

    def test_splits_single_line_that_exceeds_max_chars(self):
        chunk = self._chunk("A" * 25)

        segments = segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={},
            max_chars=10,
        )

        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segment.text for segment in segments), chunk.text)
        self.assertTrue(all(segment.line_start == 100 for segment in segments))
        self.assertTrue(all(segment.line_end == 100 for segment in segments))

    def test_leading_blank_lines_before_header_do_not_create_empty_segment(self):
        header = "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"
        text = "\n".join(
            [
                "",
                "",
                header,
                "001项目A 示例A 原始数据 个人 严重危害 一般数据3级",
            ]
        )
        chunk = self._chunk(text)

        segments = segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={"doc_1_chunk_9": {"block_signal": "table_like"}},
            max_chars=1000,
        )

        self.assertTrue(all(segment.text.strip() for segment in segments))
        self.assertEqual(len(segments), 1)
        self.assertIn("001项目A", segments[0].text)

    def test_small_normal_chunk_with_trailing_newline_uses_source_line_end(self):
        chunk = DocumentChunk(
            chunk_id="doc_1_chunk_9",
            doc_id="doc_1",
            doc_name="sample.txt",
            section_title="说明",
            text="x\n",
            position=9,
            line_start=100,
            line_end=100,
        )

        segments = segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={},
            max_chars=1000,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "x\n")
        self.assertEqual(segments[0].line_start, 100)
        self.assertEqual(segments[0].line_end, 100)


if __name__ == "__main__":
    unittest.main()
