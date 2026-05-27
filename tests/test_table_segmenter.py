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


if __name__ == "__main__":
    unittest.main()
