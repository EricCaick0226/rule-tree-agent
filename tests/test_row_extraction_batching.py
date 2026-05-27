from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.agent_state import AgentState, DocumentChunk
from src.io.table_segmenter import TableSegment
from src.steps.classification_row_extractor import (
    _build_segment_batches,
    _load_checkpoint_records,
    _segment_signature,
    extract_classification_rows_with_llm,
)


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeRowLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens=None, temperature=None, disable_thinking=False):
        self.calls += 1
        content = messages[-1]["content"]
        payload = json.loads(content)["input_payload"]
        segment = payload["table_segments"][0]
        name = f"项目{self.calls}"
        return FakeResponse(
            json.dumps(
                {
                    "classification_rows": [
                        {
                            "path_levels": ["资源", "类别", name],
                            "recommended_grade": "一般数据3级",
                            "description": f"{name}说明",
                            "description_source": "quoted",
                            "description_evidence_quote": f"{name}说明",
                            "data_range_examples": [f"{name}说明"],
                            "processing_degree": "原始数据",
                            "impact_object": "个人",
                            "impact_degree": "严重危害",
                            "grade_evidence_quote": "原始数据 个人 严重危害 一般数据3级",
                            "evidence_quote": segment["text"],
                            "evidence_chunk_ids": [segment["source_chunk_id"]],
                            "support_level": "explicit",
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                            "status": "evidence_supported",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )


class RowExtractionBatchingTests(unittest.TestCase):
    def _state(self) -> AgentState:
        lines = ["类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"]
        for index in range(1, 12):
            lines.append(f"001项目{index} 项目{index}说明 原始数据 个人 严重危害 一般数据3级")
        chunk = DocumentChunk(
            chunk_id="doc_1_chunk_1",
            doc_id="doc_1",
            doc_name="sample.txt",
            section_title="附录B",
            text="\n".join(lines),
            position=1,
            line_start=1,
            line_end=len(lines),
        )
        state = AgentState(task="test")
        state.chunks = [chunk]
        state.block_signals = {"doc_1_chunk_1": {"block_signal": "table_like"}}
        return state

    def test_row_extraction_batches_table_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state()
            llm = FakeRowLLM()

            with patch.dict("os.environ", {"ROW_BATCH_MAX_CHARS": "180"}):
                extract_classification_rows_with_llm(state, llm, output_dir=tmp, segment_max_chars=180)

            self.assertGreater(llm.calls, 1)
            self.assertEqual(len(state.classification_rows), llm.calls)
            trace = state.step_traces[-1]
            self.assertEqual(trace.step_name, "extract_classification_rows_with_llm")
            self.assertGreater(trace.input_summary["segments"], 1)
            self.assertGreater(trace.input_summary["batches"], 1)

    def test_row_extraction_resumes_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_state = self._state()
            first_llm = FakeRowLLM()
            with patch.dict("os.environ", {"ROW_BATCH_MAX_CHARS": "180"}):
                extract_classification_rows_with_llm(first_state, first_llm, output_dir=tmp, segment_max_chars=180)

            second_state = self._state()
            second_llm = FakeRowLLM()
            with patch.dict("os.environ", {"ROW_BATCH_MAX_CHARS": "180"}):
                extract_classification_rows_with_llm(second_state, second_llm, output_dir=tmp, segment_max_chars=180)

            self.assertEqual(second_llm.calls, 0)
            self.assertEqual(len(second_state.classification_rows), len(first_state.classification_rows))
            self.assertEqual(
                second_state.step_traces[-1].input_summary["cached_batches"],
                first_state.step_traces[-1].input_summary["batches"],
            )

    def test_checkpoint_signature_changes_when_prompt_changes(self):
        segment = TableSegment(
            segment_id="doc_1_chunk_1_seg_1",
            source_chunk_id="doc_1_chunk_1",
            doc_name="sample.txt",
            section_title="附录B",
            text="001项目A 示例A 原始数据 个人 严重危害 一般数据3级",
            position=1,
            page_number=None,
            line_start=1,
            line_end=1,
            source_method="text",
            source_warning="",
            block_signal="table_like",
            header_text="类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别",
        )

        self.assertNotEqual(
            _segment_signature([segment], prompt_text="prompt v1"),
            _segment_signature([segment], prompt_text="prompt v2"),
        )

    def test_corrupt_checkpoint_records_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/classification_row_batches.jsonl"
            signature = "sig"
            with open(path, "w", encoding="utf-8") as file:
                file.write(json.dumps({"signature": signature, "batch_index": "bad"}) + "\n")
                file.write(json.dumps({"signature": signature, "batch_index": 1, "segment_ids": "not-list"}) + "\n")
                file.write(json.dumps({"signature": signature, "batch_index": 2, "segment_ids": ["s2"]}) + "\n")

            records = _load_checkpoint_records(Path(path), signature)

            self.assertEqual(sorted(records), [2])

    def test_segment_batches_use_serialized_payload_size(self):
        segments = [
            TableSegment(
                segment_id=f"s{index}",
                source_chunk_id=f"c{index}",
                doc_name="sample.txt",
                section_title="附录B",
                text="短文本",
                position=index,
                page_number=None,
                line_start=index,
                line_end=index,
                source_method="text",
                source_warning="",
                block_signal="table_like",
                header_text="H" * 120,
            )
            for index in range(1, 3)
        ]

        batches = _build_segment_batches(segments, max_chars=250)

        self.assertEqual(len(batches), 2)


if __name__ == "__main__":
    unittest.main()
