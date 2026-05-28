from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.core.agent_state import AgentState, DocumentChunk
from src.llm.task_utils import LLMJSONValidationError
from src.steps.block_classifier import classify_document_blocks_with_llm


def chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        doc_name="policy.txt",
        section_title="分类表",
        text=text,
        position=int(chunk_id.rsplit("_", 1)[-1]),
        source_method="text",
        line_start=1,
        line_end=2,
    )


class BlockClassifierTests(unittest.TestCase):
    def test_classifies_blocks_without_dropping_chunks(self) -> None:
        state = AgentState(
            task="test",
            chunks=[
                chunk("doc_1_chunk_1", "一级分类 二级分类 推荐分级 分类说明\n基础资源 服务范围 3级 说明"),
                chunk("doc_1_chunk_2", "3级 一般数据3级 无条件共享"),
            ],
        )

        def fake_call_llm_json(**kwargs):
            return (
                {
                    "block_signals": [
                        {
                            "chunk_id": "doc_1_chunk_1",
                            "block_signal": "table_like",
                            "reason": "包含分类路径、推荐分级和说明列",
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                        },
                        {
                            "chunk_id": "doc_1_chunk_2",
                            "block_signal": "grade_legend",
                            "reason": "包含等级名称和共享属性",
                            "confidence": 0.85,
                            "needs_review": False,
                            "review_reason": "",
                        },
                    ]
                },
                "raw",
            )

        with TemporaryDirectory() as tmp:
            with patch("src.steps.block_classifier.call_llm_json", side_effect=fake_call_llm_json):
                result = classify_document_blocks_with_llm(state, object(), output_dir=tmp)

        self.assertEqual(result.block_signals["doc_1_chunk_1"]["block_signal"], "table_like")
        self.assertEqual(result.block_signals["doc_1_chunk_2"]["block_signal"], "grade_legend")
        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(result.step_traces[-1].step_name, "classify_document_blocks_with_llm")

    def test_invalid_and_omitted_signals_require_review(self) -> None:
        state = AgentState(
            task="test",
            chunks=[
                chunk("doc_1_chunk_1", "无法识别的块类型"),
                chunk("doc_1_chunk_2", "LLM 省略了这个 chunk"),
            ],
        )

        def fake_call_llm_json(**kwargs):
            return (
                {
                    "block_signals": [
                        {
                            "chunk_id": "doc_1_chunk_1",
                            "block_signal": "summary_table",
                            "reason": "LLM 返回了未注册的信号",
                            "confidence": 0.7,
                            "needs_review": False,
                            "review_reason": "",
                        },
                    ]
                },
                "raw",
            )

        with TemporaryDirectory() as tmp:
            with patch("src.steps.block_classifier.call_llm_json", side_effect=fake_call_llm_json):
                result = classify_document_blocks_with_llm(state, object(), output_dir=tmp)

        invalid_signal = result.block_signals["doc_1_chunk_1"]
        self.assertEqual(invalid_signal["block_signal"], "normal")
        self.assertTrue(invalid_signal["needs_review"])
        self.assertTrue(invalid_signal["review_reason"])

        omitted_signal = result.block_signals["doc_1_chunk_2"]
        self.assertEqual(omitted_signal["block_signal"], "normal")
        self.assertTrue(omitted_signal["needs_review"])
        self.assertTrue(omitted_signal["review_reason"])

    def test_batches_block_classification_and_writes_checkpoint(self) -> None:
        state = AgentState(
            task="test",
            chunks=[
                chunk("doc_1_chunk_1", "A" * 120),
                chunk("doc_1_chunk_2", "B" * 120),
                chunk("doc_1_chunk_3", "C" * 120),
            ],
        )
        seen_batch_sizes: list[int] = []

        def fake_call_llm_json(**kwargs):
            payload_chunks = kwargs["payload"]["document_chunks"]
            seen_batch_sizes.append(len(payload_chunks))
            return (
                {
                    "block_signals": [
                        {
                            "chunk_id": item["chunk_id"],
                            "block_signal": "normal",
                            "reason": "测试信号",
                            "confidence": 0.5,
                            "needs_review": True,
                            "review_reason": "测试",
                        }
                        for item in payload_chunks
                    ]
                },
                "raw",
            )

        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"BLOCK_BATCH_MAX_CHARS": "500"}):
                with patch("src.steps.block_classifier.call_llm_json", side_effect=fake_call_llm_json):
                    result = classify_document_blocks_with_llm(state, object(), output_dir=tmp)

            self.assertEqual(set(result.block_signals), {chunk.chunk_id for chunk in state.chunks})
            self.assertGreater(len(seen_batch_sizes), 1)
            checkpoint = Path(tmp, "checkpoints", "block_signal_batches.jsonl")
            self.assertTrue(checkpoint.exists())
            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), len(seen_batch_sizes))

    def test_failed_multi_chunk_batch_is_split_and_debugged(self) -> None:
        state = AgentState(
            task="test",
            chunks=[
                chunk("doc_1_chunk_1", "A" * 120),
                chunk("doc_1_chunk_2", "B" * 120),
            ],
        )
        calls: list[list[str]] = []

        def fake_call_llm_json(**kwargs):
            payload_chunks = kwargs["payload"]["document_chunks"]
            batch_ids = [item["chunk_id"] for item in payload_chunks]
            calls.append(batch_ids)
            if len(payload_chunks) > 1:
                raise LLMJSONValidationError("bad json", raw_response="{bad")
            return (
                {
                    "block_signals": [
                        {
                            "chunk_id": payload_chunks[0]["chunk_id"],
                            "block_signal": "normal",
                            "reason": "拆分后成功",
                            "confidence": 0.5,
                            "needs_review": True,
                            "review_reason": "测试",
                        }
                    ]
                },
                "raw",
            )

        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"BLOCK_BATCH_MAX_CHARS": "99999"}):
                with patch("src.steps.block_classifier.call_llm_json", side_effect=fake_call_llm_json):
                    result = classify_document_blocks_with_llm(state, object(), output_dir=tmp)

            self.assertEqual(calls[0], ["doc_1_chunk_1", "doc_1_chunk_2"])
            self.assertEqual(calls[1:], [["doc_1_chunk_1"], ["doc_1_chunk_2"]])
            self.assertEqual(set(result.block_signals), {chunk.chunk_id for chunk in state.chunks})
            debug_files = list(Path(tmp, "debug").glob("failed_block_batch_*.txt"))
            self.assertEqual(len(debug_files), 1)
            self.assertIn("{bad", debug_files[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
