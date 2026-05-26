from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.core.agent_state import AgentState, DocumentChunk
from src.llm.task_utils import chunk_payload
import src.steps.evidence_claim_extractor as extractor
from src.steps.evidence_claim_extractor import _build_claim_batches


def make_chunk(
    chunk_id: str,
    text: str,
    source_method: str = "text",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        doc_name="policy.md",
        section_title="section",
        text=text,
        position=int(chunk_id.rsplit("_", 1)[-1]),
        source_method=source_method,
    )


class ClaimBatchingTests(unittest.TestCase):
    def test_batches_merge_small_chunks_until_char_budget(self) -> None:
        chunks = [
            make_chunk("chunk_1", "a" * 40),
            make_chunk("chunk_2", "b" * 40),
            make_chunk("chunk_3", "c" * 40),
            make_chunk("chunk_4", "d" * 40),
            make_chunk("chunk_5", "e" * 40),
        ]

        batches = _build_claim_batches(chunks, max_chunks=10, max_chars=100)

        self.assertEqual(
            [[chunk.chunk_id for chunk in batch] for batch in batches],
            [["chunk_1", "chunk_2"], ["chunk_3", "chunk_4"], ["chunk_5"]],
        )

    def test_oversized_chunk_is_single_batch(self) -> None:
        chunks = [
            make_chunk("chunk_1", "a" * 30),
            make_chunk("chunk_2", "b" * 120),
            make_chunk("chunk_3", "c" * 30),
        ]

        batches = _build_claim_batches(chunks, max_chunks=10, max_chars=100)

        self.assertEqual(
            [[chunk.chunk_id for chunk in batch] for batch in batches],
            [["chunk_1"], ["chunk_2"], ["chunk_3"]],
        )

    def test_batching_respects_max_chunk_count(self) -> None:
        chunks = [
            make_chunk("chunk_1", "a" * 10),
            make_chunk("chunk_2", "b" * 10),
            make_chunk("chunk_3", "c" * 10),
        ]

        batches = _build_claim_batches(chunks, max_chunks=2, max_chars=100)

        self.assertEqual(
            [[chunk.chunk_id for chunk in batch] for batch in batches],
            [["chunk_1", "chunk_2"], ["chunk_3"]],
        )

    def test_zero_char_budget_uses_fixed_chunk_count(self) -> None:
        chunks = [
            make_chunk("chunk_1", "a" * 10),
            make_chunk("chunk_2", "b" * 10),
            make_chunk("chunk_3", "c" * 10),
        ]

        batches = _build_claim_batches(chunks, max_chunks=2, max_chars=0)

        self.assertEqual(
            [[chunk.chunk_id for chunk in batch] for batch in batches],
            [["chunk_1", "chunk_2"], ["chunk_3"]],
        )


class ChunkPayloadSignalTests(unittest.TestCase):
    def test_chunk_payload_includes_signal_without_dropping_text(self) -> None:
        chunks = [
            make_chunk("chunk_1", "1. 总则", source_method="ocr"),
            make_chunk("chunk_2", "字段\t含义\nA\tB"),
            make_chunk("chunk_3", "@@@ ### !!!", source_method="ocr"),
            make_chunk("chunk_4", "本制度规定数据分类应当依据业务属性。"),
        ]

        payload = chunk_payload(chunks)

        self.assertEqual([item["text"] for item in payload], [chunk.text for chunk in chunks])
        self.assertEqual(
            [item["chunk_signal"] for item in payload],
            ["short_ocr", "table_like", "possible_noise", "normal"],
        )


class ClaimCheckpointResumeTests(unittest.TestCase):
    def test_dynamic_batches_resume_from_checkpoint_by_chunk_ids(self) -> None:
        chunks = [
            make_chunk("chunk_1", "x" * 40),
            make_chunk("chunk_2", "y" * 40),
            make_chunk("chunk_3", "z" * 40),
        ]
        calls: list[int] = []

        def fake_call_llm_json(**kwargs):
            calls.append(kwargs["payload"]["batch_index"])
            chunk_id = kwargs["payload"]["document_chunks"][0]["chunk_id"]
            return (
                {
                    "claims": [
                        {
                            "claim_type": "definition",
                            "subject": "x",
                            "predicate": "定义",
                            "object": "",
                            "value": chunk_id,
                            "evidence_quote": "x",
                            "support_level": "explicit",
                            "evidence_chunk_ids": [chunk_id],
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                            "status": "evidence_supported",
                        }
                    ]
                },
                "raw",
            )

        env = {
            "CLAIM_BATCH_SIZE": "4",
            "CLAIM_BATCH_MAX_CHARS": "100",
            "CLAIM_CHECKPOINT_ENABLED": "true",
            "CLAIM_RESUME": "true",
        }
        with TemporaryDirectory() as output_dir:
            with patch.dict("os.environ", env, clear=False):
                with patch.object(extractor, "call_llm_json", side_effect=fake_call_llm_json):
                    with redirect_stdout(StringIO()):
                        first = extractor.extract_evidence_claims_with_llm(
                            AgentState(task="test", chunks=chunks),
                            object(),
                            output_dir,
                        )
                        second = extractor.extract_evidence_claims_with_llm(
                            AgentState(task="test", chunks=chunks),
                            object(),
                            output_dir,
                        )

        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(first.evidence_claims), 2)
        self.assertEqual(len(second.evidence_claims), 2)
        self.assertEqual(second.step_traces[-1].input_summary["cached_batches"], 2)


if __name__ == "__main__":
    unittest.main()
