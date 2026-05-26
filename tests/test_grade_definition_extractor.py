from __future__ import annotations

import unittest
from unittest.mock import patch

from src.core.agent_state import AgentState, DocumentChunk
from src.steps.grade_definition_extractor import extract_grade_definitions_with_llm


class GradeDefinitionExtractorTests(unittest.TestCase):
    def test_empty_chunks_skip_llm_and_trace_success(self) -> None:
        state = AgentState(task="test")

        with patch("src.steps.grade_definition_extractor.call_llm_json") as call_llm_json:
            result = extract_grade_definitions_with_llm(state, object())

        call_llm_json.assert_not_called()
        self.assertEqual(result.grade_scheme, [])
        self.assertEqual(result.step_traces[-1].step_name, "extract_grade_definitions_with_llm")
        self.assertEqual(result.step_traces[-1].output_summary["grade_definitions"], 0)

    def test_extracts_grade_definitions_without_hardcoded_levels(self) -> None:
        chunk = DocumentChunk(
            chunk_id="doc_1_chunk_1",
            doc_id="doc_1",
            doc_name="policy.txt",
            section_title="分级说明",
            text="3级 一般数据3级 无条件共享 有条件开放",
            position=1,
            source_method="text",
        )
        state = AgentState(task="test", chunks=[chunk])
        state.block_signals = {"doc_1_chunk_1": {"block_signal": "grade_legend"}}

        def fake_call_llm_json(**kwargs):
            return (
                {
                    "grade_definitions": [
                        {
                            "grade_name": "一般数据3级",
                            "definition": "3级 一般数据3级",
                            "criteria": ["无条件共享", "有条件开放"],
                            "evidence_quote": "3级 一般数据3级 无条件共享 有条件开放",
                            "evidence_chunk_ids": ["doc_1_chunk_1"],
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                            "status": "evidence_supported",
                        }
                    ]
                },
                "raw",
            )

        with patch("src.steps.grade_definition_extractor.call_llm_json", side_effect=fake_call_llm_json):
            result = extract_grade_definitions_with_llm(state, object())

        self.assertEqual(result.grade_scheme[0].grade_name, "一般数据3级")
        self.assertEqual(result.grade_scheme[0].evidence_refs[0].chunk_id, "doc_1_chunk_1")

    def test_missing_valid_evidence_refs_forces_review(self) -> None:
        chunk = DocumentChunk(
            chunk_id="doc_1_chunk_1",
            doc_id="doc_1",
            doc_name="policy.txt",
            section_title="分级说明",
            text="一般数据3级",
            position=1,
            source_method="text",
        )
        state = AgentState(task="test", chunks=[chunk])

        def fake_call_llm_json(**kwargs):
            return (
                {
                    "grade_definitions": [
                        {
                            "grade_name": "一般数据3级",
                            "definition": "一般数据3级",
                            "criteria": [],
                            "evidence_quote": "一般数据3级",
                            "evidence_chunk_ids": ["missing_chunk"],
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                            "status": "evidence_supported",
                        }
                    ]
                },
                "raw",
            )

        with patch("src.steps.grade_definition_extractor.call_llm_json", side_effect=fake_call_llm_json):
            result = extract_grade_definitions_with_llm(state, object())

        self.assertTrue(result.grade_scheme[0].needs_review)
        self.assertEqual(getattr(result.grade_scheme[0], "review_reason"), "分级定义缺少有效证据引用。")
        self.assertEqual(result.grade_scheme[0].status, "proposed")


if __name__ == "__main__":
    unittest.main()
