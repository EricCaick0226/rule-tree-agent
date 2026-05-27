from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.core.agent_state import AgentState, SourceDocument
from src.pipeline.agent_executor import _run_llm_steps, _run_step, create_plan, run_agent


class RowFirstPipelineTests(unittest.TestCase):
    def test_default_plan_uses_row_first_steps_and_not_tree_first_steps(self) -> None:
        tools = [step["tool"] for step in create_plan("generate_rule_tree_from_docs")]

        self.assertIn("extract_classification_rows_with_llm", tools)
        self.assertIn("project_tree_from_rows", tools)
        self.assertNotIn("discover_dimensions_with_llm", tools)
        self.assertNotIn("synthesize_taxonomy_with_llm", tools)

    def test_default_plan_order_is_row_first_mvp(self) -> None:
        tools = [step["tool"] for step in create_plan("generate_rule_tree_from_docs")]

        self.assertEqual(
            tools,
            [
                "parse_documents",
                "chunk_documents",
                "build_evidence_index",
                "extract_evidence_claims_with_llm",
                "classify_document_blocks_with_llm",
                "extract_classification_rows_with_llm",
                "extract_grade_definitions_with_llm",
                "normalize_classification_rows",
                "validate_row_grounding",
                "project_tree_from_rows",
                "export_outputs",
            ],
        )

    def test_rejects_pdf_inputs_for_row_first_mvp(self) -> None:
        state = AgentState(
            task="test",
            task_type="generate_rule_tree_from_docs",
            input_files=["/tmp/policy.pdf"],
            documents=[
                SourceDocument(
                    doc_id="doc_1",
                    doc_name="policy.pdf",
                    file_path="/tmp/policy.pdf",
                    raw_text="",
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "row-first MVP only supports .txt and .md"):
            _run_llm_steps(state, "outputs", object())

    def test_run_agent_rejects_pdf_before_llm_client_requires_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "row-first MVP only supports .txt and .md"):
            run_agent("test", ["/tmp/policy.pdf"], output_dir="outputs")

    def test_run_step_routes_deterministic_row_first_steps(self) -> None:
        state = AgentState(task="test")

        with patch("src.pipeline.agent_executor.normalize_classification_rows", return_value=state) as normalize:
            self.assertIs(_run_step("normalize_classification_rows", state, "outputs", object()), state)
        normalize.assert_called_once_with(state)

        with patch("src.pipeline.agent_executor.validate_row_grounding", return_value=[]) as validate:
            self.assertIs(_run_step("validate_row_grounding", state, "outputs", object()), state)
        validate.assert_called_once_with(state)

        with patch("src.pipeline.agent_executor.project_tree_from_rows", return_value=state) as project:
            self.assertIs(_run_step("project_tree_from_rows", state, "outputs", object()), state)
        project.assert_called_once_with(state)

    def test_llm_used_stays_false_when_llm_steps_short_circuit_without_raw_response(self) -> None:
        state = AgentState(task="test", task_type="generate_rule_tree_from_docs", input_files=[])

        with TemporaryDirectory() as tmp:
            result = _run_llm_steps(state, tmp, object())

        self.assertFalse(result.llm_used)
        self.assertTrue(result.step_traces)
        self.assertFalse(any(trace.raw_response for trace in result.step_traces))

    def test_agent_demo_help_marks_ocr_as_legacy_noop(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.agent_demo", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        normalized_help = " ".join(result.stdout.split())
        self.assertIn(
            "Legacy option retained for compatibility; row-first MVP supports .txt and .md only.",
            normalized_help,
        )
        self.assertNotIn("Use OCR for PDF pages", normalized_help)

    def test_row_extraction_prompt_requires_all_rows_and_continuation_inheritance(self) -> None:
        prompt = Path("prompts/extract_classification_rows_prompt.md").read_text(encoding="utf-8")

        self.assertIn("抽取本批次所有", prompt)
        self.assertIn("不要只抽取示例", prompt)
        self.assertIn("续表", prompt)
        self.assertIn("继承", prompt)
        self.assertIn("data_range_examples", prompt)
        self.assertIn("processing_degree", prompt)
        self.assertIn("impact_object", prompt)
        self.assertIn("impact_degree", prompt)
        self.assertIn("grade_evidence_quote", prompt)
        self.assertIn("table_segments", prompt)
        self.assertIn("segment.text", prompt)
        self.assertIn("source_chunk_id", prompt)
        self.assertNotIn("document_chunks", prompt)


if __name__ == "__main__":
    unittest.main()
