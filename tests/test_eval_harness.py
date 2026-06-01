import json
import tempfile
import unittest
from pathlib import Path

from src.eval_harness.loader import load_output_dir


class EvalHarnessLoaderTests(unittest.TestCase):
    def test_loads_complete_outputs_and_skips_corrupt_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_text(
                json.dumps({"classification_rows": [{"row_id": "R1"}]}),
                encoding="utf-8",
            )
            (output_dir / "rule_tree.json").write_text(
                json.dumps({"tree": []}),
                encoding="utf-8",
            )
            (output_dir / "review_report.md").write_text(
                "# Review\n\nNeeds review.",
                encoding="utf-8",
            )
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                json.dumps(
                    {
                        "signature": "sig-1",
                        "cache_schema_version": 1,
                        "batch_index": 0,
                        "batch_count": 1,
                    }
                )
                + "\n"
                + "{bad json\n",
                encoding="utf-8",
            )
            (checkpoints_dir / "evidence_claim_batches.jsonl").write_text(
                json.dumps({"batch_index": 0, "batch_count": 1}) + "\n",
                encoding="utf-8",
            )
            (checkpoints_dir / "evidence_checkpoint.jsonl").write_text(
                json.dumps({"batch_index": 0, "batch_count": 99}) + "\n",
                encoding="utf-8",
            )
            (checkpoints_dir / "block_signal_batches.jsonl").write_text(
                json.dumps({"batch_index": 0, "batch_count": 1}) + "\n",
                encoding="utf-8",
            )
            (checkpoints_dir / "block_checkpoint.jsonl").write_text(
                json.dumps({"batch_index": 0, "batch_count": 99}) + "\n",
                encoding="utf-8",
            )
            debug_dir = output_dir / "debug"
            debug_dir.mkdir()
            (debug_dir / "row_retry_debug.txt").write_text(
                "Unterminated string",
                encoding="utf-8",
            )
            traces_dir = output_dir / "traces"
            traces_dir.mkdir()
            (traces_dir / "03_extract_classification_rows_with_llm.txt").write_text(
                "trace text",
                encoding="utf-8",
            )

            loaded = load_output_dir(output_dir)

            self.assertTrue(loaded.files["rule_table_json"].exists)
            self.assertEqual(
                loaded.files["rule_table_json"].data["classification_rows"][0]["row_id"],
                "R1",
            )
            self.assertTrue(loaded.files["rule_tree_json"].exists)
            self.assertTrue(loaded.files["review_report_md"].exists)
            self.assertEqual(
                loaded.files["review_report_md"].data,
                "# Review\n\nNeeds review.",
            )
            self.assertTrue(loaded.row_checkpoint.exists)
            self.assertEqual(len(loaded.row_checkpoint.records), 1)
            self.assertEqual(loaded.row_checkpoint.records[0]["signature"], "sig-1")
            self.assertEqual(loaded.row_checkpoint.corrupt_records, 1)
            self.assertEqual(len(loaded.row_checkpoint.errors), 1)
            self.assertEqual(
                loaded.evidence_checkpoint.path.name,
                "evidence_claim_batches.jsonl",
            )
            self.assertEqual(loaded.evidence_checkpoint.records[0]["batch_count"], 1)
            self.assertEqual(
                loaded.block_checkpoint.path.name,
                "block_signal_batches.jsonl",
            )
            self.assertEqual(loaded.block_checkpoint.records[0]["batch_count"], 1)
            self.assertEqual(len(loaded.debug_files), 1)
            self.assertEqual(loaded.debug_files[0].text, "Unterminated string")
            self.assertEqual(len(loaded.trace_files), 1)
            self.assertEqual(loaded.trace_files[0].text, "trace text")

    def test_loads_partial_run_with_checkpoint_and_missing_rule_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                json.dumps({"batch_index": 0, "batch_count": 2}) + "\n",
                encoding="utf-8",
            )
            (checkpoints_dir / "row_checkpoint.jsonl").write_text(
                json.dumps({"batch_index": 0, "batch_count": 99}) + "\n",
                encoding="utf-8",
            )

            loaded = load_output_dir(output_dir)

            self.assertFalse(loaded.files["rule_table_json"].exists)
            self.assertEqual(loaded.files["rule_table_json"].data, None)
            self.assertEqual(loaded.files["rule_table_json"].error, "")
            self.assertTrue(loaded.row_checkpoint.exists)
            self.assertEqual(
                loaded.row_checkpoint.path.name,
                "classification_row_batches.jsonl",
            )
            self.assertEqual(loaded.row_checkpoint.records[0]["batch_count"], 2)
            self.assertFalse(loaded.evidence_checkpoint.exists)
            self.assertFalse(loaded.block_checkpoint.exists)

    def test_records_invalid_utf8_errors_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_bytes(b"\xff")
            (output_dir / "review_report.md").write_bytes(b"\xff")
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_bytes(b"\xff")
            debug_dir = output_dir / "debug"
            debug_dir.mkdir()
            (debug_dir / "bad_debug.txt").write_bytes(b"\xff")
            traces_dir = output_dir / "traces"
            traces_dir.mkdir()
            (traces_dir / "bad_trace.txt").write_bytes(b"\xff")

            loaded = load_output_dir(output_dir)

            self.assertTrue(loaded.files["rule_table_json"].exists)
            self.assertIn("invalid", loaded.files["rule_table_json"].error.lower())
            self.assertTrue(loaded.files["review_report_md"].exists)
            self.assertIn("invalid", loaded.files["review_report_md"].error.lower())
            self.assertTrue(loaded.row_checkpoint.exists)
            self.assertEqual(loaded.row_checkpoint.records, [])
            self.assertEqual(len(loaded.row_checkpoint.errors), 1)
            self.assertIn("invalid", loaded.row_checkpoint.errors[0].lower())
            self.assertEqual(len(loaded.debug_files), 1)
            self.assertTrue(loaded.debug_files[0].text.startswith("READ_ERROR:"))
            self.assertEqual(len(loaded.trace_files), 1)
            self.assertTrue(loaded.trace_files[0].text.startswith("READ_ERROR:"))


if __name__ == "__main__":
    unittest.main()
