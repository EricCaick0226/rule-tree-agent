import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src.eval_harness.loader import load_output_dir
from src.eval_harness.metrics import build_eval_report
from src.eval_harness.report import render_json_report, render_markdown_report


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


class EvalHarnessMetricsTests(unittest.TestCase):
    def test_builds_quality_checkpoint_risk_and_recommendation_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "path_levels": ["A", "B"],
                                "needs_review": False,
                                "evidence_quote": "source quote",
                                "evidence_refs": [{"page": 1}],
                            },
                            {
                                "path_levels": ["A", "B"],
                                "needs_review": True,
                                "evidence_quote": "",
                                "evidence_refs": [{"page": 2}],
                            },
                            {
                                "path_levels": ["项目", "占位"],
                                "needs_review": True,
                                "evidence_quote": "header-like fragment",
                                "evidence_refs": [],
                            },
                        ],
                        "validation_issues": [
                            {
                                "severity": "high",
                                "target": "classification_rows[1]",
                                "message": "failed validation",
                            },
                            {
                                "severity": "low",
                                "target": "classification_rows[2]",
                                "message": "weak evidence",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "batch_index": 1,
                                "batch_count": 2,
                                "rows": [{"id": "r1"}],
                                "elapsed_seconds": 4.0,
                                "split_retry_debug_paths": [],
                            }
                        ),
                        json.dumps(
                            {
                                "batch_index": 2,
                                "batch_count": 2,
                                "classification_rows": [{"id": "r2"}, {"id": "r3"}],
                                "elapsed_seconds": 5.0,
                                "split_retry_debug_paths": ["debug/row_retry.txt"],
                            }
                        ),
                    ]
                )
                + "\n{bad json\n",
                encoding="utf-8",
            )
            debug_dir = output_dir / "debug"
            debug_dir.mkdir()
            (debug_dir / "row_retry.txt").write_text(
                "Unterminated string while parsing retry response",
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertEqual(report["quality"]["classification_row_count"], 3)
            self.assertEqual(report["quality"]["unique_path_count"], 2)
            self.assertEqual(report["quality"]["duplicate_path_count"], 1)
            self.assertEqual(report["quality"]["needs_review_count"], 2)
            self.assertEqual(report["quality"]["missing_evidence_quote_count"], 1)
            self.assertEqual(report["quality"]["missing_evidence_refs_count"], 1)
            self.assertEqual(report["quality"]["validation_issue_count"], 2)
            self.assertEqual(
                report["quality"]["validation_issue_count_by_severity"]["high"],
                1,
            )
            self.assertEqual(
                report["quality"]["high_severity_targets"],
                ["classification_rows[1]"],
            )
            self.assertEqual(report["row_extraction"]["batch_count"], 2)
            self.assertEqual(report["row_extraction"]["completed_batch_indices"], [1, 2])
            self.assertEqual(report["row_extraction"]["rows_per_batch"], {"1": 1, "2": 2})
            self.assertEqual(
                report["row_extraction"]["elapsed_seconds_per_batch"],
                {"1": 4.0, "2": 5.0},
            )
            self.assertEqual(
                report["row_extraction"]["slowest_batches"],
                [
                    {"batch_index": 2, "elapsed_seconds": 5.0},
                    {"batch_index": 1, "elapsed_seconds": 4.0},
                ],
            )
            self.assertEqual(report["row_extraction"]["batches_with_debug_paths"], [2])
            self.assertTrue(report["row_extraction"]["appears_complete"])
            self.assertEqual(report["row_extraction"]["total_checkpoint_rows"], 3)
            self.assertEqual(report["row_extraction"]["total_elapsed_seconds"], 9.0)
            self.assertEqual(report["row_extraction"]["corrupt_record_count"], 1)
            self.assertEqual(
                report["run_completeness"]["checkpoint_corrupt_records"]["row"],
                1,
            )
            self.assertFalse(report["recommendation"]["merge_ready"])
            self.assertTrue(
                any(
                    item["type"] == "debug_json_failure"
                    and item["severity"] == "review"
                    for item in report["risk_signals"]
                )
            )
            self.assertTrue(
                any(
                    item["type"] == "generic_path_fragment"
                    and item["severity"] == "review"
                    and item["path_levels"] == ["项目", "占位"]
                    for item in report["risk_signals"]
                )
            )

    def test_partial_run_without_rule_table_returns_stable_zero_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                json.dumps(
                    {
                        "batch_index": 1,
                        "batch_count": 2,
                        "rows": [{"id": "r1"}],
                        "elapsed_seconds": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertFalse(report["inputs"]["rule_table_json"])
            self.assertEqual(report["quality"]["classification_row_count"], 0)
            self.assertFalse(report["row_extraction"]["appears_complete"])
            self.assertFalse(report["recommendation"]["merge_ready"])
            self.assertIn(
                "missing final artifact: rule_table.json",
                report["recommendation"]["reasons"],
            )

    def test_missing_final_artifacts_are_reported_individually(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                json.dumps({"batch_index": 1, "batch_count": 1, "rows": []}) + "\n",
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertEqual(
                report["run_completeness"]["missing_final_artifacts"],
                ["rule_table.json", "rule_tree.json", "review_report.md"],
            )
            self.assertIn(
                "missing final artifact: rule_table.json",
                report["recommendation"]["reasons"],
            )
            self.assertIn(
                "missing final artifact: rule_tree.json",
                report["recommendation"]["reasons"],
            )
            self.assertIn(
                "missing final artifact: review_report.md",
                report["recommendation"]["reasons"],
            )

    def test_debug_json_failure_alone_blocks_merge_ready_for_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "path_levels": ["A", "B"],
                                "evidence_quote": "quote",
                                "evidence_refs": [{"page": 1}],
                            }
                        ],
                        "validation_issues": [],
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "rule_tree.json").write_text(
                json.dumps({"tree": []}),
                encoding="utf-8",
            )
            (output_dir / "review_report.md").write_text(
                "# Review\n",
                encoding="utf-8",
            )
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                json.dumps({"batch_index": 1, "batch_count": 1, "rows": [{}]}) + "\n",
                encoding="utf-8",
            )
            debug_dir = output_dir / "debug"
            debug_dir.mkdir()
            (debug_dir / "recovered_retry.txt").write_text(
                "Unterminated string in recovered retry",
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertTrue(
                any(item["type"] == "debug_json_failure" for item in report["risk_signals"])
            )
            self.assertTrue(
                any(
                    item["type"] == "debug_json_failure"
                    and item["path"] == "debug/recovered_retry.txt"
                    for item in report["risk_signals"]
                )
            )
            self.assertFalse(report["recommendation"]["merge_ready"])
            self.assertIn(
                "row extraction debug failures found",
                report["recommendation"]["reasons"],
            )

    def test_non_rule_table_final_artifact_read_error_blocks_merge_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_text(
                json.dumps({"classification_rows": [], "validation_issues": []}),
                encoding="utf-8",
            )
            (output_dir / "rule_tree.json").write_bytes(b"\xff")
            (output_dir / "review_report.md").write_text(
                "# Review\n",
                encoding="utf-8",
            )
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                json.dumps({"batch_index": 1, "batch_count": 1, "rows": []}) + "\n",
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertIn("rule_tree_json", report["run_completeness"]["file_errors"])
            self.assertFalse(report["recommendation"]["merge_ready"])
            self.assertTrue(
                any(
                    reason.startswith("rule_tree.json read error:")
                    for reason in report["recommendation"]["reasons"]
                )
            )

    def test_review_report_read_error_blocks_merge_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_text(
                json.dumps({"classification_rows": [], "validation_issues": []}),
                encoding="utf-8",
            )
            (output_dir / "rule_tree.json").write_text(
                json.dumps({"tree": []}),
                encoding="utf-8",
            )
            (output_dir / "review_report.md").write_bytes(b"\xff")
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                json.dumps({"batch_index": 1, "batch_count": 1, "rows": []}) + "\n",
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertFalse(report["recommendation"]["merge_ready"])
            self.assertTrue(
                any(
                    reason.startswith("review_report.md read error:")
                    for reason in report["recommendation"]["reasons"]
                )
            )

    def test_absent_row_checkpoint_does_not_add_incomplete_checkpoint_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_text(
                json.dumps({"classification_rows": [], "validation_issues": []}),
                encoding="utf-8",
            )
            (output_dir / "rule_tree.json").write_text(
                json.dumps({"tree": []}),
                encoding="utf-8",
            )
            (output_dir / "review_report.md").write_text(
                "# Review\n",
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertFalse(report["inputs"]["row_checkpoint"])
            self.assertFalse(report["row_extraction"]["appears_complete"])
            self.assertNotIn(
                "row checkpoint does not appear complete",
                report["recommendation"]["reasons"],
            )

    def test_non_finite_elapsed_seconds_is_zero_and_json_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_text(
                json.dumps({"classification_rows": [], "validation_issues": []}),
                encoding="utf-8",
            )
            (output_dir / "rule_tree.json").write_text(
                json.dumps({"tree": []}),
                encoding="utf-8",
            )
            (output_dir / "review_report.md").write_text(
                "# Review\n",
                encoding="utf-8",
            )
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "classification_row_batches.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "batch_index": 1,
                                "batch_count": 2,
                                "rows": [],
                                "elapsed_seconds": "NaN",
                            }
                        ),
                        json.dumps(
                            {
                                "batch_index": 2,
                                "batch_count": 2,
                                "rows": [],
                                "elapsed_seconds": "Infinity",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertEqual(
                report["row_extraction"]["elapsed_seconds_per_batch"],
                {"1": 0.0, "2": 0.0},
            )
            self.assertEqual(report["row_extraction"]["total_elapsed_seconds"], 0.0)
            json.dumps(report, allow_nan=False)

    def test_reports_structure_risks_from_rows_and_evidence_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "rule_table.json").write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "path_levels": ["项", "06 血液管理", "目", "001 献血服务"],
                                "evidence_quote": "06 血液管理 001 献血服务",
                                "evidence_refs": [
                                    {
                                        "section_title": "6 多重标识符",
                                        "text": "同一资源可以有多个标识符。\n附 录 B\n表B.1 （续）\n类（1 位数字） 项（2 位数字） 目（3 位数字）\n06 血液管理 001 献血服务",
                                    }
                                ],
                            },
                            {
                                "path_levels": ["业务资源", "1 公共卫生", "01 疾病控制"],
                                "evidence_quote": "1 公共卫生 01 疾病控制",
                                "evidence_refs": [
                                    {
                                        "section_title": "附录 B / 业务资源分类 / 表B.1 业务资源分类目录（示例）",
                                        "text": "附 录 B\n表B.1 业务资源分类目录（示例）\n1 公共卫生 01 疾病控制",
                                    }
                                ],
                            },
                        ],
                        "validation_issues": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = build_eval_report(load_output_dir(output_dir))

            self.assertEqual(report["structure"]["header_as_path_count"], 1)
            self.assertEqual(report["structure"]["generic_column_path_count"], 1)
            self.assertEqual(report["structure"]["stale_section_title_count"], 1)
            self.assertEqual(report["structure"]["appendix_table_detected_count"], 2)
            self.assertEqual(report["structure"]["continued_table_count"], 1)
            self.assertTrue(
                any(item["type"] == "header_as_path" for item in report["risk_signals"])
            )
            self.assertTrue(
                any(item["type"] == "stale_section_title" for item in report["risk_signals"])
            )


class EvalHarnessReportTests(unittest.TestCase):
    def test_render_json_report_returns_stable_machine_readable_text(self):
        report = {
            "quality": {"classification_row_count": 3},
            "recommendation": {"merge_ready": True, "reasons": []},
        }

        text = render_json_report(report)

        self.assertTrue(text.endswith("\n"))
        parsed = json.loads(text)
        self.assertEqual(parsed["quality"]["classification_row_count"], 3)
        self.assertIn('"merge_ready": true', text)

    def test_render_json_report_rejects_nan_values(self):
        report = {"quality": {"total_elapsed_seconds": float("nan")}}

        with self.assertRaises(ValueError):
            render_json_report(report)

    def test_render_markdown_report_returns_required_sections_and_verdict(self):
        report = {
            "inputs": {
                "rule_table_json": True,
                "rule_tree_json": False,
                "review_report_md": True,
                "row_checkpoint": True,
                "debug_file_count": 1,
            },
            "row_extraction": {
                "batch_count": 2,
                "completed_batch_indices": [1],
                "appears_complete": False,
                "total_checkpoint_rows": 10,
                "total_elapsed_seconds": 12.5,
                "slowest_batches": [
                    {"batch_index": 1, "elapsed_seconds": 12.5},
                ],
            },
            "quality": {
                "classification_row_count": 10,
                "unique_path_count": 9,
                "duplicate_path_count": 1,
                "needs_review_count": 2,
                "missing_evidence_quote_count": 1,
                "missing_evidence_refs_count": 1,
                "validation_issue_count": 1,
                "validation_issue_count_by_severity": {"high": 1},
                "high_severity_targets": ["classification_rows[3]"],
            },
            "structure": {
                "header_as_path_count": 2,
                "generic_column_path_count": 3,
                "stale_section_title_count": 1,
                "appendix_table_detected_count": 4,
                "continued_table_count": 2,
            },
            "risk_signals": [
                {
                    "type": "debug_json_failure",
                    "severity": "review",
                    "message": "Debug file contains JSON parsing failure text.",
                },
                "legacy string risk",
            ],
            "recommendation": {
                "merge_ready": False,
                "reasons": ["high severity validation issues present"],
                "note": "Advisory-only diagnostics.",
            },
        }

        text = render_markdown_report(report)

        self.assertTrue(text.endswith("\n"))
        for section in (
            "# Eval Report",
            "## Verdict",
            "## Inputs",
            "## Runtime Summary",
            "## Quality Summary",
            "## Slowest Batches",
            "## Structure Risks",
            "## Validation Issues",
            "## Risk Signals",
            "## Recommended Next Action",
        ):
            self.assertIn(section, text)
        self.assertIn("merge_ready: false", text)
        self.assertIn("header_as_path_count: 2", text)
        self.assertIn("stale_section_title_count: 1", text)

    def test_render_markdown_report_bounds_multiline_risk_signal_detail(self):
        long_detail = "first line\n" + ("x" * 180) + "\ntail-token"
        report = {
            "risk_signals": [
                long_detail,
                {
                    "type": "debug_json_failure",
                    "severity": "review",
                    "message": long_detail,
                },
            ],
            "recommendation": {"merge_ready": False, "reasons": []},
        }

        text = render_markdown_report(report)
        risk_section = text.split("## Risk Signals\n", 1)[1].split(
            "\n## Recommended Next Action",
            1,
        )[0]

        self.assertNotIn("tail-token", risk_section)
        for line in risk_section.splitlines():
            if line:
                self.assertTrue(line.startswith("- "))
                self.assertLessEqual(len(line), 140)


class EvalHarnessCliTests(unittest.TestCase):
    def _build_complete_output_dir(self, output_dir):
        (output_dir / "rule_table.json").write_text(
            json.dumps(
                {
                    "classification_rows": [
                        {
                            "path_levels": ["root"],
                            "evidence_quote": "quoted evidence",
                            "evidence_refs": ["doc:1"],
                        }
                    ],
                    "validation_issues": [],
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "rule_tree.json").write_text(
            json.dumps({"tree": []}),
            encoding="utf-8",
        )
        (output_dir / "review_report.md").write_text(
            "# Review\n",
            encoding="utf-8",
        )
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "classification_row_batches.jsonl").write_text(
            json.dumps(
                {
                    "batch_index": 0,
                    "batch_count": 1,
                    "rows": [{"row_id": "R1"}],
                    "elapsed_seconds": 1.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_diagnose_writes_json_and_markdown_reports(self):
        from src.eval_harness.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            self._build_complete_output_dir(output_dir)
            top_level_before = {path.name for path in output_dir.iterdir()}

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(["diagnose", str(output_dir)])

            json_report_path = output_dir / "eval_report.json"
            markdown_report_path = output_dir / "eval_report.md"
            top_level_after = {path.name for path in output_dir.iterdir()}
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                top_level_after - top_level_before,
                {"eval_report.json", "eval_report.md"},
            )
            self.assertTrue(json_report_path.exists())
            self.assertTrue(markdown_report_path.exists())
            self.assertIn(str(json_report_path), stdout.getvalue())
            self.assertIn(str(markdown_report_path), stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")
            report = json.loads(json_report_path.read_text(encoding="utf-8"))
            self.assertIn("quality", report)
            self.assertIn(
                "# Eval Report",
                markdown_report_path.read_text(encoding="utf-8"),
            )

    def test_diagnose_returns_2_for_missing_output_dir(self):
        from src.eval_harness.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "missing"

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(["diagnose", str(missing_dir)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("output_dir is not a directory", stderr.getvalue())

    def test_diagnose_returns_1_when_report_write_fails(self):
        from src.eval_harness.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            self._build_complete_output_dir(output_dir)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch(
                "src.eval_harness.__main__.Path.write_text",
                side_effect=OSError("disk full"),
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = main(["diagnose", str(output_dir)])

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("failed to write eval reports", stderr.getvalue())
            self.assertIn("disk full", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
