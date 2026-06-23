from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.split_reference_candidates_from_output import split_reference_candidates


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SplitReferenceCandidatesFromOutputTests(unittest.TestCase):
    def test_splits_reference_candidates_out_of_rule_table(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "old_output"
            output_dir = root / "cleaned_output"
            _write_json(
                input_dir / "rule_table.json",
                {
                    "classification_schema": {"max_depth": 3, "source": "test"},
                    "grade_scheme": [],
                    "classification_rows": [
                        {
                            "row_id": "current",
                            "path_levels": ["基础资源", "设备资源", "硬件设备"],
                            "description": "硬件设备相关信息。",
                            "description_source": "classification_standard_excel",
                            "content_source": "reference_library",
                            "reference_prefilled_fields": ["path_levels", "description"],
                            "needs_review": True,
                            "review_reason": "层级路径继承自上下文，原文存在换行与表头穿插排版干扰",
                        },
                        {
                            "row_id": "candidate",
                            "path_levels": ["基础资源", "设备资源", "软件设备"],
                            "description": "软件设备相关信息。",
                            "description_source": "reference_library",
                            "row_source": "reference_library",
                            "content_source": "reference_library",
                            "inclusion_status": "review_candidate",
                            "evidence_status": "reference_only",
                            "reference_matches": [
                                {"reference_row_id": "ref_software"}
                            ],
                        },
                        {
                            "row_id": "term_definition",
                            "path_levels": ["术语和定义", "核心数据"],
                            "description": "核心数据是指...",
                            "description_source": "quoted",
                            "review_reason": "原文为术语定义段落而非标准分类分级表格明细行，需人工确认是否作为分级基准行纳入规则树。",
                        },
                    ],
                    "validation_issues": [],
                },
            )
            _write_json(
                input_dir / "rule_tree.json",
                {
                    "task": "old run",
                    "task_type": "generate_rule_tree_from_docs",
                    "nodes": [
                        {
                            "node_id": "old_current",
                            "name": "硬件设备",
                            "path": "基础资源 / 设备资源 / 硬件设备",
                            "level": 3,
                            "parent_id": "old_parent",
                        },
                        {
                            "node_id": "old_candidate",
                            "name": "软件设备",
                            "path": "基础资源 / 设备资源 / 软件设备",
                            "level": 3,
                            "parent_id": "old_parent",
                        },
                        {
                            "node_id": "old_term",
                            "name": "核心数据",
                            "path": "术语和定义 / 核心数据",
                            "level": 2,
                            "parent_id": "old_terms",
                        },
                    ],
                    "llm_enabled": True,
                    "llm_used": True,
                    "llm_model": "test-model",
                    "llm_base_url": "https://example.test/v1",
                },
            )

            summary = split_reference_candidates(input_dir, output_dir)
            table = json.loads((output_dir / "rule_table.json").read_text(encoding="utf-8"))
            candidates = json.loads(
                (output_dir / "reference_candidates.json").read_text(encoding="utf-8")
            )
            tree = json.loads((output_dir / "rule_tree.json").read_text(encoding="utf-8"))
            readme = (output_dir / "README.md").read_text(encoding="utf-8")
            has_review_report = (output_dir / "review_report.md").exists()
            has_candidates_md = (output_dir / "reference_candidates.md").exists()

        self.assertEqual(summary["original_rows"], 3)
        self.assertEqual(summary["classification_rows"], 1)
        self.assertEqual(summary["reference_candidate_rows"], 1)
        table_paths = [" / ".join(row["path_levels"]) for row in table["classification_rows"]]
        self.assertNotIn("术语和定义 / 核心数据", table_paths)
        self.assertEqual(table["classification_rows"][0]["row_id"], "current")
        self.assertFalse(table["classification_rows"][0]["needs_review"])
        self.assertEqual(table["classification_rows"][0]["review_reason"], "")
        self.assertEqual(candidates["reference_candidate_rows"][0]["row_id"], "candidate")
        self.assertEqual(tree["llm_model"], "test-model")
        node_paths = [node["path"] for node in tree["nodes"]]
        self.assertIn("基础资源 / 设备资源 / 硬件设备", node_paths)
        self.assertNotIn("基础资源 / 设备资源 / 软件设备", node_paths)
        self.assertNotIn("术语和定义 / 核心数据", node_paths)
        self.assertFalse(has_review_report)
        self.assertFalse(has_candidates_md)
        self.assertIn("- Reference candidate rows: 1", readme)


if __name__ == "__main__":
    unittest.main()
