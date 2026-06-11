from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_guangdong_failure import (
    CATEGORY_LLM_OUTPUT_PRESSURE,
    CATEGORY_NON_CATALOG,
    CATEGORY_QUOTE_MATCHING,
    CATEGORY_TABLE_HIERARCHY,
    CATEGORY_TEXT_STRUCTURE,
    analyze_failure_artifacts,
    classify_row,
    classify_validation_issue,
    normalized_contains,
    render_markdown,
)


class GuangdongFailureAnalysisTests(unittest.TestCase):
    def test_normalized_contains_matches_wrapped_source_text(self) -> None:
        source = "07 供应管\n- 47 -\n一级类别 二级类别 三级类别 四级类别 数据说明 建议级别\n理 管理"

        self.assertTrue(normalized_contains(source, "07 供应管理"))

    def test_classify_validation_issue_detects_non_catalog_content(self) -> None:
        issue = {
            "severity": "high",
            "path": ["影响事项", "重要民生保障"],
            "problem": "分类层级未出现在输入文档中：影响事项",
        }

        result = classify_validation_issue(issue, source_text="")

        self.assertEqual(result.primary_category, CATEGORY_NON_CATALOG)
        self.assertIn(CATEGORY_NON_CATALOG, result.categories)

    def test_classify_validation_issue_detects_text_structure_damage(self) -> None:
        issue = {
            "severity": "high",
            "path": ["5 药品供应", "07 供应管理", "001 订单信息"],
            "problem": "分类层级未出现在输入文档中：07 供应管理",
        }
        source = "5 药品供应\n07 供应管 001 订单信息 订单编号\n- 47 -\n理 管理 采购订单描述"

        result = classify_validation_issue(issue, source)

        self.assertEqual(result.primary_category, CATEGORY_TEXT_STRUCTURE)
        self.assertIn(CATEGORY_TEXT_STRUCTURE, result.categories)

    def test_classify_row_detects_quote_matching_fragility(self) -> None:
        row = {
            "path_levels": ["2 医疗服务（医院）", "01 临床服务", "008 医嘱执行"],
            "needs_review": True,
            "support_level": "structural",
            "description_source": "summarized",
            "evidence_quote": "008 医嘱执行 主医嘱 ID、医嘱分组 一般数据 3 级 执行时间、医嘱结束日期",
            "review_reason": "quote not found",
            "evidence_refs": [{"chunk_id": "doc_1_chunk_350"}],
        }
        source = "008 医嘱执行 主医嘱 ID、医嘱分组 一般数据 3 级\n- 41 -\n执行时间、医嘱结束日期"

        result = classify_row(row, source)

        self.assertEqual(result.primary_category, CATEGORY_QUOTE_MATCHING)
        self.assertIn(CATEGORY_QUOTE_MATCHING, result.categories)

    def test_classify_row_detects_table_hierarchy_inheritance_risk(self) -> None:
        row = {
            "path_levels": ["5 药品供应", "07 供应管理", "001 订单信息"],
            "needs_review": True,
            "support_level": "structural",
            "description_source": "quoted",
            "evidence_quote": "001 订单信息 订单编号 一般数据 2 级",
            "review_reason": "路径层级依赖表格结构推断与上下文继承，需人工核对层级归属。",
            "evidence_refs": [{"chunk_id": "doc_1_chunk_362"}],
        }

        result = classify_row(row, source_text="")

        self.assertEqual(result.primary_category, CATEGORY_TABLE_HIERARCHY)
        self.assertIn(CATEGORY_TABLE_HIERARCHY, result.categories)

    def test_analyze_failure_artifacts_counts_debug_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = root / "outputs"
            debug_dir = out_dir / "debug"
            debug_dir.mkdir(parents=True)
            source = root / "guangdong.txt"
            source.write_text("07 供应管\n- 47 -\n理 管理", encoding="utf-8")
            (out_dir / "rule_table.json").write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "path_levels": ["5 药品供应", "07 供应管理", "001 订单信息"],
                                "needs_review": True,
                                "support_level": "structural",
                                "description_source": "quoted",
                                "evidence_quote": "001 订单信息 订单编号 一般数据 2 级",
                                "review_reason": "路径层级依赖表格结构推断",
                                "evidence_refs": [{"chunk_id": "doc_1_chunk_362"}],
                            }
                        ],
                        "validation_issues": [
                            {
                                "severity": "high",
                                "path": ["5 药品供应", "07 供应管理", "001 订单信息"],
                                "problem": "分类层级未出现在输入文档中：07 供应管理",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (out_dir / "review_report.md").write_text("# Human Review Report\n", encoding="utf-8")
            (out_dir / "eval_report.md").write_text("debug_json_failure\nbatch 36: 1448.2s\n", encoding="utf-8")
            (debug_dir / "failed_row_batch_36.txt").write_text(
                "error=LLM JSON output failed validation after 2 attempt(s): Unterminated string",
                encoding="utf-8",
            )

            analysis = analyze_failure_artifacts(out_dir, source)

            self.assertEqual(analysis.total_rows, 1)
            self.assertEqual(analysis.needs_review_rows, 1)
            self.assertEqual(analysis.validation_issue_count, 1)
            self.assertEqual(analysis.debug_failure_count, 1)
            self.assertGreaterEqual(analysis.category_counts[CATEGORY_LLM_OUTPUT_PRESSURE], 1)

    def test_render_markdown_contains_counts_and_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = root / "outputs"
            out_dir.mkdir()
            source = root / "guangdong.txt"
            source.write_text("影响事项\n重要民生保障", encoding="utf-8")
            (out_dir / "rule_table.json").write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "path_levels": ["影响事项", "重要民生保障"],
                                "needs_review": True,
                                "support_level": "explicit",
                                "description_source": "quoted",
                                "evidence_quote": "重要民生保障",
                                "review_reason": "",
                                "evidence_refs": [{"chunk_id": "doc_1_chunk_115"}],
                            }
                        ],
                        "validation_issues": [
                            {
                                "severity": "high",
                                "path": ["影响事项", "重要民生保障"],
                                "problem": "分类层级未出现在输入文档中：影响事项",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (out_dir / "review_report.md").write_text("# Human Review Report\n", encoding="utf-8")
            (out_dir / "eval_report.md").write_text("# Eval Report\n", encoding="utf-8")

            analysis = analyze_failure_artifacts(out_dir, source)
            markdown = render_markdown(analysis)

            self.assertIn("Guangdong Failure Analysis", markdown)
            self.assertIn("Non-Catalog Content Entered Tree", markdown)
            self.assertIn("影响事项 / 重要民生保障", markdown)


if __name__ == "__main__":
    unittest.main()
