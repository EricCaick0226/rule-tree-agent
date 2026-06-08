from __future__ import annotations

import unittest

from src.io.rule_table_linker import (
    build_rule_table_links,
    render_rule_table_link_markdown,
)


class RuleTableLinkerTests(unittest.TestCase):
    def test_links_rows_by_shared_path_and_evidence_terms(self) -> None:
        current_rows = [
            {
                "row_id": "cur_1",
                "path_levels": ["主题资源类数据", "电子病历数据库", "临床诊疗"],
                "description": "临床诊疗相关数据。",
                "description_source": "summarized",
                "data_range_examples": ["门诊记录", "住院记录"],
            }
        ]
        reference_rows = [
            {
                "row_id": "ref_1",
                "path_levels": ["3.2 电子病历数据库", "3.2.2 临床诊疗"],
                "description": "证据不足，无法从当前文档确定",
                "description_source": "insufficient",
                "data_range_examples": ["住院记录"],
            },
            {
                "row_id": "ref_2",
                "path_levels": ["基础资源类数据", "法律法规"],
                "description": "法律法规资源。",
                "description_source": "quoted",
                "data_range_examples": ["法规名称"],
            },
        ]

        links = build_rule_table_links(current_rows, reference_rows, top_k=2, min_score=0.2)

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].current_row_id, "cur_1")
        self.assertEqual(links[0].matches[0].reference_row_id, "ref_1")
        self.assertGreaterEqual(links[0].matches[0].score, 0.2)
        self.assertIn("临床诊疗", links[0].matches[0].shared_terms)
        self.assertEqual(len(links[0].matches), 1)

    def test_markdown_report_summarizes_links(self) -> None:
        links = build_rule_table_links(
            [
                {
                    "row_id": "cur_1",
                    "path_levels": ["业务资源类数据", "医疗服务"],
                    "description": "医疗服务业务数据。",
                    "description_source": "quoted",
                    "data_range_examples": ["诊疗信息"],
                }
            ],
            [
                {
                    "row_id": "ref_1",
                    "path_levels": ["医疗服务", "诊疗信息"],
                    "description": "诊疗信息。",
                    "description_source": "quoted",
                    "data_range_examples": ["诊疗信息"],
                }
            ],
            top_k=1,
            min_score=0.2,
        )

        markdown = render_rule_table_link_markdown(links)

        self.assertIn("# Rule Table Link Report", markdown)
        self.assertIn("业务资源类数据 / 医疗服务", markdown)
        self.assertIn("医疗服务 / 诊疗信息", markdown)
        self.assertIn("shared:", markdown)


if __name__ == "__main__":
    unittest.main()
