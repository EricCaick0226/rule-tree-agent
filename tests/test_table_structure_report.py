from dataclasses import FrozenInstanceError
import unittest

from src.core.agent_state import DocumentChunk
from src.io.table_segmenter import TableSegment, segment_table_chunks_for_row_extraction
from src.io.table_structure_report import (
    TableStructureItem,
    TableStructureReport,
    build_table_structure_report,
    filter_reviewable_table_structure_items,
    render_table_structure_markdown,
    report_to_dict,
)


class TableStructureReportTests(unittest.TestCase):
    def _segments(self, text: str, section_title: str = "附录 A / 表 A.1 数据分类分级表"):
        chunk = DocumentChunk(
            chunk_id="doc_1_chunk_1",
            doc_id="doc_1",
            doc_name="sample.txt",
            section_title=section_title,
            text=text,
            position=1,
            line_start=10,
            line_end=10 + len(text.splitlines()) - 1,
        )
        return segment_table_chunks_for_row_extraction(
            [chunk],
            block_signals={"doc_1_chunk_1": {"block_signal": "table_like"}},
            max_chars=1000,
        )

    def test_builds_classification_grading_table_report_item(self):
        header = "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"
        segments = self._segments(
            "\n".join(
                [
                    header,
                    "业务 数据 个人信息 身份证号 原始数据 个人 严重危害 一般数据3级",
                ]
            )
        )

        report = build_table_structure_report(segments)

        self.assertEqual(report.total_segments, 1)
        self.assertEqual(report.segmentation_mode, "all_nonempty_chunks_as_table_candidates")
        item = report.items[0]
        self.assertEqual(item.segment_id, "doc_1_chunk_1_seg_1")
        self.assertEqual(item.section_title, "附录 A / 表 A.1 数据分类分级表")
        self.assertEqual(item.table_title, "表 A.1 数据分类分级表")
        self.assertEqual(item.hierarchy_header, header)
        self.assertEqual(item.content_type, "classification_grading_table")
        self.assertEqual(item.line_span, {"start": 11, "end": 11})
        self.assertIn({"field": "类", "role": "classification_path"}, item.field_roles)
        self.assertIn({"field": "数据范围及示例", "role": "description_evidence"}, item.field_roles)
        self.assertIn({"field": "影响程度", "role": "grading_factor"}, item.field_roles)
        self.assertIn({"field": "数据级别", "role": "grade_result"}, item.field_roles)
        self.assertIn("detected table title", item.review_notes)
        self.assertIn("detected hierarchy header", item.review_notes)

    def test_reports_flattened_row_hints_and_review_notes(self):
        segments = self._segments(
            "2.5 药品供应 2.5.7 供应管理",
            section_title="正文",
        )

        report = build_table_structure_report(segments)

        self.assertEqual(report.total_segments, 1)
        item = report.items[0]
        self.assertEqual(item.content_type, "unknown")
        self.assertEqual(item.flattened_row_hints_count, 1)
        self.assertIn("has flattened parent-child code lines", item.review_notes)
        self.assertIn("missing header text", item.review_notes)
        self.assertIn("unknown content type", item.review_notes)

    def test_report_value_objects_are_frozen(self):
        item = TableStructureItem(
            segment_id="seg_1",
            source_chunk_id="chunk_1",
            doc_name="sample.txt",
            section_title="正文",
            table_title="",
            hierarchy_header="",
            content_type="unknown",
            line_span={"start": None, "end": None},
            field_roles=[],
            flattened_row_hints_count=0,
            review_notes=[],
        )
        report = TableStructureReport(total_segments=1, items=[item])

        with self.assertRaises(FrozenInstanceError):
            item.content_type = "classification_grading_table"
        with self.assertRaises(FrozenInstanceError):
            report.total_segments = 2

    def test_malformed_structure_context_variants_fall_back_to_header_text(self):
        header = "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"
        for context in [None, {}, "bad-context"]:
            with self.subTest(context=context):
                segment = TableSegment(
                    segment_id="doc_1_chunk_1_seg_1",
                    source_chunk_id="doc_1_chunk_1",
                    doc_name="sample.txt",
                    section_title="正文",
                    text="业务 数据 个人信息 身份证号 原始数据 个人 严重危害 一般数据3级",
                    position=1,
                    page_number=None,
                    line_start=11,
                    line_end=11,
                    source_method="text",
                    source_warning="",
                    block_signal="table_like",
                    header_text=header,
                )
                object.__setattr__(segment, "structure_context", context)

                report = build_table_structure_report([segment])
                markdown = render_table_structure_markdown(report)

                self.assertEqual(report.items[0].hierarchy_header, header)
                self.assertIn("类 -> classification_path", markdown)
                self.assertIn("数据级别 -> grade_result", markdown)

    def test_report_outputs_are_reviewable(self):
        header = "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别"
        report = build_table_structure_report(
            self._segments(
                "\n".join(
                    [
                        header,
                        "业务 数据 个人信息 身份证号 原始数据 个人 严重危害 一般数据3级",
                    ]
                )
            )
        )

        report_dict = report_to_dict(report)
        markdown = render_table_structure_markdown(report)

        self.assertEqual(report_dict["total_segments"], 1)
        self.assertEqual(report_dict["segmentation_mode"], "all_nonempty_chunks_as_table_candidates")
        self.assertIn(
            {"field": "类", "role": "classification_path"},
            report_dict["items"][0]["field_roles"],
        )
        self.assertIn(
            {"field": "数据级别", "role": "grade_result"},
            report_dict["items"][0]["field_roles"],
        )
        self.assertIn("# Table Structure Report", markdown)
        self.assertIn("- Segmentation mode: all_nonempty_chunks_as_table_candidates", markdown)
        self.assertIn("field_roles:", markdown)
        self.assertIn("类 -> classification_path", markdown)
        self.assertIn("数据级别 -> grade_result", markdown)

    def test_markdown_uses_placeholders_for_empty_fields_and_notes(self):
        item = TableStructureItem(
            segment_id="seg_1",
            source_chunk_id="chunk_1",
            doc_name="sample.txt",
            section_title="正文",
            table_title="",
            hierarchy_header="",
            content_type="unknown",
            line_span={"start": None, "end": None},
            field_roles=[],
            flattened_row_hints_count=0,
            review_notes=[],
        )

        markdown = render_table_structure_markdown(TableStructureReport(total_segments=1, items=[item]))

        self.assertIn("field_roles:\n  - (none)", markdown)
        self.assertIn("review_notes:\n  - (none)", markdown)

    def test_filtered_items_exclude_title_only_unknown_segments(self):
        title_only_unknown = TableStructureItem(
            segment_id="title_only",
            source_chunk_id="chunk_1",
            doc_name="sample.txt",
            section_title="附录 A / 表A.1 数据分类目录(续)",
            table_title="表A.1 数据分类目录(续)",
            hierarchy_header="",
            content_type="unknown",
            line_span={"start": 1, "end": 1},
            field_roles=[],
            flattened_row_hints_count=0,
            review_notes=["detected table title", "missing header text", "unknown content type"],
        )
        catalog = TableStructureItem(
            segment_id="catalog",
            source_chunk_id="chunk_2",
            doc_name="sample.txt",
            section_title="附录 A / 表A.1 数据分类目录",
            table_title="表A.1 数据分类目录",
            hierarchy_header="资源属性 类 项 目",
            content_type="classification_catalog",
            line_span={"start": 2, "end": 10},
            field_roles=[],
            flattened_row_hints_count=0,
            review_notes=["detected hierarchy header"],
        )
        flattened = TableStructureItem(
            segment_id="flattened",
            source_chunk_id="chunk_3",
            doc_name="sample.txt",
            section_title="正文",
            table_title="",
            hierarchy_header="",
            content_type="unknown",
            line_span={"start": 11, "end": 11},
            field_roles=[],
            flattened_row_hints_count=1,
            review_notes=["has flattened parent-child code lines"],
        )
        report = TableStructureReport(
            total_segments=3,
            items=[title_only_unknown, catalog, flattened],
        )

        filtered = filter_reviewable_table_structure_items(report)

        self.assertEqual([item.segment_id for item in filtered], ["catalog", "flattened"])


if __name__ == "__main__":
    unittest.main()
