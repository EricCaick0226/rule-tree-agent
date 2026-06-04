import unittest

from src.io.data_classification_profile import (
    build_description_query_terms,
    classify_content_type,
    classify_field_role,
    contains_grade_or_risk,
)


class DataClassificationProfileTests(unittest.TestCase):
    def test_classifies_common_standard_document_shapes(self) -> None:
        self.assertEqual(
            classify_content_type(
                header_text="类（1 位数字） 项（2 位数字） 目（3 位数字）",
                table_title="表A.1 基础资源分类目录（示例）",
                text="类（1 位数字） 项（2 位数字） 目（3 位数字）",
            ),
            "classification_catalog",
        )
        self.assertEqual(
            classify_content_type(
                header_text="类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别",
                table_title="表B.1 基础资源数据分类分级表",
                text="001患者信息 患者姓名 原始数据 个人 严重危害 一般数据3级",
            ),
            "classification_grading_table",
        )
        self.assertEqual(
            classify_content_type(
                header_text="",
                table_title="",
                text="按照第6章开展数据分类工作，并参考附录进行分级。",
            ),
            "rule_guidance",
        )

    def test_classifies_field_roles(self) -> None:
        self.assertEqual(classify_field_role("类"), "classification_path")
        self.assertEqual(classify_field_role("项"), "classification_path")
        self.assertEqual(classify_field_role("目"), "classification_path")
        self.assertEqual(classify_field_role("数据范围及示例"), "description_evidence")
        self.assertEqual(classify_field_role("数据加工程度"), "grading_factor")
        self.assertEqual(classify_field_role("影响对象"), "grading_factor")
        self.assertEqual(classify_field_role("影响程度"), "grading_factor")
        self.assertEqual(classify_field_role("数据级别"), "grade_result")
        self.assertEqual(classify_field_role("其他字段"), "metadata_or_unknown")

    def test_description_query_terms_exclude_grading_fields(self) -> None:
        row = {
            "path_levels": ["1公共卫生", "01疾病控制", "001传染病动态监测"],
            "data_range_examples": ["疫源地消毒情况，机构消毒情况"],
            "processing_degree": "统计数据",
            "impact_object": "组织",
            "impact_degree": "严重危害",
            "recommended_grade": "一般数据3级",
        }

        terms = build_description_query_terms(row)

        self.assertIn("001传染病动态监测", terms)
        self.assertIn("疫源地消毒情况，机构消毒情况", terms)
        self.assertNotIn("统计数据", terms)
        self.assertNotIn("组织", terms)
        self.assertNotIn("严重危害", terms)
        self.assertNotIn("一般数据3级", terms)

    def test_contains_grade_or_risk_uses_profile_boundary(self) -> None:
        self.assertTrue(contains_grade_or_risk("影响程度：严重危害"))
        self.assertTrue(contains_grade_or_risk("数据级别为一般数据3级"))
        self.assertFalse(contains_grade_or_risk("患者姓名、生日、性别"))

    def test_builds_row_evidence_pack_with_clean_description_sources(self) -> None:
        from src.io.data_classification_profile import build_row_evidence_pack

        row = {
            "path_levels": ["1公共卫生", "01疾病控制", "001传染病动态监测"],
            "data_range_examples": ["疫源地消毒情况，机构消毒情况"],
            "processing_degree": "统计数据",
            "impact_object": "组织",
            "impact_degree": "严重危害",
            "recommended_grade": "一般数据3级",
        }
        context_pack = {
            "primary_contexts": [
                {
                    "unit_id": "table_row_unit_1",
                    "kind": "table_row_unit",
                    "text": "001传染病动态监测 疫源地消毒情况，机构消毒情况 统计数据 组织 严重危害 一般数据3级",
                    "line_start": 10,
                    "line_end": 10,
                    "score": 90,
                }
            ],
            "definition_contexts": [
                {
                    "unit_id": "definition_unit_1",
                    "kind": "definition_unit",
                    "text": "业务资源类数据：在具体业务处理过程中产生、使用和存储的数据。",
                    "line_start": 3,
                    "line_end": 3,
                    "score": 50,
                }
            ],
            "sibling_contexts": [],
            "excluded_contexts": [
                {
                    "unit_id": "negative_unit_1",
                    "kind": "negative_unit",
                    "text": "影响程度：严重危害是指数据泄露后可能影响个人权益。",
                    "line_start": 20,
                    "line_end": 20,
                    "score": 10,
                }
            ],
            "retrieval_warnings": ["excluded_grade_or_risk_context"],
        }

        evidence_pack = build_row_evidence_pack(row, context_pack)

        description_text = "\n".join(source["text"] for source in evidence_pack["description_sources"])
        grading_text = "\n".join(source["text"] for source in evidence_pack["grading_sources"])
        excluded_text = "\n".join(source["text"] for source in evidence_pack["excluded_sources"])

        self.assertIn("001传染病动态监测", description_text)
        self.assertIn("疫源地消毒情况，机构消毒情况", description_text)
        self.assertIn("业务资源类数据", description_text)
        self.assertNotIn("一般数据3级", description_text)
        self.assertIn("统计数据", grading_text)
        self.assertIn("一般数据3级", grading_text)
        self.assertIn("影响程度", excluded_text)
        self.assertIn("excluded_grade_or_risk_context", evidence_pack["warnings"])


if __name__ == "__main__":
    unittest.main()
