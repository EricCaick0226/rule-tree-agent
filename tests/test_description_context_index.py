import unittest

from src.steps.description_context_index import (
    build_description_context_index,
    retrieve_description_context_pack,
)


class DescriptionContextIndexTests(unittest.TestCase):
    def test_builds_typed_units_for_definition_table_rows_and_sibling_groups(self) -> None:
        text = "\n".join(
            [
                "业务资源类数据：在具体业务处理过程中产生、使用和存储的数据。",
                "表B.1 业务资源数据分类分级表",
                "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别",
                "1公共卫生",
                "01疾病控制",
                "001传染病动态监测 疫源地消毒情况，机构消毒情况 统计数据 组织 严重危害 一般数据3级",
                "002疾病监测 发病报告，病例信息 原始数据 个人 严重危害 一般数据3级",
                "003免疫规划 疫苗接种记录 原始数据 个人 严重危害 一般数据3级",
                "影响程度：严重危害是指数据泄露后可能影响个人权益。",
            ]
        )

        units = build_description_context_index(text)
        kinds = {unit["kind"] for unit in units}

        self.assertIn("definition_unit", kinds)
        self.assertIn("table_row_unit", kinds)
        self.assertIn("sibling_group_unit", kinds)
        self.assertTrue(any(unit["contains_grade_signal"] for unit in units))
        self.assertTrue(any(unit["contains_description_signal"] for unit in units))

    def test_retrieves_context_pack_with_primary_definition_siblings_and_excluded_noise(self) -> None:
        text = "\n".join(
            [
                "业务资源类数据：在具体业务处理过程中产生、使用和存储的数据。",
                "表B.1 业务资源数据分类分级表",
                "1公共卫生",
                "01疾病控制",
                "001传染病动态监测 疫源地消毒情况，机构消毒情况 统计数据 组织 严重危害 一般数据3级",
                "002疾病监测 发病报告，病例信息 原始数据 个人 严重危害 一般数据3级",
                "003免疫规划 疫苗接种记录 原始数据 个人 严重危害 一般数据3级",
                "影响程度：严重危害是指数据泄露后可能影响个人权益。",
            ]
        )
        row = {
            "path_levels": ["1公共卫生", "01疾病控制", "001传染病动态监测"],
            "data_range_examples": ["疫源地消毒情况，机构消毒情况"],
            "processing_degree": "统计数据",
            "impact_object": "组织",
            "impact_degree": "严重危害",
            "recommended_grade": "一般数据3级",
        }

        pack = retrieve_description_context_pack(row, build_description_context_index(text), top_k=5)

        self.assertTrue(pack["primary_contexts"])
        self.assertIn("001传染病动态监测", pack["primary_contexts"][0]["text"])
        self.assertTrue(pack["definition_contexts"])
        self.assertIn("业务资源类数据", pack["definition_contexts"][0]["text"])
        self.assertTrue(pack["sibling_contexts"])
        self.assertIn("002疾病监测", pack["sibling_contexts"][0]["text"])
        self.assertTrue(pack["excluded_contexts"])
        self.assertIn("影响程度", pack["excluded_contexts"][0]["text"])
        self.assertIn("excluded_grade_or_risk_context", pack["retrieval_warnings"])

    def test_ignores_dates_as_table_rows_and_detects_bulleted_definitions(self) -> None:
        text = "\n".join(
            [
                "2025-07-01实施",
                "a) 基础资源类数据：信息资源中最基础的数据。",
                "001患者信息 患者姓名，生日，性别，民族 原始数据 个人 严重危害 一般数据3级",
            ]
        )

        units = build_description_context_index(text)
        table_rows = [unit for unit in units if unit["kind"] == "table_row_unit"]
        definitions = [unit for unit in units if unit["kind"] == "definition_unit"]

        self.assertEqual(len(table_rows), 1)
        self.assertNotIn("2025-07-01", table_rows[0]["text"])
        self.assertTrue(definitions)
        self.assertIn("基础资源类数据", definitions[0]["text"])

    def test_merges_wrapped_table_item_rows(self) -> None:
        text = "\n".join(
            [
                "001患者信息“",
                "患者姓名，生日，性别，民族 原始数据 个人 严重危害 一般数据3级",
                "002患者敏感信息“",
                "患者身份证号，联系方式，住址 原始数据 个人 特别严重危害 一般数据4级",
                "02健康人",
                "001个人信息 姓名，生日，性别，民族 原始数据 个人 严重危害 一般数据3级",
            ]
        )

        units = build_description_context_index(text)
        table_rows = [unit for unit in units if unit["kind"] == "table_row_unit"]

        self.assertEqual(len(table_rows), 3)
        self.assertIn("患者姓名，生日，性别，民族", table_rows[0]["text"])
        self.assertIn("患者身份证号，联系方式，住址", table_rows[1]["text"])
        self.assertNotIn("02健康人", table_rows[1]["text"])
        self.assertIn("001个人信息", table_rows[2]["text"])


if __name__ == "__main__":
    unittest.main()
