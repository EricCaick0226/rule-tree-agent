from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import scripts.build_wst363_data_elements as builder
from src.io.wst363_data_elements import build_wst363_payload, parse_wst363_data_elements


class Wst363DataElementsTests(unittest.TestCase):
    def test_parses_data_element_fields_and_value_domain_refs(self) -> None:
        text = """
WS/T 363.10—2023
卫生健康信息数据元目录
第 10 部分：医学诊断
4.2 数据元专用属性
数据元标识符 DE05.01.001.00
数据元名称 西医疾病诊断代码
定义 患者所患疾病诊断在特定编码体系中的代码
WS/T 363.10—2023
2
数据元值的数据类型 S3
表示格式 AN..20
数据元允许值 WS/T 364.10 CV05.01.001 西医疾病诊断代码表
"""

        elements = parse_wst363_data_elements(text, source_path="part10.txt")

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["element_code"], "DE05.01.001.00")
        self.assertEqual(elements[0]["element_name"], "西医疾病诊断代码")
        self.assertEqual(elements[0]["definition"], "患者所患疾病诊断在特定编码体系中的代码")
        self.assertEqual(elements[0]["data_type"], "S3")
        self.assertEqual(elements[0]["display_format"], "AN..20")
        self.assertEqual(
            elements[0]["allowed_values"],
            "WS/T 364.10 CV05.01.001 西医疾病诊断代码表",
        )
        self.assertEqual(elements[0]["value_domain_refs"], ["WS/T 364.10:CV05.01.001"])
        self.assertEqual(elements[0]["source_path"], "part10.txt")

    def test_build_payload_identifies_standard_part_and_title(self) -> None:
        text = """
WS/T 363.2—2023
卫生健康信息数据元目录
第 2 部分：标识
数据元标识符 DE01.00.001.00
数据元名称 报告卡编号
定义 按照某一特定编码规则赋予报告卡的顺序号
数据元值的数据类型 S1
表示格式 AN..20
数据元允许值
"""

        payload = build_wst363_payload(text, source_path="part02.txt")

        self.assertEqual(payload["standard"], "WS/T 363.2—2023")
        self.assertEqual(payload["part"], "02")
        self.assertEqual(payload["title"], "标识")
        self.assertEqual(payload["element_count"], 1)
        self.assertEqual(payload["elements"][0]["element_name"], "报告卡编号")

    def test_real_wst363_text_files_are_parseable(self) -> None:
        cases = [
            ("data/input_docs/WST 363.2—2023 标识.txt", "02", 20),
            ("data/input_docs/WST 363.3—2023 人口学及社会经济学特征.txt", "03", 300),
            ("data/input_docs/WST 363.10—2023 医学诊断.txt", "10", 150),
        ]

        for path_text, expected_part, min_elements in cases:
            with self.subTest(path=path_text):
                path = Path(path_text)
                payload = build_wst363_payload(
                    path.read_text(encoding="utf-8"),
                    source_path=str(path),
                )

                self.assertEqual(payload["part"], expected_part)
                self.assertGreaterEqual(payload["element_count"], min_elements)
                self.assertFalse(
                    any("WS/T 363." in element["definition"] for element in payload["elements"])
                )

    def test_build_script_discovers_and_writes_all_real_wst363_parts(self) -> None:
        inputs = builder.discover_wst363_inputs(Path("data/input_docs"))

        self.assertEqual(sorted(inputs), [f"{part:02d}" for part in range(1, 18)])

        with TemporaryDirectory() as tmp:
            manifest = builder.build_library(
                inputs=inputs,
                output_dir=Path(tmp) / "wst363",
            )

            self.assertEqual(len(manifest["parts"]), 17)
            self.assertEqual(
                sorted(path.name for path in (Path(tmp) / "wst363").glob("part_*.json")),
                [f"part_{part:02d}.json" for part in range(1, 18)],
            )
            self.assertGreater(manifest["total_elements"], 1900)


if __name__ == "__main__":
    unittest.main()
