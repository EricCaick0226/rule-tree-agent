from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.io.wps_txt_cleaner import clean_wps_txt_file, clean_wps_txt_text, write_review_json


class WpsTxtCleanerTests(unittest.TestCase):
    def cleaned_lines(self, text: str) -> list[str]:
        return clean_wps_txt_text(text).text.splitlines()

    def test_merges_single_character_chinese_lines(self) -> None:
        result = clean_wps_txt_text(
            "患\n"
            "者\n"
            "信\n"
            "息\n"
            "\n"
            "表A.1 基础资源分类目录\n"
        )

        self.assertEqual(result.text.splitlines(), ["患者信息", "", "表A.1 基础资源分类目录"])
        self.assertEqual(result.stats["merged_single_char_lines"], 1)
        self.assertEqual(result.mapping[0].source_line_start, 1)
        self.assertEqual(result.mapping[0].source_line_end, 4)
        self.assertEqual(result.mapping[0].transform, "merge_single_char")

    def test_removes_page_footers_and_standalone_page_numbers(self) -> None:
        result = clean_wps_txt_text(
            "表A.1 基础资源分类目录\n"
            "- 12 -\n"
            "13\n"
            "1 服务范围与对象 01 患者\n"
        )

        self.assertEqual(
            result.text.splitlines(),
            ["表A.1 基础资源分类目录", "1 服务范围与对象 01 患者"],
        )
        self.assertEqual(result.stats["removed_page_noise_lines"], 2)

    def test_preserves_structural_lines(self) -> None:
        lines = self.cleaned_lines(
            "附 录 A\n"
            "基础资源分类\n"
            "表A.1 基础资源分类目录\n"
            "续表A.1 基础资源分类目录\n"
            "1 服务范围与对象 01 患者\n"
        )

        self.assertEqual(
            lines,
            [
                "附 录 A",
                "基础资源分类",
                "表A.1 基础资源分类目录",
                "续表A.1 基础资源分类目录",
                "1 服务范围与对象 01 患者",
            ],
        )

    def test_merges_wrapped_table_row_continuation(self) -> None:
        result = clean_wps_txt_text(
            "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别\n"
            "1 服务范围与对象 01 患者 001 患者信息\n"
            "患者姓名、出生日期、身份证件号码\n"
            "原始数据 个人 严重危害 一般数据3级\n"
            "002 就诊信息 门诊号、住院号\n"
        )

        self.assertEqual(
            result.text.splitlines(),
            [
                "类 项 目 数据范围及示例 数据加工程度 影响对象 影响程度 数据级别",
                "1 服务范围与对象 01 患者 001 患者信息 患者姓名、出生日期、身份证件号码 原始数据 个人 严重危害 一般数据3级",
                "002 就诊信息 门诊号、住院号",
            ],
        )
        self.assertEqual(result.stats["merged_wrapped_rows"], 1)

    def test_does_not_merge_normal_numbered_headings_into_previous_row(self) -> None:
        lines = self.cleaned_lines(
            "1 范围\n"
            "本文件规定了数据分类要求。\n"
            "2 规范性引用文件\n"
            "下列文件中的内容构成本文件条款。\n"
        )

        self.assertEqual(
            lines,
            [
                "1 范围",
                "本文件规定了数据分类要求。",
                "2 规范性引用文件",
                "下列文件中的内容构成本文件条款。",
            ],
        )

    def test_does_not_merge_normal_paragraph_or_title_like_lines(self) -> None:
        result = clean_wps_txt_text(
            "数据安全管理要求\n"
            "本文件规定了卫生健康数据分类分级要求\n"
            "发布单位：上海市卫生健康委员会\n"
        )

        self.assertEqual(
            result.text.splitlines(),
            [
                "数据安全管理要求",
                "本文件规定了卫生健康数据分类分级要求",
                "发布单位：上海市卫生健康委员会",
            ],
        )
        self.assertEqual(result.stats["merged_wrapped_sentences"], 0)

    def test_does_not_change_business_words(self) -> None:
        result = clean_wps_txt_text(
            "001 患者信息\n"
            "技资管理、患者姓名\n"
        )

        self.assertIn("技资管理", result.text)
        self.assertNotIn("投资管理", result.text)

    def test_file_helpers_create_parent_dirs_and_write_final_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_path = base / "input.txt"
            output_path = base / "nested" / "cleaned.txt"
            review_path = base / "review" / "review.json"
            input_path.write_text("患\n者\n", encoding="utf-8")

            result = clean_wps_txt_file(input_path, output_path, review_path)

            self.assertEqual(result.text, "患者")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "患者\n")
            review_text = review_path.read_text(encoding="utf-8")
            self.assertTrue(review_text.endswith("\n"))
            self.assertEqual(json.loads(review_text)["stats"]["merged_single_char_lines"], 1)

    def test_write_review_json_creates_parent_dir_and_final_newline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = clean_wps_txt_text("表A.1 基础资源分类目录\n")
            review_path = Path(temp_dir) / "nested" / "review.json"

            write_review_json(result, review_path)

            review_text = review_path.read_text(encoding="utf-8")
            self.assertTrue(review_text.endswith("\n"))
            self.assertEqual(json.loads(review_text)["stats"]["removed_page_noise_lines"], 0)


if __name__ == "__main__":
    unittest.main()
