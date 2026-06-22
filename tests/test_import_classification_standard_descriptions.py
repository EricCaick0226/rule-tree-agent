from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openpyxl import Workbook

from scripts.import_classification_standard_descriptions import (
    import_descriptions,
    normalize_path_level,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_excel(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "分类"
    sheet.append(["填写须知", None, None, None, None, None, None])
    sheet.append(["一级分类", "二级分类", "三级分类", "四级分类", "五级分类", "推荐分级", "分类说明"])
    for row in rows:
        sheet.append(row)
    workbook.save(path)


class ImportClassificationStandardDescriptionsTests(unittest.TestCase):
    def test_normalize_path_level_strips_only_delimited_list_markers(self) -> None:
        self.assertEqual(normalize_path_level("A类资源"), "A类资源")
        self.assertEqual(normalize_path_level("1型糖尿病"), "1型糖尿病")
        self.assertEqual(normalize_path_level("2号楼"), "2号楼")
        self.assertEqual(normalize_path_level("A、基础资源"), "基础资源")
        self.assertEqual(normalize_path_level("1.基础资源"), "基础资源")

    def test_imports_description_for_exact_full_path_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            excel = root / "标准.xlsx"
            rule_table = root / "rule_table.json"
            report = root / "report.json"
            _write_excel(
                excel,
                [
                    [
                        "基础资源",
                        "服务范围与对象",
                        "患者",
                        "患者信息",
                        None,
                        "3级",
                        "患者信息包括患者身份识别和联系方式等基本资料。",
                    ]
                ],
            )
            _write_json(
                rule_table,
                {
                    "classification_rows": [
                        {
                            "row_id": "row_patient_info",
                            "path_levels": ["基础资源", "服务范围与对象", "患者", "患者信息"],
                            "description": "证据不足，无法从当前文档确定",
                        }
                    ]
                },
            )

            summary = import_descriptions(rule_table, excel, report)
            data = json.loads(rule_table.read_text(encoding="utf-8"))
            report_data = json.loads(report.read_text(encoding="utf-8"))

        row = data["classification_rows"][0]
        self.assertEqual(summary["excel_rows"], 1)
        self.assertEqual(summary["exact_path_matches"], 1)
        self.assertEqual(summary["descriptions_imported"], 1)
        self.assertEqual(row["description"], "患者信息包括患者身份识别和联系方式等基本资料。")
        self.assertEqual(row["description_source"], "classification_standard_excel")
        self.assertEqual(row["source_confidence"], "curated_answer")
        self.assertNotIn("description_evidence_quote", row)
        self.assertEqual(report_data["descriptions_imported"], 1)

    def test_skips_prefix_match_to_parent_reference_row(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            excel = root / "标准.xlsx"
            rule_table = root / "rule_table.json"
            _write_excel(
                excel,
                [
                    [
                        "基础资源",
                        "服务范围与对象",
                        "患者",
                        "患者信息",
                        None,
                        "3级",
                        "患者信息说明。",
                    ]
                ],
            )
            _write_json(
                rule_table,
                {
                    "classification_rows": [
                        {
                            "row_id": "row_patient",
                            "path_levels": ["基础资源", "服务范围与对象", "患者"],
                            "description": "证据不足，无法从当前文档确定",
                        }
                    ]
                },
            )

            summary = import_descriptions(rule_table, excel)
            data = json.loads(rule_table.read_text(encoding="utf-8"))

        row = data["classification_rows"][0]
        self.assertEqual(summary["exact_path_matches"], 0)
        self.assertEqual(summary["descriptions_imported"], 0)
        self.assertEqual(summary["new_rows_appended"], 1)
        self.assertEqual(summary["skipped_no_exact_reference_path"], 0)
        self.assertEqual(row["description"], "证据不足，无法从当前文档确定")
        self.assertEqual(
            data["classification_rows"][1]["path_levels"],
            ["基础资源", "服务范围与对象", "患者", "患者信息"],
        )

    def test_preserves_existing_stronger_description_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            excel = root / "标准.xlsx"
            rule_table = root / "rule_table.json"
            _write_excel(
                excel,
                [
                    ["基础资源", "服务范围与对象", "患者", "患者信息", None, "3级", "Excel 说明。"]
                ],
            )
            _write_json(
                rule_table,
                {
                    "classification_rows": [
                        {
                            "row_id": "row_patient_info",
                            "path_levels": ["基础资源", "服务范围与对象", "患者", "患者信息"],
                            "description": "原文证据说明。",
                            "description_source": "local_standard_quote",
                            "description_evidence_quote": "患者信息包括身份识别资料。",
                        }
                    ]
                },
            )

            summary = import_descriptions(rule_table, excel)
            data = json.loads(rule_table.read_text(encoding="utf-8"))

        row = data["classification_rows"][0]
        self.assertEqual(summary["exact_path_matches"], 1)
        self.assertEqual(summary["descriptions_imported"], 0)
        self.assertEqual(summary["skipped_existing_stronger_source"], 1)
        self.assertEqual(row["description"], "原文证据说明。")
        self.assertEqual(row["description_source"], "local_standard_quote")
        self.assertEqual(row["description_evidence_quote"], "患者信息包括身份识别资料。")

    def test_skips_duplicate_excel_paths_as_ambiguous(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            excel = root / "标准.xlsx"
            rule_table = root / "rule_table.json"
            _write_excel(
                excel,
                [
                    ["基础资源", "服务范围与对象", "患者", "患者信息", None, "3级", "说明 A。"],
                    ["基础资源", "服务范围与对象", "患者", "患者信息", None, "3级", "说明 B。"],
                ],
            )
            _write_json(
                rule_table,
                {
                    "classification_rows": [
                        {
                            "row_id": "row_patient_info",
                            "path_levels": ["基础资源", "服务范围与对象", "患者", "患者信息"],
                            "description": "证据不足，无法从当前文档确定",
                        }
                    ]
                },
            )

            summary = import_descriptions(rule_table, excel)
            data = json.loads(rule_table.read_text(encoding="utf-8"))

        row = data["classification_rows"][0]
        self.assertEqual(summary["ambiguous_excel_paths"], 1)
        self.assertEqual(summary["ambiguous_excel_rows"], 2)
        self.assertEqual(summary["skipped_ambiguous_excel_rows"], 2)
        self.assertEqual(summary["descriptions_imported"], 0)
        self.assertEqual(row["description"], "证据不足，无法从当前文档确定")

    def test_appends_unmatched_excel_row_as_curated_reference_row(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            excel = root / "标准.xlsx"
            rule_table = root / "rule_table.json"
            _write_excel(
                excel,
                [
                    [
                        "基础资源",
                        "服务范围与对象",
                        "患者",
                        "患者信息",
                        None,
                        "3级",
                        "患者信息包括患者身份识别和联系方式等基本资料。",
                    ]
                ],
            )
            _write_json(rule_table, {"classification_rows": []})

            summary = import_descriptions(rule_table, excel)
            data = json.loads(rule_table.read_text(encoding="utf-8"))

        self.assertEqual(summary["new_rows_appended"], 1)
        row = data["classification_rows"][0]
        self.assertTrue(row["row_id"].startswith("row_excel_"))
        self.assertEqual(row["path_levels"], ["基础资源", "服务范围与对象", "患者", "患者信息"])
        self.assertEqual(row["recommended_grade"], "3级")
        self.assertEqual(row["description"], "患者信息包括患者身份识别和联系方式等基本资料。")
        self.assertEqual(row["description_source"], "classification_standard_excel")
        self.assertEqual(row["source_confidence"], "curated_answer")
        self.assertEqual(row["row_source"], "classification_standard_excel")
        self.assertEqual(row["curation_status"], "classification_standard_excel_import")
        self.assertNotIn("description_evidence_quote", row)

    def test_append_mode_is_idempotent_by_normalized_full_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            excel = root / "标准.xlsx"
            rule_table = root / "rule_table.json"
            _write_excel(
                excel,
                [
                    [
                        "基础资源",
                        "服务范围与对象",
                        "患者",
                        "患者信息",
                        None,
                        "3级",
                        "患者信息包括患者身份识别和联系方式等基本资料。",
                    ]
                ],
            )
            _write_json(rule_table, {"classification_rows": []})

            first_summary = import_descriptions(rule_table, excel)
            second_summary = import_descriptions(rule_table, excel)
            data = json.loads(rule_table.read_text(encoding="utf-8"))

        self.assertEqual(first_summary["new_rows_appended"], 1)
        self.assertEqual(second_summary["new_rows_appended"], 0)
        self.assertEqual(len(data["classification_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
