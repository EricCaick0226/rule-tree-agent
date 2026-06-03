import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DescriptionContextIndexPOCTests(unittest.TestCase):
    def test_script_writes_retrieval_pack_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            txt_path = tmp_path / "source.txt"
            rule_table_path = tmp_path / "rule_table.json"
            out_path = tmp_path / "description_context_index.json"
            txt_path.write_text(
                "\n".join(
                    [
                        "业务资源类数据：在具体业务处理过程中产生、使用和存储的数据。",
                        "表B.1 业务资源数据分类分级表",
                        "1公共卫生",
                        "01疾病控制",
                        "001传染病动态监测 疫源地消毒情况，机构消毒情况 统计数据 组织 严重危害 一般数据3级",
                        "002疾病监测 发病报告，病例信息 原始数据 个人 严重危害 一般数据3级",
                        "影响程度：严重危害是指数据泄露后可能影响个人权益。",
                    ]
                ),
                encoding="utf-8",
            )
            rule_table_path.write_text(
                json.dumps(
                    {
                        "classification_rows": [
                            {
                                "row_id": "row_0",
                                "path_levels": ["A、基础资源"],
                                "description": "A、基础资源",
                                "data_range_examples": [],
                                "recommended_grade": None,
                            },
                            {
                                "row_id": "row_1",
                                "path_levels": ["1公共卫生", "01疾病控制", "001传染病动态监测"],
                                "description": "疫源地消毒情况，机构消毒情况",
                                "data_range_examples": ["疫源地消毒情况，机构消毒情况"],
                                "recommended_grade": "一般数据3级",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/description_context_index_poc.py",
                    "--txt",
                    str(txt_path),
                    "--rule-table",
                    str(rule_table_path),
                    "--out",
                    str(out_path),
                    "--limit",
                    "1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["sampled_row_count"], 1)
            self.assertEqual(report["rows"][0]["row_id"], "row_1")
            self.assertTrue(report["rows"][0]["context_pack"]["primary_contexts"])
            self.assertTrue(report["rows"][0]["context_pack"]["excluded_contexts"])


if __name__ == "__main__":
    unittest.main()
