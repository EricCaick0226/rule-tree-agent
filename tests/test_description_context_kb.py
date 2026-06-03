import unittest
from unittest.mock import patch

from src.core.agent_state import AgentState, ClassificationRow, SourceDocument
from src.llm.task_utils import load_prompt
from src.steps.description_context_kb import (
    build_context_units,
    build_row_query_terms,
    enhance_descriptions_with_context,
    flag_description_quality,
    generate_description_candidates,
    generate_description_candidates_batched,
    retrieve_contexts,
)


class DescriptionContextKBTests(unittest.TestCase):
    def test_flags_leaf_and_data_range_descriptions(self) -> None:
        row = {
            "path_levels": ["业务资源", "公共卫生", "免疫规划监测"],
            "description": "免疫规划监测",
            "data_range_examples": ["疫苗接种记录"],
        }

        self.assertIn("description_equals_leaf", flag_description_quality(row))

        row["description"] = "疫苗接种记录"

        self.assertIn("description_duplicates_data_range", flag_description_quality(row))

    def test_flags_missing_and_label_like_descriptions(self) -> None:
        row = {
            "path_levels": ["基础资源", "财务资源", "预算管理"],
            "description": "证据不足，无法从当前文档确定",
            "data_range_examples": [],
        }

        self.assertIn("description_insufficient", flag_description_quality(row))

        row["description"] = "预算"

        self.assertIn("description_label_like", flag_description_quality(row))

    def test_retrieves_context_by_path_and_data_terms(self) -> None:
        text = "\n".join(
            [
                "业务资源",
                "公共卫生包括免疫规划监测、疾病监测等数据。",
                "免疫规划监测涉及疫苗接种记录。",
                "其他内容。",
            ]
        )
        units = build_context_units(text, window_lines=2)
        row = {
            "path_levels": ["业务资源", "公共卫生", "免疫规划监测"],
            "data_range_examples": ["疫苗接种记录"],
            "recommended_grade": "3级",
        }

        terms = build_row_query_terms(row)
        contexts = retrieve_contexts(units, terms, top_k=2)

        self.assertTrue(contexts)
        self.assertIn("免疫规划监测", contexts[0]["text"])
        self.assertGreater(contexts[0]["score"], 0)
        self.assertLessEqual(len(contexts), 2)

    def test_merges_wrapped_table_rows_into_context_units(self) -> None:
        text = "\n".join(
            [
                "003医疗指标类数据查",
                "询接口",
                "围绕患者健康的指标数据，如：BMI、健康风险、多重",
                "用药、营养不良等评估",
                "统计数据 个人 严重危害 一般数据3级",
                "004过程类数据查询接",
                "口",
                "医疗过程类数据；挂号、就诊、收费、领药；入院登记、",
                "转床转科、出院等流程数据查询接口",
                "原始数据 个人 严重危害 一般数据3级",
                "表B.1 基础资源数据分类分级表(续)",
            ]
        )
        units = build_context_units(text, window_lines=3)
        row = {
            "path_levels": ["10卫生健康信息化", "04数据接口", "004过程类数据查询接口"],
            "data_range_examples": [
                "医疗过程类数据；挂号、就诊、收费、领药；入院登记、转床转科、出院等流程数据查询接口"
            ],
            "processing_degree": "原始数据",
            "impact_object": "个人",
            "impact_degree": "严重危害",
            "recommended_grade": "一般数据3级",
        }

        contexts = retrieve_contexts(units, build_row_query_terms(row), top_k=1)

        self.assertTrue(contexts)
        self.assertIn("004过程类数据查询接", contexts[0]["text"])
        self.assertIn("转床转科、出院等流程数据查询接口", contexts[0]["text"])
        self.assertIn("原始数据 个人 严重危害 一般数据3级", contexts[0]["text"])

    def test_generates_description_candidates_from_retrieved_context(self) -> None:
        report_row = {
            "row_id": "row_1",
            "path": "业务资源 / 公共卫生 / 免疫规划监测",
            "current_description": "免疫规划监测",
            "description_quality_flags": ["description_equals_leaf"],
            "query_terms": ["业务资源", "公共卫生", "免疫规划监测"],
            "retrieved_contexts": [
                {
                    "unit_id": "txt_lines_1_2",
                    "line_start": 1,
                    "line_end": 2,
                    "text": "公共卫生包括免疫规划监测、疾病监测等数据。",
                }
            ],
        }

        def fake_call_llm_json(**kwargs):
            self.assertEqual(kwargs["prompt_file"], "generate_classification_descriptions_prompt.md")
            self.assertEqual(kwargs["payload"]["rows"][0]["row_id"], "row_1")
            self.assertIn("公共卫生包括免疫规划监测", kwargs["payload"]["rows"][0]["retrieved_contexts"][0]["text"])
            return (
                {
                    "description_candidates": [
                        {
                            "row_id": "row_1",
                            "proposed_description": "公共卫生下用于免疫规划监测相关业务的数据分类项。",
                            "description_source": "summarized",
                            "description_evidence_quote": "公共卫生包括免疫规划监测、疾病监测等数据。",
                            "needs_review": True,
                            "review_reason": "基于检索上下文总结生成，需要人工确认。",
                        }
                    ]
                },
                "raw",
            )

        with patch("src.steps.description_context_kb.call_llm_json", side_effect=fake_call_llm_json):
            candidates, raw_response = generate_description_candidates(object(), [report_row])

        self.assertEqual(raw_response, "raw")
        self.assertEqual(candidates[0]["row_id"], "row_1")
        self.assertEqual(candidates[0]["description_source"], "summarized")
        self.assertTrue(candidates[0]["needs_review"])

    def test_batches_description_candidate_generation(self) -> None:
        rows = [
            {
                "row_id": f"row_{index}",
                "path": f"业务资源 / 项目{index}",
                "current_description": f"项目{index}",
                "description_quality_flags": ["description_equals_leaf"],
                "query_terms": [f"项目{index}"],
                "retrieved_contexts": [{"text": f"项目{index}相关数据。"}],
            }
            for index in range(5)
        ]
        batch_lengths: list[int] = []

        def fake_generate(_llm_client, batch):
            batch_lengths.append(len(batch))
            return (
                [
                    {
                        "row_id": row["row_id"],
                        "proposed_description": f"{row['path']}相关业务数据。",
                        "description_source": "summarized",
                        "description_evidence_quote": row["retrieved_contexts"][0]["text"],
                        "needs_review": True,
                        "review_reason": "基于检索上下文总结生成，需要人工确认。",
                    }
                    for row in batch
                ],
                f"raw_{len(batch_lengths)}",
            )

        with patch("src.steps.description_context_kb.generate_description_candidates", side_effect=fake_generate):
            candidates, raw_response = generate_description_candidates_batched(object(), rows, batch_size=2)

        self.assertEqual(batch_lengths, [2, 2, 1])
        self.assertEqual(len(candidates), 5)
        self.assertIn("raw_1", raw_response)
        self.assertIn("raw_3", raw_response)

    def test_description_generation_prompt_excludes_grade_conclusions(self) -> None:
        prompt = load_prompt("generate_classification_descriptions_prompt.md")

        self.assertIn("不要写推荐分级", prompt)
        self.assertIn("不要写“定级为几级”", prompt)
        self.assertIn("分类说明只解释分类项是什么", prompt)
        self.assertIn("不要写影响程度", prompt)
        self.assertIn("不要写危害后果", prompt)
        self.assertIn("优先改写原文中的分类名和数据范围", prompt)
        self.assertIn("不要把“原始数据/统计数据”等表字段作为说明主体", prompt)

    def test_enhancement_step_is_disabled_by_default(self) -> None:
        state = AgentState(
            task="test",
            documents=[SourceDocument("doc_1", "source.txt", "source.txt", "患者信息 患者姓名")],
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["患者信息"],
                    description="患者信息",
                    data_range_examples=["患者姓名"],
                )
            ],
        )

        with patch("src.steps.description_context_kb.generate_description_candidates") as generate:
            result = enhance_descriptions_with_context(state, object(), output_dir="outputs")

        generate.assert_not_called()
        self.assertIs(result, state)
        self.assertEqual(state.classification_rows[0].description, "患者信息")
        self.assertEqual(state.step_traces[-1].status, "skipped")

    def test_enhancement_step_updates_weak_descriptions_with_context_candidate(self) -> None:
        state = AgentState(
            task="test",
            documents=[
                SourceDocument(
                    "doc_1",
                    "source.txt",
                    "source.txt",
                    "001患者信息 患者姓名，生日，性别，民族 原始数据 个人 严重危害 一般数据3级",
                )
            ],
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["1服务范围与对象", "01患者", "001患者信息"],
                    description="患者姓名，生日，性别，民族",
                    description_source="quoted",
                    data_range_examples=["患者姓名，生日，性别，民族"],
                    evidence_quote="001患者信息 患者姓名，生日，性别，民族 原始数据 个人 严重危害 一般数据3级",
                )
            ],
        )

        def fake_generate(_llm_client, rows, batch_size=20):
            self.assertEqual(batch_size, 2)
            self.assertEqual(rows[0]["row_id"], "row_1")
            self.assertTrue(rows[0]["retrieved_contexts"])
            return (
                [
                    {
                        "row_id": "row_1",
                        "proposed_description": "面向患者群体的基础人口统计学信息。",
                        "description_source": "summarized",
                        "description_evidence_quote": "001患者信息 患者姓名，生日，性别，民族 原始数据 个人 严重危害 一般数据3级",
                        "needs_review": True,
                        "review_reason": "基于检索上下文总结生成，需要人工确认。",
                    }
                ],
                "raw",
            )

        with patch.dict(
            "os.environ",
            {
                "DESCRIPTION_CONTEXT_ENABLED": "true",
                "DESCRIPTION_CONTEXT_LIMIT": "5",
                "DESCRIPTION_CONTEXT_BATCH_SIZE": "2",
            },
        ):
            with patch("src.steps.description_context_kb.generate_description_candidates_batched", side_effect=fake_generate):
                result = enhance_descriptions_with_context(state, object(), output_dir="outputs")

        row = result.classification_rows[0]
        self.assertEqual(row.description, "面向患者群体的基础人口统计学信息。")
        self.assertEqual(row.description_source, "summarized")
        self.assertEqual(
            row.description_evidence_quote,
            "001患者信息 患者姓名，生日，性别，民族 原始数据 个人 严重危害 一般数据3级",
        )
        self.assertTrue(row.needs_review)
        self.assertTrue(row.evidence_refs)
        self.assertIn("基于检索上下文总结生成", row.review_reason)
        self.assertEqual(state.step_traces[-1].status, "success")


if __name__ == "__main__":
    unittest.main()
