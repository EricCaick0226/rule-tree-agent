from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from src.core.agent_state import AgentState, DocumentChunk
from src.steps.classification_row_extractor import extract_classification_rows_with_llm


def make_state() -> AgentState:
    chunk = DocumentChunk(
        chunk_id="doc_1_chunk_1",
        doc_id="doc_1",
        doc_name="policy.txt",
        section_title="分类表",
        text="一级分类 二级分类 三级分类 推荐分级 分类说明\n基础资源 服务范围与对象 患者 3级 患者信息包括身份识别资料",
        position=1,
        source_method="text",
        line_start=1,
        line_end=2,
    )
    state = AgentState(task="test", chunks=[chunk])
    state.block_signals = {
        "doc_1_chunk_1": {
            "block_signal": "table_like",
            "reason": "表格化分类分级行",
            "confidence": 0.9,
            "needs_review": False,
            "review_reason": "",
        }
    }
    return state


class ClassificationRowExtractorTests(unittest.TestCase):
    def test_extracts_rows_with_refs_and_insufficient_description_policy(self) -> None:
        state = make_state()

        def fake_call_llm_json(**kwargs):
            return (
                {
                    "classification_rows": [
                        {
                            "path_levels": ["基础资源", "服务范围与对象", "患者"],
                            "recommended_grade": "3级",
                            "description": "患者信息包括身份识别资料",
                            "description_source": "quoted",
                            "description_evidence_quote": "患者信息包括身份识别资料",
                            "evidence_quote": "基础资源 服务范围与对象 患者 3级 患者信息包括身份识别资料",
                            "evidence_chunk_ids": ["doc_1_chunk_1"],
                            "support_level": "explicit",
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                            "status": "evidence_supported",
                        }
                    ]
                },
                "raw",
            )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.steps.classification_row_extractor.call_llm_json", side_effect=fake_call_llm_json):
                result = extract_classification_rows_with_llm(state, object(), output_dir=tmp)

        self.assertEqual(len(result.classification_rows), 1)
        row = result.classification_rows[0]
        self.assertEqual(row.path_levels, ["基础资源", "服务范围与对象", "患者"])
        self.assertEqual(row.recommended_grade, "3级")
        self.assertEqual(row.description_source, "quoted")
        self.assertEqual(row.evidence_refs[0].chunk_id, "doc_1_chunk_1")

    def test_invalid_evidence_refs_force_review_and_proposed_status(self) -> None:
        state = make_state()

        def fake_call_llm_json(**kwargs):
            return (
                {
                    "classification_rows": [
                        {
                            "path_levels": ["基础资源", "服务范围与对象", "患者"],
                            "recommended_grade": "3级",
                            "description": "患者信息包括身份识别资料",
                            "description_source": "quoted",
                            "evidence_quote": "基础资源 服务范围与对象 患者 3级 患者信息包括身份识别资料",
                            "evidence_chunk_ids": ["missing_chunk"],
                            "support_level": "explicit",
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                            "status": "evidence_supported",
                        }
                    ]
                },
                "raw",
            )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.steps.classification_row_extractor.call_llm_json", side_effect=fake_call_llm_json):
                result = extract_classification_rows_with_llm(state, object(), output_dir=tmp)

        row = result.classification_rows[0]
        self.assertEqual(row.evidence_refs, [])
        self.assertTrue(row.needs_review)
        self.assertEqual(row.status, "proposed")
        self.assertEqual(row.review_reason, "分类行缺少有效证据引用。")

    def test_empty_description_is_normalized_to_insufficient(self) -> None:
        state = make_state()

        def fake_call_llm_json(**kwargs):
            return (
                {
                    "classification_rows": [
                        {
                            "path_levels": ["基础资源", "服务范围与对象", "患者"],
                            "recommended_grade": "3级",
                            "description": "",
                            "description_source": "quoted",
                            "evidence_quote": "基础资源 服务范围与对象 患者 3级",
                            "evidence_chunk_ids": ["doc_1_chunk_1"],
                            "support_level": "explicit",
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                            "status": "evidence_supported",
                        }
                    ]
                },
                "raw",
            )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.steps.classification_row_extractor.call_llm_json", side_effect=fake_call_llm_json):
                result = extract_classification_rows_with_llm(state, object(), output_dir=tmp)

        row = result.classification_rows[0]
        self.assertEqual(row.description_source, "insufficient")
        self.assertEqual(row.description, "证据不足，无法从当前文档确定")
        self.assertTrue(row.needs_review)
        self.assertEqual(row.review_reason, "当前文档未提供该分类项的说明或范围描述。")

    def test_duplicate_rows_keep_stronger_evidence_and_description(self) -> None:
        state = make_state()

        def fake_call_llm_json(**kwargs):
            return (
                {
                    "classification_rows": [
                        {
                            "path_levels": ["基础资源", "服务范围与对象", "患者"],
                            "recommended_grade": "3级",
                            "description": "证据不足，无法从当前文档确定",
                            "description_source": "insufficient",
                            "evidence_quote": "基础资源 服务范围与对象 患者 3级",
                            "evidence_chunk_ids": [],
                            "support_level": "weak",
                            "confidence": 0.4,
                            "needs_review": True,
                            "review_reason": "说明不足",
                            "status": "proposed",
                        },
                        {
                            "path_levels": ["基础资源", "服务范围与对象", "患者"],
                            "recommended_grade": "3级",
                            "description": "患者信息包括身份识别资料",
                            "description_source": "quoted",
                            "description_evidence_quote": "患者信息包括身份识别资料",
                            "evidence_quote": "基础资源 服务范围与对象 患者 3级 患者信息包括身份识别资料",
                            "evidence_chunk_ids": ["doc_1_chunk_1"],
                            "support_level": "explicit",
                            "confidence": 0.9,
                            "needs_review": False,
                            "review_reason": "",
                            "status": "evidence_supported",
                        },
                    ]
                },
                "raw",
            )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.steps.classification_row_extractor.call_llm_json", side_effect=fake_call_llm_json):
                result = extract_classification_rows_with_llm(state, object(), output_dir=tmp)

        self.assertEqual(len(result.classification_rows), 1)
        row = result.classification_rows[0]
        self.assertEqual(row.description_source, "quoted")
        self.assertEqual(row.description, "患者信息包括身份识别资料")
        self.assertEqual(row.support_level, "explicit")
        self.assertEqual(row.evidence_refs[0].chunk_id, "doc_1_chunk_1")
        self.assertFalse(row.needs_review)


if __name__ == "__main__":
    unittest.main()
