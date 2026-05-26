from __future__ import annotations

import unittest

from src.core.agent_state import (
    AgentState,
    ClassificationRow,
    DocumentPage,
    EvidenceRef,
    SourceDocument,
)
from src.validation.row_grounding_validator import validate_row_grounding


def _doc() -> SourceDocument:
    text = "基础资源 服务范围与对象 患者 3级 分类说明 患者相关资料"
    return SourceDocument(
        doc_id="doc_1",
        doc_name="policy.txt",
        file_path="policy.txt",
        raw_text=text,
        pages=[DocumentPage(page_number=None, text=text)],
    )


def _evidence_ref(text: str = "基础资源 服务范围与对象 患者 3级 分类说明 患者相关资料") -> EvidenceRef:
    return EvidenceRef(
        evidence_id="ev_1",
        chunk_id="doc_1_chunk_1",
        doc_name="policy.txt",
        section_title="分类表",
        text=text,
        used_for="classification_row",
        relevance_score=0.9,
    )


class RowGroundingValidatorTests(unittest.TestCase):
    def test_flags_missing_quote_and_preserves_insufficient_description(self) -> None:
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="policy.txt",
            file_path="policy.txt",
            raw_text="基础资源 服务范围与对象 患者 3级",
            pages=[DocumentPage(page_number=None, text="基础资源 服务范围与对象 患者 3级")],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="3级",
            description="证据不足，无法从当前文档确定",
            description_source="insufficient",
            evidence_quote="不存在的 quote",
            evidence_refs=[
                EvidenceRef(
                    evidence_id="ev_1",
                    chunk_id="doc_1_chunk_1",
                    doc_name="policy.txt",
                    section_title="分类表",
                    text="基础资源 服务范围与对象 患者 3级",
                    used_for="classification_row",
                    relevance_score=0.9,
                )
            ],
            support_level="explicit",
            needs_review=True,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        issue_types = {issue.issue_type for issue in issues}
        self.assertIn("hardcoded_or_ungrounded_content", issue_types)
        self.assertNotIn("invalid_insufficient_description", issue_types)

    def test_flags_missing_classification_rows(self) -> None:
        state = AgentState(task="test", documents=[_doc()])

        issues = validate_row_grounding(state)

        issue = next(issue for issue in issues if issue.issue_type == "missing_classification_rows")
        self.assertEqual(issue.severity, "high")

    def test_flags_weak_or_structural_row_without_review(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="3级",
            evidence_quote="基础资源 服务范围与对象 患者 3级",
            evidence_refs=[_evidence_ref()],
            support_level="weak",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[_doc()], classification_rows=[row])

        issues = validate_row_grounding(state)

        issue_types = {issue.issue_type for issue in issues}
        self.assertIn("schema_contract_violation", issue_types)

    def test_flags_non_insufficient_description_without_quote(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="3级",
            description="患者相关资料",
            description_source="quoted",
            description_evidence_quote="",
            evidence_quote="基础资源 服务范围与对象 患者 3级",
            evidence_refs=[_evidence_ref()],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[_doc()], classification_rows=[row])

        issues = validate_row_grounding(state)

        issue_types = {issue.issue_type for issue in issues}
        self.assertIn("weak_trace", issue_types)

    def test_flags_duplicate_path(self) -> None:
        row_1 = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="3级",
            evidence_quote="基础资源 服务范围与对象 患者 3级",
            evidence_refs=[_evidence_ref()],
            support_level="explicit",
        )
        row_2 = ClassificationRow(
            row_id="row_2",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="3级",
            evidence_quote="基础资源 服务范围与对象 患者 3级",
            evidence_refs=[_evidence_ref()],
            support_level="explicit",
        )
        state = AgentState(task="test", documents=[_doc()], classification_rows=[row_1, row_2])

        issues = validate_row_grounding(state)

        issue = next(issue for issue in issues if issue.issue_type == "duplicated_path")
        self.assertEqual(issue.severity, "medium")


if __name__ == "__main__":
    unittest.main()
