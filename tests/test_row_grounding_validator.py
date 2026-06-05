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

    def test_path_level_match_allows_heading_punctuation_variants(self) -> None:
        doc_text = "C、主题资源\n5其他数据库\n002住院记录摘要"
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=doc_text,
            pages=[DocumentPage(page_number=None, text=doc_text)],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["C主题资源", "5其他数据库", "002住院记录摘要"],
            evidence_quote="C、主题资源；5其他数据库；002住院记录摘要",
            evidence_refs=[
                _evidence_ref("C、主题资源"),
                _evidence_ref("5其他数据库"),
                _evidence_ref("002住院记录摘要"),
            ],
            description="证据不足，无法从当前文档确定",
            description_source="insufficient",
            support_level="explicit",
            needs_review=True,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        path_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "分类层级未出现在输入文档中" in issue.problem
        ]
        self.assertEqual(path_issues, [])

    def test_flags_grade_in_ref_text_but_missing_from_row_quote(self) -> None:
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="policy.txt",
            file_path="policy.txt",
            raw_text="基础资源 服务范围与对象 患者\n医生 4级",
            pages=[
                DocumentPage(
                    page_number=None,
                    text="基础资源 服务范围与对象 患者\n医生 4级",
                )
            ],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="4级",
            evidence_quote="患者",
            evidence_refs=[_evidence_ref("患者\n医生 4级")],
            support_level="explicit",
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        grade_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "推荐分级" in issue.problem
        ]
        self.assertTrue(grade_issues)
        self.assertEqual(grade_issues[0].severity, "medium")

    def test_evidence_quote_can_cover_description_quote(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="3级",
            description="患者相关资料",
            description_source="quoted",
            description_evidence_quote="",
            evidence_quote="基础资源 服务范围与对象 患者 3级 分类说明 患者相关资料",
            evidence_refs=[_evidence_ref()],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[_doc()], classification_rows=[row])

        issues = validate_row_grounding(state)

        weak_trace_problems = [
            issue.problem for issue in issues if issue.issue_type == "weak_trace"
        ]
        self.assertNotIn("分类说明缺少 description_evidence_quote。", weak_trace_problems)

    def test_multiline_quote_must_be_contiguous_after_whitespace_normalization(self) -> None:
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者"],
            recommended_grade="3级",
            evidence_quote="患者\n3级",
            evidence_refs=[_evidence_ref("基础资源 服务范围与对象 患者\n中间无关内容\n3级")],
            support_level="explicit",
        )
        state = AgentState(task="test", documents=[_doc()], classification_rows=[row])

        issues = validate_row_grounding(state)

        quote_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "evidence_quote" in issue.problem
        ]
        self.assertTrue(quote_issues)
        self.assertEqual(quote_issues[0].severity, "high")

    def test_grade_evidence_quote_can_support_recommended_grade(self) -> None:
        text = "姓名 原始数据 个人 严重危害 一般数据3级"
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=text,
            pages=[DocumentPage(page_number=None, text=text)],
        )
        state = AgentState(task="test", documents=[doc])
        ref = EvidenceRef(
            evidence_id="ev1",
            chunk_id="c1",
            doc_name="sample.txt",
            section_title="附录B",
            text=text,
            used_for="row",
            relevance_score=0.9,
        )
        state.classification_rows = [
            ClassificationRow(
                row_id="r1",
                path_levels=["姓名"],
                recommended_grade="一般数据3级",
                description="姓名",
                description_source="quoted",
                evidence_quote="姓名",
                grade_evidence_quote="原始数据 个人 严重危害 一般数据3级",
                evidence_refs=[ref],
                support_level="explicit",
                needs_review=False,
            )
        ]

        issues = validate_row_grounding(state)

        self.assertEqual(issues, [])

    def test_grade_evidence_quote_must_match_referenced_evidence_text(self) -> None:
        text = "姓名 原始数据 个人 严重危害 一般数据3级"
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=text,
            pages=[DocumentPage(page_number=None, text=text)],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["姓名"],
            recommended_grade="一般数据3级",
            description="姓名",
            description_source="quoted",
            evidence_quote="姓名 一般数据3级",
            grade_evidence_quote="原始数据 个人 中间无关内容 严重危害 一般数据3级",
            evidence_refs=[_evidence_ref(text)],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        quote_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "grade_evidence_quote" in issue.problem
        ]
        self.assertTrue(quote_issues)
        self.assertEqual(quote_issues[0].severity, "high")

    def test_grade_evidence_quote_cannot_be_stitched_across_refs(self) -> None:
        text = "姓名 原始数据 个人 严重危害 一般数据3级"
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=text,
            pages=[DocumentPage(page_number=None, text=text)],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["姓名"],
            recommended_grade="一般数据3级",
            description="姓名",
            description_source="quoted",
            evidence_quote="姓名 一般数据3级",
            grade_evidence_quote="原始数据 个人 严重危害 一般数据3级",
            evidence_refs=[
                _evidence_ref("姓名 原始数据 个人"),
                _evidence_ref("严重危害 一般数据3级"),
            ],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        quote_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "grade_evidence_quote" in issue.problem
        ]
        self.assertTrue(quote_issues)

    def test_merged_evidence_quote_parts_can_match_separate_refs(self) -> None:
        doc_text = (
            "A、基础资源 1服务范围与对象\n"
            "001患者信息 患者姓名 原始数据 个人 严重危害 一般数据3级"
        )
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=doc_text,
            pages=[DocumentPage(page_number=None, text=doc_text)],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["基础资源", "服务范围与对象", "患者信息", "患者姓名"],
            recommended_grade="一般数据3级",
            description="证据不足，无法从当前文档确定",
            description_source="insufficient",
            evidence_quote=(
                "A、基础资源 1服务范围与对象；"
                "001患者信息 患者姓名 原始数据 个人 严重危害 一般数据3级"
            ),
            evidence_refs=[
                _evidence_ref("A、基础资源 1服务范围与对象"),
                _evidence_ref(
                    "001患者信息 患者姓名 原始数据 个人 严重危害 一般数据3级"
                ),
            ],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        quote_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "evidence_quote" in issue.problem
        ]
        self.assertEqual(quote_issues, [])

    def test_merged_description_quote_parts_can_match_evidence_text(self) -> None:
        doc_text = "患者信息 与患者相关的数据\n患者姓名 患者姓名信息"
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=doc_text,
            pages=[DocumentPage(page_number=None, text=doc_text)],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["患者信息", "患者姓名"],
            recommended_grade="",
            description="患者姓名信息",
            description_source="quoted",
            evidence_quote="患者信息；患者姓名",
            description_evidence_quote="与患者相关的数据；患者姓名信息",
            evidence_refs=[
                _evidence_ref("患者信息 与患者相关的数据"),
                _evidence_ref("患者姓名 患者姓名信息"),
            ],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        quote_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "description_evidence_quote" in issue.problem
        ]
        self.assertEqual(quote_issues, [])

    def test_merged_grade_quote_parts_each_need_one_ref(self) -> None:
        doc_text = (
            "患者姓名 原始数据 个人 严重危害\n"
            "患者姓名适用 一般数据3级"
        )
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=doc_text,
            pages=[DocumentPage(page_number=None, text=doc_text)],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["患者姓名"],
            recommended_grade="一般数据3级",
            description="患者姓名",
            description_source="quoted",
            evidence_quote="患者姓名；一般数据3级",
            grade_evidence_quote=(
                "患者姓名 原始数据 个人 严重危害；"
                "患者姓名适用 一般数据3级"
            ),
            evidence_refs=[
                _evidence_ref("患者姓名 原始数据 个人 严重危害"),
                _evidence_ref("患者姓名适用 一般数据3级"),
            ],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        quote_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "grade_evidence_quote" in issue.problem
        ]
        self.assertEqual(quote_issues, [])

    def test_merged_grade_quote_part_cannot_be_stitched_across_refs(self) -> None:
        doc_text = "患者姓名 原始数据 个人 严重危害 一般数据3级"
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=doc_text,
            pages=[DocumentPage(page_number=None, text=doc_text)],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["患者姓名"],
            recommended_grade="一般数据3级",
            description="患者姓名",
            description_source="quoted",
            evidence_quote="患者姓名；一般数据3级",
            grade_evidence_quote="患者姓名 原始数据 个人 严重危害；一般数据3级",
            evidence_refs=[
                _evidence_ref("患者姓名 原始数据 个人"),
                _evidence_ref("严重危害"),
                _evidence_ref("一般数据3级"),
            ],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        quote_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "grade_evidence_quote" in issue.problem
        ]
        self.assertTrue(quote_issues)

    def test_recommended_grade_cannot_be_stitched_across_row_quotes(self) -> None:
        text = "姓名 一般数据 3级"
        doc = SourceDocument(
            doc_id="doc_1",
            doc_name="sample.txt",
            file_path="sample.txt",
            raw_text=text,
            pages=[DocumentPage(page_number=None, text=text)],
        )
        row = ClassificationRow(
            row_id="row_1",
            path_levels=["姓名"],
            recommended_grade="一般数据3级",
            description="姓名",
            description_source="quoted",
            evidence_quote="姓名 一般数据",
            grade_evidence_quote="3级",
            evidence_refs=[
                _evidence_ref("姓名 一般数据"),
                _evidence_ref("3级"),
            ],
            support_level="explicit",
            needs_review=False,
        )
        state = AgentState(task="test", documents=[doc], classification_rows=[row])

        issues = validate_row_grounding(state)

        grade_issues = [
            issue
            for issue in issues
            if issue.issue_type == "hardcoded_or_ungrounded_content"
            and "推荐分级" in issue.problem
        ]
        self.assertTrue(grade_issues)


if __name__ == "__main__":
    unittest.main()
