from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.core.agent_state import AgentState, ClassificationRow, EvidenceRef
from src.steps.reference_row_prefill import (
    apply_reference_row_reuse,
    prefill_rows_from_reference_library,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "description": "国家卫生信息资源分类与编码管理规范",
            "reuse_policy": "direct",
            "reference_trust_level": "authoritative",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_hardware",
                    "path_levels": ["基础资源", "设备资源", "硬件设备"],
                    "recommended_grade": "一般数据 2 级",
                    "description": "硬件设备相关信息。",
                    "description_source": "quoted",
                    "data_range_examples": ["设备名称", "设备编号"],
                    "data_element_refs": ["WS/T 363.16—2023:DE08.10.001.00"],
                },
                {
                    "row_id": "ref_software",
                    "path_levels": ["基础资源", "设备资源", "软件设备"],
                    "recommended_grade": "一般数据 2 级",
                    "description": "软件设备相关信息。",
                    "description_source": "quoted",
                    "data_range_examples": ["软件名称"],
                },
            ]
        },
    )
    return library


def _untrusted_complete_reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": "assist",
            "reference_trust_level": "auxiliary",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_hardware",
                    "path_levels": ["基础资源", "设备资源", "硬件设备"],
                    "recommended_grade": "一般数据 2 级",
                    "description": "硬件设备相关信息。",
                    "data_range_examples": ["设备名称", "设备编号"],
                    "data_element_refs": ["WS/T 363.16—2023:DE08.10.001.00"],
                }
            ]
        },
    )
    return library


def _restrictive_reuse_allowed_fields_reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": "direct",
            "reference_trust_level": "authoritative",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_hardware",
                    "path_levels": ["基础资源", "设备资源", "硬件设备"],
                    "description": "硬件设备相关信息。",
                    "data_range_examples": ["设备名称", "设备编号"],
                    "data_element_refs": ["WS/T 363.16—2023:DE08.10.001.00"],
                    "reuse_allowed_fields": ["description"],
                }
            ]
        },
    )
    return library


def _mixed_case_direct_reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": " Direct ",
            "reference_trust_level": " Authoritative ",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_hardware",
                    "path_levels": ["基础资源", "设备资源", "硬件设备"],
                    "description": "硬件设备相关信息。",
                    "data_range_examples": ["设备名称", "设备编号"],
                    "data_element_refs": ["WS/T 363.16—2023:DE08.10.001.00"],
                }
            ]
        },
    )
    return library


def _complete_without_optional_fields_reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": "direct",
            "reference_trust_level": "authoritative",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_hardware",
                    "path_levels": ["基础资源", "设备资源", "硬件设备"],
                    "description": "硬件设备相关信息。",
                    "data_range_examples": ["设备名称", "设备编号"],
                }
            ]
        },
    )
    return library


def _path_only_reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": "direct",
            "reference_trust_level": "authoritative",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_patient",
                    "path_levels": ["基础资源类", "服务范围与对象", "患者信息"],
                    "description": "证据不足，无法从当前文档确定",
                    "description_source": "insufficient",
                    "data_range_examples": [],
                }
            ]
        },
    )
    return library


def _generic_suffix_reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": "direct",
            "reference_trust_level": "authoritative",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_patient",
                    "path_levels": ["患者"],
                    "description": "证据不足，无法从当前文档确定",
                    "description_source": "insufficient",
                    "data_range_examples": [],
                }
            ]
        },
    )
    return library


def _alias_reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": "direct",
            "reference_trust_level": "authoritative",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_patient",
                    "path_levels": ["患者"],
                    "aliases": ["患者信息"],
                    "description": "证据不足，无法从当前文档确定",
                    "description_source": "insufficient",
                    "data_range_examples": [],
                }
            ]
        },
    )
    return library


def _complete_alias_reference_library(root: Path) -> Path:
    library = root / "reference_library"
    _write_json(
        library / "wst787_2021" / "metadata.json",
        {
            "name": "WST 787-2021",
            "source_type": "national_standard",
            "reuse_policy": "direct",
            "reference_trust_level": "authoritative",
        },
    )
    _write_json(
        library / "wst787_2021" / "rule_table.json",
        {
            "classification_rows": [
                {
                    "row_id": "ref_patient",
                    "path_levels": ["患者"],
                    "aliases": ["患者信息"],
                    "description": "患者相关信息。",
                    "data_range_examples": ["患者姓名"],
                    "data_element_refs": ["WS/T 363.3—2023:DE02.01.039.01"],
                }
            ]
        },
    )
    return library


def _evidence_ref() -> EvidenceRef:
    return EvidenceRef(
        evidence_id="ev_1",
        chunk_id="chunk_1",
        doc_name="policy.txt",
        section_title="分类表",
        text="基础资源 设备资源 硬件设备",
        used_for="classification_row",
        relevance_score=0.9,
    )


class ReferenceRowPrefillTests(unittest.TestCase):
    def test_direct_reuse_replaces_reusable_fields_without_reusing_grade(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_hardware",
                        path_levels=["设备资源", "硬件设备"],
                        recommended_grade="敏感数据 4 级",
                        description="当前文档中的硬件设备描述。",
                        description_source="quoted",
                        description_evidence_quote="当前文档描述证据",
                        data_range_examples=["本地设备字段"],
                        data_element_refs=["LOCAL:DE01"],
                        evidence_quote="设备资源 硬件设备",
                        evidence_refs=[_evidence_ref()],
                        support_level="explicit",
                        needs_review=False,
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_hardware")
        self.assertEqual(current.path_levels, ["基础资源", "设备资源", "硬件设备"])
        self.assertEqual(current.original_path_levels, ["设备资源", "硬件设备"])
        self.assertEqual(current.description, "硬件设备相关信息。")
        self.assertEqual(current.description_source, "quoted")
        self.assertEqual(current.description_evidence_quote, "")
        self.assertEqual(current.recommended_grade, "敏感数据 4 级")
        self.assertEqual(current.data_range_examples, ["设备名称", "设备编号"])
        self.assertEqual(current.data_element_refs, ["WS/T 363.16—2023:DE08.10.001.00"])
        self.assertEqual(
            current.reference_prefilled_fields,
            ["path_levels", "description", "data_range_examples", "data_element_refs"],
        )
        self.assertEqual(current.content_source, "reference_library")
        self.assertEqual(current.row_source, "current_document")
        self.assertEqual(current.evidence_status, "current_document_supported")
        self.assertEqual(current.reference_matches[0]["match_type"], "parent_and_leaf")
        self.assertEqual(current.reference_matches[0]["usage"], "direct_reuse")
        self.assertEqual(result.step_traces[-1].step_name, "apply_reference_row_reuse")
        self.assertEqual(
            result.step_traces[-1].output_summary,
            {
                "direct_reused_rows": 1,
                "reused_fields": 4,
                "candidate_rows": 1,
                "classification_rows": 2,
            },
        )

    def test_restrictive_reuse_allowed_fields_does_not_limit_direct_reuse(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _restrictive_reuse_allowed_fields_reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_hardware",
                        path_levels=["设备资源", "硬件设备"],
                        description="本地硬件描述。",
                        description_source="quoted",
                        data_range_examples=["本地设备字段"],
                        data_element_refs=["LOCAL:DE01"],
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_hardware")
        self.assertEqual(current.path_levels, ["基础资源", "设备资源", "硬件设备"])
        self.assertEqual(current.description, "硬件设备相关信息。")
        self.assertEqual(current.data_range_examples, ["设备名称", "设备编号"])
        self.assertEqual(current.data_element_refs, ["WS/T 363.16—2023:DE08.10.001.00"])
        self.assertEqual(
            current.reference_prefilled_fields,
            ["path_levels", "description", "data_range_examples", "data_element_refs"],
        )
        self.assertEqual(current.reference_matches[0]["usage"], "direct_reuse")

    def test_direct_reuse_policy_metadata_is_case_and_whitespace_insensitive(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _mixed_case_direct_reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_hardware",
                        path_levels=["设备资源", "硬件设备"],
                        description="证据不足，无法从当前文档确定",
                        description_source="insufficient",
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_hardware")
        self.assertEqual(current.description, "硬件设备相关信息。")
        self.assertEqual(current.data_range_examples, ["设备名称", "设备编号"])
        self.assertEqual(current.reference_matches[0]["usage"], "direct_reuse")

    def test_direct_reuse_clears_optional_fields_omitted_by_reference(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _complete_without_optional_fields_reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_hardware",
                        path_levels=["设备资源", "硬件设备"],
                        description="本地硬件描述。",
                        description_source="quoted",
                        data_element_refs=["LOCAL:DE01"],
                        processing_degree="本地处理程度",
                        impact_object="本地影响对象",
                        impact_degree="本地影响程度",
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_hardware")
        self.assertEqual(current.description, "硬件设备相关信息。")
        self.assertEqual(current.data_range_examples, ["设备名称", "设备编号"])
        self.assertEqual(current.data_element_refs, [])
        self.assertEqual(current.processing_degree, "")
        self.assertEqual(current.impact_object, "")
        self.assertEqual(current.impact_degree, "")

    def test_untrusted_complete_reference_row_is_ignored(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _untrusted_complete_reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_hardware",
                        path_levels=["设备资源", "硬件设备"],
                        description="证据不足，无法从当前文档确定",
                        description_source="insufficient",
                        evidence_quote="设备资源 硬件设备",
                        evidence_refs=[_evidence_ref()],
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_hardware")
        self.assertEqual(current.path_levels, ["设备资源", "硬件设备"])
        self.assertEqual(current.description_source, "insufficient")
        self.assertEqual(current.reference_prefilled_fields, [])
        self.assertEqual(current.reference_matches, [])
        self.assertFalse(
            any(
                row.row_source == "reference_library"
                and row.inclusion_status == "review_candidate"
                for row in result.classification_rows
            )
        )

    def test_adds_unmatched_reference_rows_as_review_candidates(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_hardware",
                        path_levels=["基础资源", "设备资源", "硬件设备"],
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        candidates = [
            row
            for row in result.classification_rows
            if row.row_source == "reference_library" and row.inclusion_status == "review_candidate"
        ]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].path_levels, ["基础资源", "设备资源", "软件设备"])
        self.assertIsNone(candidates[0].recommended_grade)
        self.assertEqual(candidates[0].evidence_status, "reference_only")
        self.assertEqual(candidates[0].content_source, "reference_library")
        self.assertTrue(candidates[0].needs_review)

    def test_path_only_reference_row_is_ignored_without_match_or_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _path_only_reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_patient",
                        path_levels=["服务范围与对象", "患者信息"],
                        description="证据不足，无法从当前文档确定",
                        description_source="insufficient",
                        evidence_quote="服务范围与对象 患者信息",
                        evidence_refs=[_evidence_ref()],
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_patient")
        self.assertEqual(current.description_source, "insufficient")
        self.assertEqual(current.reference_prefilled_fields, [])
        self.assertEqual(current.reference_matches, [])
        self.assertFalse(
            any(
                row.row_source == "reference_library"
                and row.inclusion_status == "review_candidate"
                for row in result.classification_rows
            )
        )

    def test_generic_information_suffix_does_not_match_without_reviewed_alias(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _generic_suffix_reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_patient",
                        path_levels=["基础资源类", "服务范围与对象", "患者信息"],
                        description="证据不足，无法从当前文档确定",
                        description_source="insufficient",
                        evidence_quote="服务范围与对象 患者信息",
                        evidence_refs=[_evidence_ref()],
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_patient")
        self.assertEqual(current.reference_prefilled_fields, [])
        self.assertEqual(current.reference_matches, [])
        self.assertFalse(
            any(
                row.row_source == "reference_library"
                and row.inclusion_status == "review_candidate"
                for row in result.classification_rows
            )
        )

    def test_reviewed_alias_without_reusable_content_is_ignored(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _alias_reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_patient",
                        path_levels=["基础资源类", "服务范围与对象", "患者信息"],
                        description="证据不足，无法从当前文档确定",
                        description_source="insufficient",
                        evidence_quote="服务范围与对象 患者信息",
                        evidence_refs=[_evidence_ref()],
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_patient")
        self.assertEqual(current.reference_prefilled_fields, [])
        self.assertEqual(current.reference_matches, [])
        self.assertFalse(
            any(
                row.row_source == "reference_library"
                and row.inclusion_status == "review_candidate"
                for row in result.classification_rows
            )
        )

    def test_reviewed_alias_can_direct_reuse_complete_reference_row(self) -> None:
        with TemporaryDirectory() as tmp:
            library = _complete_alias_reference_library(Path(tmp))
            state = AgentState(
                task="test",
                classification_rows=[
                    ClassificationRow(
                        row_id="cur_patient",
                        path_levels=["基础资源类", "服务范围与对象", "患者信息"],
                        description="证据不足，无法从当前文档确定",
                        description_source="insufficient",
                        evidence_quote="服务范围与对象 患者信息",
                        evidence_refs=[_evidence_ref()],
                    )
                ],
            )

            result = apply_reference_row_reuse(state, library_dir=str(library))

        current = next(row for row in result.classification_rows if row.row_id == "cur_patient")
        self.assertEqual(
            current.reference_prefilled_fields,
            ["path_levels", "description", "data_range_examples", "data_element_refs"],
        )
        self.assertEqual(current.reference_matches[0]["usage"], "direct_reuse")
        self.assertEqual(current.reference_matches[0]["match_type"], "exact_alias")
        self.assertEqual(current.path_levels, ["患者"])
        self.assertEqual(current.description, "患者相关信息。")
        self.assertEqual(current.data_range_examples, ["患者姓名"])

    def test_skips_when_library_is_not_configured(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[ClassificationRow(row_id="cur", path_levels=["A"])],
        )

        result = prefill_rows_from_reference_library(state, library_dir="")

        self.assertEqual(len(result.classification_rows), 1)
        self.assertEqual(result.step_traces[-1].status, "skipped")
        self.assertEqual(result.step_traces[-1].step_name, "apply_reference_row_reuse")

    def test_prefill_wrapper_keeps_empty_library_dir_compatibility(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[ClassificationRow(row_id="cur", path_levels=["A"])],
        )

        result = prefill_rows_from_reference_library(state, library_dir="")

        self.assertIs(result, state)
        self.assertEqual(len(result.classification_rows), 1)
        self.assertEqual(result.step_traces[-1].status, "skipped")


if __name__ == "__main__":
    unittest.main()
