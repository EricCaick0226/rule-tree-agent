from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.agent_state import (  # noqa: E402
    AgentState,
    ClassificationRow,
    ClassificationSchema,
    DocumentChunk,
    DocumentPage,
    EvidenceRef,
    EvidenceClaim,
    GradeDefinition,
    SourceDocument,
    StepTrace,
    TreeNode,
    ValidationIssue,
)
from src.output.exporter import export_outputs  # noqa: E402
from src.steps.tree_projector import project_tree_from_rows  # noqa: E402

T = TypeVar("T")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _dataclass_from_dict(cls: type[T], data: object) -> T | None:
    if not isinstance(data, dict):
        return None
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type: {cls}")
    allowed = {field.name for field in fields(cls)}
    payload = {key: deepcopy(value) for key, value in data.items() if key in allowed}
    return cls(**payload)


def _evidence_refs(items: object) -> list[EvidenceRef]:
    if not isinstance(items, list):
        return []
    refs: list[EvidenceRef] = []
    for item in items:
        ref = _dataclass_from_dict(EvidenceRef, item)
        if ref is not None:
            refs.append(ref)
    return refs


def _classification_row(data: object) -> ClassificationRow | None:
    row = _dataclass_from_dict(ClassificationRow, data)
    if row is None:
        return None
    if isinstance(data, dict):
        row.evidence_refs = _evidence_refs(data.get("evidence_refs"))
    return row


def _classification_schema(data: object) -> ClassificationSchema | None:
    schema = _dataclass_from_dict(ClassificationSchema, data)
    if schema is None:
        return None
    if isinstance(data, dict):
        schema.evidence_refs = _evidence_refs(data.get("evidence_refs"))
    return schema


def _grade_definition(data: object) -> GradeDefinition | None:
    grade = _dataclass_from_dict(GradeDefinition, data)
    if grade is None:
        return None
    if isinstance(data, dict):
        grade.evidence_refs = _evidence_refs(data.get("evidence_refs"))
    return grade


def _validation_issue(data: object) -> ValidationIssue | None:
    return _dataclass_from_dict(ValidationIssue, data)


def _document_page(data: object) -> DocumentPage | None:
    return _dataclass_from_dict(DocumentPage, data)


def _source_document(data: object) -> SourceDocument | None:
    document = _dataclass_from_dict(SourceDocument, data)
    if document is None:
        return None
    if isinstance(data, dict):
        document.pages = [
            page
            for item in data.get("pages", [])
            if (page := _document_page(item)) is not None
        ]
    return document


def _document_chunk(data: object) -> DocumentChunk | None:
    return _dataclass_from_dict(DocumentChunk, data)


def _evidence_claim(data: object) -> EvidenceClaim | None:
    claim = _dataclass_from_dict(EvidenceClaim, data)
    if claim is None:
        return None
    if isinstance(data, dict):
        claim.evidence_refs = _evidence_refs(data.get("evidence_refs"))
    return claim


def _tree_node(data: object) -> TreeNode | None:
    node = _dataclass_from_dict(TreeNode, data)
    if node is None:
        return None
    if isinstance(data, dict):
        node.evidence_refs = _evidence_refs(data.get("evidence_refs"))
        node.description_evidence_refs = _evidence_refs(data.get("description_evidence_refs"))
        node.grade_evidence_refs = _evidence_refs(data.get("grade_evidence_refs"))
    return node


def _step_trace(data: object) -> StepTrace | None:
    return _dataclass_from_dict(StepTrace, data)


def _is_reference_candidate(row: ClassificationRow) -> bool:
    return (
        row.inclusion_status == "review_candidate"
        or row.row_source == "reference_library"
        or row.evidence_status == "reference_only"
    )


def _load_state_from_rule_table(rule_table_path: Path) -> AgentState:
    payload = _load_json(rule_table_path)
    rows_payload = payload.get("classification_rows")
    if not isinstance(rows_payload, list):
        raise ValueError(f"rule_table.json missing classification_rows array: {rule_table_path}")

    rows = [
        row
        for item in rows_payload
        if (row := _classification_row(item)) is not None
    ]
    grade_scheme = [
        grade
        for item in payload.get("grade_scheme", [])
        if (grade := _grade_definition(item)) is not None
    ]
    validation_issues = [
        issue
        for item in payload.get("validation_issues", [])
        if (issue := _validation_issue(item)) is not None
    ]
    return AgentState(
        task="split_reference_candidates_from_output",
        classification_schema=_classification_schema(payload.get("classification_schema")),
        grade_scheme=grade_scheme,
        classification_rows=[row for row in rows if not _is_reference_candidate(row)],
        reference_candidate_rows=[row for row in rows if _is_reference_candidate(row)],
        validation_issues=validation_issues,
        llm_enabled=False,
        llm_used=False,
    )


def _merge_run_state(input_dir: Path, state: AgentState) -> AgentState:
    rule_tree_path = input_dir / "rule_tree.json"
    if not rule_tree_path.exists():
        return state

    payload = _load_json(rule_tree_path)
    state.task = str(payload.get("task") or state.task)
    state.task_type = str(payload.get("task_type") or state.task_type)
    state.input_files = [
        str(item) for item in payload.get("input_files", []) if str(item or "").strip()
    ]
    state.documents = [
        document
        for item in payload.get("documents", [])
        if (document := _source_document(item)) is not None
    ]
    state.chunks = [
        chunk
        for item in payload.get("chunks", [])
        if (chunk := _document_chunk(item)) is not None
    ]
    state.evidence_claims = [
        claim
        for item in payload.get("evidence_claims", [])
        if (claim := _evidence_claim(item)) is not None
    ]
    state.nodes = [
        node
        for item in payload.get("nodes", [])
        if (node := _tree_node(item)) is not None
    ]
    state.step_traces = [
        trace
        for item in payload.get("step_traces", [])
        if (trace := _step_trace(item)) is not None
    ]
    state.llm_enabled = bool(payload.get("llm_enabled"))
    state.llm_used = bool(payload.get("llm_used"))
    state.llm_model = str(payload.get("llm_model") or "")
    state.llm_base_url = str(payload.get("llm_base_url") or "")
    state.llm_error = str(payload.get("llm_error") or "")
    state.pdf_ocr_enabled = bool(payload.get("pdf_ocr_enabled"))
    return state


def split_reference_candidates(input_dir: Path, output_dir: Path) -> dict[str, int]:
    rule_table_path = input_dir / "rule_table.json"
    state = _load_state_from_rule_table(rule_table_path)
    state = _merge_run_state(input_dir, state)
    original_rows = len(state.classification_rows) + len(state.reference_candidate_rows)
    state = project_tree_from_rows(state)
    export_outputs(state, str(output_dir))
    return {
        "original_rows": original_rows,
        "classification_rows": len(state.classification_rows),
        "reference_candidate_rows": len(state.reference_candidate_rows),
        "reference_prefilled_rows": sum(
            1 for row in state.classification_rows if row.reference_prefilled_fields
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split legacy reference review candidates out of an existing output rule_table."
    )
    parser.add_argument("--input-dir", required=True, help="Existing output directory containing rule_table.json.")
    parser.add_argument("--out", required=True, help="Cleaned output directory.")
    args = parser.parse_args()

    summary = split_reference_candidates(Path(args.input_dir), Path(args.out))
    print(f"Wrote cleaned output: {Path(args.out)}")
    print(
        "original_rows={original_rows} classification_rows={classification_rows} "
        "reference_candidate_rows={reference_candidate_rows} reference_prefilled_rows={reference_prefilled_rows}".format(
            **summary
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
