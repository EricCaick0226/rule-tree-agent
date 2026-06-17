from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SourceDocument:
    doc_id: str
    doc_name: str
    file_path: str
    raw_text: str
    pages: list["DocumentPage"] = field(default_factory=list)


@dataclass
class DocumentPage:
    page_number: Optional[int]
    text: str
    source_method: str = "text"
    needs_review: bool = False
    warning: str = ""


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    section_title: str
    text: str
    position: int
    page_number: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    source_method: str = "text"
    source_warning: str = ""


@dataclass
class EvidenceRef:
    evidence_id: str
    chunk_id: str
    doc_name: str
    section_title: str
    text: str
    used_for: str
    relevance_score: float
    page_number: Optional[int] = None
    source_method: str = "text"
    source_warning: str = ""


@dataclass
class EvidenceClaim:
    claim_id: str
    claim_type: str
    subject: str
    predicate: str
    object: str = ""
    value: str = ""
    evidence_quote: str = ""
    support_level: str = "weak"
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    review_reason: str = ""
    status: str = "proposed"


@dataclass
class TreeNode:
    node_id: str
    name: str
    path: str
    level: int
    parent_id: Optional[str]
    description: str = ""
    description_evidence_refs: list[EvidenceRef] = field(default_factory=list)
    grade: Optional[str] = None
    grade_evidence_refs: list[EvidenceRef] = field(default_factory=list)
    grade_reason: str = ""
    confidence: float = 0.0
    needs_review: bool = True
    status: str = "proposed"
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    evidence_claim_ids: list[str] = field(default_factory=list)
    description_evidence_level: str = "D"


@dataclass
class GradeDefinition:
    grade_id: str
    grade_name: str
    definition: str
    criteria: list[str] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    evidence_claim_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    review_reason: str = ""
    status: str = "proposed"


@dataclass
class ClassificationSchema:
    max_depth: int = 0
    source: str = "insufficient_evidence"
    evidence_quote: str = ""
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    review_reason: str = "证据不足，无法从当前文档确定分类层级。"


@dataclass
class ClassificationRow:
    row_id: str
    path_levels: list[str] = field(default_factory=list)
    original_path_levels: list[str] = field(default_factory=list)
    recommended_grade: Optional[str] = None
    description: str = "证据不足，无法从当前文档确定"
    description_source: str = "insufficient"
    description_evidence_quote: str = ""
    evidence_quote: str = ""
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    data_range_examples: list[str] = field(default_factory=list)
    data_element_refs: list[str] = field(default_factory=list)
    processing_degree: str = ""
    impact_object: str = ""
    impact_degree: str = ""
    grade_evidence_quote: str = ""
    support_level: str = "weak"
    confidence: float = 0.0
    needs_review: bool = True
    review_reason: str = ""
    status: str = "proposed"
    row_source: str = "current_document"
    content_source: str = "current_document"
    inclusion_status: str = "accepted"
    evidence_status: str = "current_document_supported"
    reference_matches: list[dict[str, Any]] = field(default_factory=list)
    reference_prefilled_fields: list[str] = field(default_factory=list)


@dataclass
class ValidationIssue:
    issue_id: str
    issue_type: str
    severity: str
    target: str
    problem: str
    suggested_action: str
    status: str = "open"


@dataclass
class StepTrace:
    step_name: str
    status: str
    message: str = ""
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    raw_response_path: str = ""


@dataclass
class AgentState:
    task: str
    task_type: str = "unknown"
    input_files: list[str] = field(default_factory=list)
    documents: list[SourceDocument] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    evidence_index: dict[str, Any] = field(default_factory=dict)
    evidence_claims: list[EvidenceClaim] = field(default_factory=list)
    grade_scheme: list[GradeDefinition] = field(default_factory=list)
    block_signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    classification_schema: Optional[ClassificationSchema] = None
    classification_rows: list[ClassificationRow] = field(default_factory=list)
    nodes: list[TreeNode] = field(default_factory=list)
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)
    step_traces: list[StepTrace] = field(default_factory=list)
    llm_enabled: bool = False
    llm_used: bool = False
    llm_model: str = ""
    llm_base_url: str = ""
    llm_error: str = ""
    pdf_ocr_enabled: bool = False
