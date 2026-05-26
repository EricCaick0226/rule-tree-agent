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
class CandidateConcept:
    concept_id: str
    text: str
    normalized_text: str
    concept_type: str
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True


@dataclass
class EvidenceClaim:
    claim_id: str
    claim_type: str
    subject: str
    predicate: str
    object: str = ""
    value: str = ""
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    status: str = "proposed"


@dataclass
class ConceptProfile:
    concept_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    included_items: list[str] = field(default_factory=list)
    excluded_items: list[str] = field(default_factory=list)
    related_claim_ids: list[str] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    status: str = "proposed"


@dataclass
class ClassificationDimension:
    dimension_id: str
    name: str
    description: str
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    evidence_claim_ids: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    needs_review: bool = True


@dataclass
class MatchingRule:
    rule_id: str
    target_node_id: str
    rule_type: str
    conditions: list[str] = field(default_factory=list)
    negative_conditions: list[str] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    evidence_claim_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
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
    rules: list[MatchingRule] = field(default_factory=list)
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
    status: str = "proposed"


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
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    evidence_claims: list[EvidenceClaim] = field(default_factory=list)
    concept_profiles: list[ConceptProfile] = field(default_factory=list)
    candidate_concepts: list[CandidateConcept] = field(default_factory=list)
    classification_dimensions: list[ClassificationDimension] = field(default_factory=list)
    selected_dimension: Optional[ClassificationDimension] = None
    grade_scheme: list[GradeDefinition] = field(default_factory=list)
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
