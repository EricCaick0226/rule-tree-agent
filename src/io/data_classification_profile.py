from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


CONTENT_CLASSIFICATION_CATALOG = "classification_catalog"
CONTENT_CLASSIFICATION_GRADING_TABLE = "classification_grading_table"
CONTENT_RULE_GUIDANCE = "rule_guidance"
CONTENT_UNKNOWN = "unknown"

ROLE_CLASSIFICATION_PATH = "classification_path"
ROLE_DESCRIPTION_EVIDENCE = "description_evidence"
ROLE_GRADING_FACTOR = "grading_factor"
ROLE_GRADE_RESULT = "grade_result"
ROLE_METADATA_OR_UNKNOWN = "metadata_or_unknown"

FIELD_ROLE_ALIASES = {
    "类": ROLE_CLASSIFICATION_PATH,
    "项": ROLE_CLASSIFICATION_PATH,
    "目": ROLE_CLASSIFICATION_PATH,
    "数据范围及示例": ROLE_DESCRIPTION_EVIDENCE,
    "数据范围": ROLE_DESCRIPTION_EVIDENCE,
    "示例": ROLE_DESCRIPTION_EVIDENCE,
    "分类说明": ROLE_DESCRIPTION_EVIDENCE,
    "数据加工程度": ROLE_GRADING_FACTOR,
    "影响对象": ROLE_GRADING_FACTOR,
    "影响程度": ROLE_GRADING_FACTOR,
    "数据级别": ROLE_GRADE_RESULT,
    "推荐级别": ROLE_GRADE_RESULT,
    "推荐分级": ROLE_GRADE_RESULT,
}

GRADE_OR_RISK_PATTERNS = [
    re.compile(r"影响程度"),
    re.compile(r"数据级别"),
    re.compile(r"一般数据\d级"),
    re.compile(r"重要数据"),
    re.compile(r"核心数据"),
    re.compile(r"严重危害"),
    re.compile(r"特别严重危害"),
    re.compile(r"泄露后"),
]

DEFAULT_RESOURCE_TYPE_TERMS = ["基础资源", "业务资源", "主题资源"]


@dataclass(frozen=True)
class EvidenceSource:
    source_type: str
    text: str
    role: str
    unit_id: str = ""
    line_start: int | None = None
    line_end: int | None = None
    score: int = 0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "text": self.text,
            "role": self.role,
            "unit_id": self.unit_id,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "score": self.score,
            "reason": self.reason,
            "metadata": self.metadata,
        }


def clean_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def contains_grade_or_risk(text: str) -> bool:
    return any(pattern.search(str(text or "")) for pattern in GRADE_OR_RISK_PATTERNS)


def classify_field_role(header: str) -> str:
    normalized = clean_text(header)
    return FIELD_ROLE_ALIASES.get(normalized, ROLE_METADATA_OR_UNKNOWN)


def _has_hierarchy_header(text: str) -> bool:
    compact = clean_text(text)
    return "类" in compact and "项" in compact and "目" in compact


def _has_grading_header(text: str) -> bool:
    compact = clean_text(text)
    return all(term in compact for term in ["数据范围及示例", "影响程度", "数据级别"])


def classify_content_type(header_text: str = "", table_title: str = "", text: str = "") -> str:
    joined = "\n".join(str(value or "") for value in [header_text, table_title, text])
    if _has_grading_header(joined) or ("分类分级表" in joined and "数据级别" in joined):
        return CONTENT_CLASSIFICATION_GRADING_TABLE
    if _has_hierarchy_header(joined) or "分类目录" in joined:
        return CONTENT_CLASSIFICATION_CATALOG
    if re.search(r"分类工作|分级工作|按照第\d+章|参考附录|指南", joined):
        return CONTENT_RULE_GUIDANCE
    return CONTENT_UNKNOWN


def build_description_query_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add(value: object) -> None:
        term = str(value or "").strip()
        if term and term not in terms:
            terms.append(term)

    for level in string_list(row.get("path_levels")):
        add(level)
    for example in string_list(row.get("data_range_examples")):
        add(example)
    return terms


def resource_type_terms_for_row(row: dict[str, Any]) -> list[str]:
    values = [
        row.get("resource_type"),
        row.get("_context_table_title"),
        row.get("table_title"),
    ]
    values.extend(string_list(row.get("path_levels")))
    joined = " ".join(str(value or "") for value in values)
    return [term for term in DEFAULT_RESOURCE_TYPE_TERMS if term in joined]


def is_resource_type_definition(text: str) -> bool:
    return any(f"{term}类数据" in str(text or "") for term in DEFAULT_RESOURCE_TYPE_TERMS)


def _context_source(context: dict[str, Any], role: str, reason: str) -> dict[str, Any]:
    return EvidenceSource(
        source_type=str(context.get("kind") or "context"),
        text=str(context.get("text") or "").strip(),
        role=role,
        unit_id=str(context.get("unit_id") or ""),
        line_start=context.get("line_start"),
        line_end=context.get("line_end"),
        score=int(context.get("score") or 0),
        reason=reason,
        metadata={
            "context_group": context.get("context_group", ""),
            "table_title": context.get("table_title", ""),
            "path_hint": context.get("path_hint", []),
        },
    ).to_dict()


def _row_source(source_type: str, value: object, role: str, reason: str) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if not text:
        return None
    return EvidenceSource(source_type=source_type, text=text, role=role, reason=reason).to_dict()


def _append_if_present(target: list[dict[str, Any]], source: dict[str, Any] | None) -> None:
    if source and source.get("text"):
        target.append(source)


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sources:
        key = (
            str(source.get("source_type") or ""),
            clean_text(source.get("text")),
            str(source.get("role") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def build_row_evidence_pack(row: dict[str, Any], context_pack: dict[str, Any]) -> dict[str, Any]:
    description_sources: list[dict[str, Any]] = []
    grading_sources: list[dict[str, Any]] = []
    excluded_sources: list[dict[str, Any]] = []

    for level in string_list(row.get("path_levels")):
        _append_if_present(
            description_sources,
            _row_source("row_field", level, ROLE_CLASSIFICATION_PATH, "classification path level"),
        )
    for example in string_list(row.get("data_range_examples")):
        _append_if_present(
            description_sources,
            _row_source("row_field", example, ROLE_DESCRIPTION_EVIDENCE, "data range/example"),
        )

    for field_name in ["processing_degree", "impact_object", "impact_degree"]:
        _append_if_present(
            grading_sources,
            _row_source("row_field", row.get(field_name), ROLE_GRADING_FACTOR, field_name),
        )
    _append_if_present(
        grading_sources,
        _row_source("row_field", row.get("recommended_grade"), ROLE_GRADE_RESULT, "recommended grade"),
    )

    for context in context_pack.get("primary_contexts") or []:
        text = str(context.get("text") or "")
        if contains_grade_or_risk(text):
            compact_text = clean_text(text)
            path_hits = [
                level
                for level in string_list(row.get("path_levels"))
                if clean_text(level) and clean_text(level) in compact_text
            ]
            example_hits = [
                example
                for example in string_list(row.get("data_range_examples"))
                if clean_text(example) and clean_text(example) in compact_text
            ]
            for item in path_hits:
                _append_if_present(
                    description_sources,
                    _row_source("context_excerpt", item, ROLE_CLASSIFICATION_PATH, "path excerpt from primary row"),
                )
            for item in example_hits:
                _append_if_present(
                    description_sources,
                    _row_source("context_excerpt", item, ROLE_DESCRIPTION_EVIDENCE, "example excerpt from primary row"),
                )
            _append_if_present(
                excluded_sources,
                _context_source(context, ROLE_METADATA_OR_UNKNOWN, "primary row contains grading or risk text"),
            )
            continue
        _append_if_present(
            description_sources,
            _context_source(context, ROLE_DESCRIPTION_EVIDENCE, "primary row context"),
        )

    for context in context_pack.get("definition_contexts") or []:
        if contains_grade_or_risk(str(context.get("text") or "")):
            _append_if_present(
                excluded_sources,
                _context_source(context, ROLE_METADATA_OR_UNKNOWN, "definition context contains risk text"),
            )
            continue
        _append_if_present(
            description_sources,
            _context_source(context, ROLE_DESCRIPTION_EVIDENCE, "definition context"),
        )

    for context in context_pack.get("sibling_contexts") or []:
        if contains_grade_or_risk(str(context.get("text") or "")):
            _append_if_present(
                excluded_sources,
                _context_source(context, ROLE_METADATA_OR_UNKNOWN, "sibling context contains grading or risk text"),
            )
            continue
        _append_if_present(
            description_sources,
            _context_source(context, ROLE_DESCRIPTION_EVIDENCE, "sibling context"),
        )

    for context in context_pack.get("excluded_contexts") or []:
        _append_if_present(
            excluded_sources,
            _context_source(context, ROLE_METADATA_OR_UNKNOWN, "excluded by retrieval"),
        )

    warnings = list(context_pack.get("retrieval_warnings") or [])
    if not description_sources:
        warnings.append("missing_description_sources")

    return {
        "description_sources": _dedupe_sources(description_sources),
        "grading_sources": _dedupe_sources(grading_sources),
        "excluded_sources": _dedupe_sources(excluded_sources),
        "warnings": list(dict.fromkeys(warnings)),
    }
