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
