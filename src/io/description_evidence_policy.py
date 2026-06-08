from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


EMPTY_DATA_RANGE_MARKERS = {"", "-", "—", "－", "一"}
FALLBACK_DESCRIPTION_REASON = "兜底类目且数据范围为空，缺少可支撑分类说明的行级证据。"


@dataclass(frozen=True)
class InsufficientDescriptionDecision:
    force: bool
    reason: str = ""


def clean_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def clean_label_text(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"[，,。；;：:、/\\|（）()〈〉<>《》\"'“”‘’\[\]【】.\-—－_]+", "", text)
    text = re.sub(r"\d+(?:\.\d+)*", "", text)
    return text


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def is_empty_data_range(value: object) -> bool:
    return clean_text(value) in EMPTY_DATA_RANGE_MARKERS


def is_fallback_leaf(path_levels: object) -> bool:
    levels = string_list(path_levels)
    if not levels:
        return False
    leaf = clean_text(levels[-1]).strip("“\"'")
    return bool(re.fullmatch(r"\d{3,4}其他", leaf))


def _row_value(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def has_specific_description_evidence(row: Any) -> bool:
    return any(
        not is_empty_data_range(item)
        for item in string_list(_row_value(row, "data_range_examples"))
    )


def is_label_only_evidence(row: Any, evidence_quote: object) -> bool:
    quote = clean_label_text(evidence_quote)
    if not quote:
        return False
    levels = string_list(_row_value(row, "path_levels"))
    labels = [clean_label_text(level) for level in levels]
    labels = [label for label in labels if label]
    if quote in set(labels):
        return True
    return bool(labels) and quote == clean_label_text("".join(levels))


def is_generic_grade_evidence(evidence_quote: object) -> bool:
    text = clean_text(evidence_quote)
    if not text:
        return False
    grade_terms = ["核心数据", "重要数据", "一般数据", "敏感个人信息", "个人信息"]
    threshold_terms = ["万人", "及以上", "特定群体", "特定区域", "国家安全", "公共健康"]
    return any(term in text for term in grade_terms) and any(term in text for term in threshold_terms)


def should_reject_label_only_description(row: Any, proposed_description: object, evidence_quote: object) -> InsufficientDescriptionDecision:
    description = clean_label_text(proposed_description)
    labels = [clean_label_text(level) for level in string_list(_row_value(row, "path_levels"))]
    labels = [label for label in labels if label]
    if description and description in set(labels):
        return InsufficientDescriptionDecision(True, "分类说明仅重复分类标题，缺少解释性行级证据。")
    if is_label_only_evidence(row, evidence_quote):
        return InsufficientDescriptionDecision(True, "引用证据仅包含分类标题，无法支撑解释性说明。")
    return InsufficientDescriptionDecision(False, "")


def should_reject_summarized_description(row: Any, proposed_description: object, evidence_quote: object) -> InsufficientDescriptionDecision:
    label_only = should_reject_label_only_description(row, proposed_description, evidence_quote)
    if label_only.force:
        return label_only
    if is_generic_grade_evidence(evidence_quote):
        return InsufficientDescriptionDecision(True, "引用证据仅包含通用分级标准，无法支撑该分类的业务说明。")
    return InsufficientDescriptionDecision(False, "")


def should_force_insufficient_description(row: Any) -> InsufficientDescriptionDecision:
    if is_fallback_leaf(_row_value(row, "path_levels")) and not has_specific_description_evidence(row):
        return InsufficientDescriptionDecision(True, FALLBACK_DESCRIPTION_REASON)
    return InsufficientDescriptionDecision(False, "")
