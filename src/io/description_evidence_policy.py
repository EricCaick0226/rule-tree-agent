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


def should_force_insufficient_description(row: Any) -> InsufficientDescriptionDecision:
    if is_fallback_leaf(_row_value(row, "path_levels")) and not has_specific_description_evidence(row):
        return InsufficientDescriptionDecision(True, FALLBACK_DESCRIPTION_REASON)
    return InsufficientDescriptionDecision(False, "")
