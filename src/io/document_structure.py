from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class StructureSignal:
    kind: str
    title: str
    confidence: float
    line_number: int | None = None


APPENDIX_HEADING_RE = re.compile(r"^附\s*录\s*([A-ZＡ-Ｚ])\s*$", re.IGNORECASE)
CONTINUED_TABLE_TITLE_RE = re.compile(
    r"^(?:续\s*表\s*[A-ZＡ-Ｚ]\s*\.?\s*\d+|表\s*[A-ZＡ-Ｚ]\s*\.?\s*\d+\s*[（(]\s*续\s*[）)]).+",
    re.IGNORECASE,
)
TABLE_TITLE_RE = re.compile(r"^表\s*[A-ZＡ-Ｚ]\s*\.?\s*\d+\s+.+", re.IGNORECASE)
CLASSIFICATION_TITLE_RE = re.compile(r"^[\u4e00-\u9fff]{2,20}分类\s*$")
HIERARCHY_HEADER_PATTERNS = [
    re.compile(r"类\s+项\s+目\s+数据范围及示例\s+数据加工程度\s+影响对象\s+影响程度\s+数据级别"),
    re.compile(r"类(?:（[^）]+）|\([^)]*\))?\s+项(?:（[^）]+）|\([^)]*\))?\s+目(?:（[^）]+）|\([^)]*\))?"),
    re.compile(r"资源属性\s+类.*项.*目"),
]


def _normalize_latin(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _compact_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _appendix_title(line: str) -> str | None:
    match = APPENDIX_HEADING_RE.match(line.strip())
    if not match:
        return None
    marker = _normalize_latin(match.group(1)).upper()
    return f"附录 {marker}"


def detect_structure_signal(line: str, line_number: int | None = None) -> StructureSignal | None:
    stripped = line.strip()
    if not stripped:
        return None

    appendix_title = _appendix_title(stripped)
    if appendix_title:
        return StructureSignal(
            kind="appendix_heading",
            title=appendix_title,
            confidence=0.98,
            line_number=line_number,
        )

    if CONTINUED_TABLE_TITLE_RE.match(stripped):
        return StructureSignal(
            kind="continued_table_title",
            title=_compact_spaces(stripped),
            confidence=0.95,
            line_number=line_number,
        )

    if TABLE_TITLE_RE.match(stripped):
        return StructureSignal(
            kind="table_title",
            title=_compact_spaces(stripped),
            confidence=0.95,
            line_number=line_number,
        )

    if CLASSIFICATION_TITLE_RE.match(stripped):
        return StructureSignal(
            kind="classification_title",
            title=_compact_spaces(stripped),
            confidence=0.9,
            line_number=line_number,
        )

    compact = _compact_spaces(stripped)
    if any(pattern.search(compact) for pattern in HIERARCHY_HEADER_PATTERNS):
        return StructureSignal(
            kind="hierarchy_header",
            title=compact,
            confidence=0.9,
            line_number=line_number,
        )

    return None


def build_structure_context(
    section_title: str,
    header_text: str = "",
    page_number: int | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "section_title": section_title,
        "appendix_heading": "",
        "classification_title": "",
        "table_title": "",
        "hierarchy_header": "",
        "page_number": page_number,
        "line_span": {"start": line_start, "end": line_end},
    }

    for part in [item.strip() for item in section_title.split("/") if item.strip()]:
        signal = detect_structure_signal(part)
        if signal is None:
            continue
        if signal.kind == "appendix_heading":
            context["appendix_heading"] = signal.title
        elif signal.kind == "classification_title":
            context["classification_title"] = signal.title
        elif signal.kind in {"table_title", "continued_table_title"}:
            context["table_title"] = signal.title

    header_signal = detect_structure_signal(header_text)
    if header_signal and header_signal.kind == "hierarchy_header":
        context["hierarchy_header"] = header_signal.title

    return context
