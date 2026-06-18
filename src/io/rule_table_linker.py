from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any


INSUFFICIENT_DESCRIPTION = "证据不足，无法从当前文档确定"
STOP_TERMS = {
    "",
    "数据",
    "分类",
    "分级",
    "信息",
    "资源",
    "其他",
    "类数据",
    "数据库",
}


@dataclass(frozen=True)
class RuleTableReference:
    name: str
    source_type: str
    path: str
    rows: list[dict[str, Any]]
    reuse_policy: str = "assist"
    reference_trust_level: str = "auxiliary"


@dataclass(frozen=True)
class RuleTableMatch:
    reference_name: str
    reference_type: str
    reference_file: str
    reference_row_id: str
    reference_path: list[str]
    score: float
    shared_terms: list[str]
    reference_description_source: str


@dataclass(frozen=True)
class RuleTableLink:
    current_row_id: str
    current_path: list[str]
    current_description_source: str
    matches: list[RuleTableMatch]


def load_rule_table_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"rule_table must contain a JSON object: {path}")
    rows = data.get("classification_rows") or data.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError(f"rule_table rows must be a JSON array: {path}")
    return [row for row in rows if isinstance(row, dict)]


def links_to_dicts(links: list[RuleTableLink]) -> list[dict[str, Any]]:
    return [asdict(link) for link in links]


def _compact_text(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^[A-Za-z]?[）)、.．]?", "", text)
    text = re.sub(r"^\d+(?:[.．]\d+)*", "", text)
    return text


def _path_levels(row: dict[str, Any]) -> list[str]:
    value = row.get("path_levels") or row.get("path") or []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _list_values(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _terms_from_text(value: object) -> set[str]:
    text = _compact_text(value)
    if not text or text == INSUFFICIENT_DESCRIPTION:
        return set()
    parts = re.split(r"[，,。；;：:、/\\|（）()《》〈〉<>【】\[\]\"'“”‘’\s]+", text)
    terms = {_compact_text(part) for part in parts}
    terms = {term for term in terms if len(term) >= 2 and term not in STOP_TERMS}
    if len(text) <= 12 and len(text) >= 2 and text not in STOP_TERMS:
        terms.add(text)
    return {term for term in terms if len(term) >= 2 and term not in STOP_TERMS}


def _row_terms(row: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for level in _path_levels(row):
        terms.update(_terms_from_text(level))
    for example in _list_values(row.get("data_range_examples")):
        terms.update(_terms_from_text(example))
    if row.get("description_source") != "insufficient":
        terms.update(_terms_from_text(row.get("description")))
    return terms


def _score_terms(current_terms: set[str], reference_terms: set[str]) -> tuple[float, list[str]]:
    shared = current_terms & reference_terms
    if not shared:
        return 0.0, []
    coverage = len(shared) / max(1, min(len(current_terms), len(reference_terms)))
    containment = len(shared) / max(1, len(current_terms | reference_terms))
    score = round((coverage * 0.75) + (containment * 0.25), 4)
    shared_terms = sorted(shared, key=lambda term: (-len(term), term))[:8]
    return score, shared_terms


def build_rule_table_links(
    current_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    top_k: int = 3,
    min_score: float = 0.5,
    max_links: int | None = None,
) -> list[RuleTableLink]:
    reference = RuleTableReference(
        name="reference",
        source_type="existing_rule_table",
        path="",
        rows=reference_rows,
    )
    return build_rule_table_links_from_references(
        current_rows=current_rows,
        references=[reference],
        top_k=top_k,
        min_score=min_score,
        max_links=max_links,
    )


def build_rule_table_links_from_references(
    current_rows: list[dict[str, Any]],
    references: list[RuleTableReference],
    top_k: int = 3,
    min_score: float = 0.5,
    max_links: int | None = None,
) -> list[RuleTableLink]:
    reference_index = [
        (reference, row, _row_terms(row))
        for reference in references
        for row in reference.rows
    ]
    links: list[RuleTableLink] = []

    for current in current_rows:
        current_terms = _row_terms(current)
        if not current_terms:
            continue
        matches: list[RuleTableMatch] = []
        for reference, reference_row, reference_terms in reference_index:
            score, shared_terms = _score_terms(current_terms, reference_terms)
            if score < min_score:
                continue
            matches.append(
                RuleTableMatch(
                    reference_name=reference.name,
                    reference_type=reference.source_type,
                    reference_file=reference.path,
                    reference_row_id=str(reference_row.get("row_id") or ""),
                    reference_path=_path_levels(reference_row),
                    score=score,
                    shared_terms=shared_terms,
                    reference_description_source=str(reference_row.get("description_source") or ""),
                )
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        if matches:
            links.append(
                RuleTableLink(
                    current_row_id=str(current.get("row_id") or ""),
                    current_path=_path_levels(current),
                    current_description_source=str(current.get("description_source") or ""),
                    matches=matches[: max(1, top_k)],
                )
            )

    links.sort(key=lambda item: item.matches[0].score if item.matches else 0.0, reverse=True)
    if max_links is not None:
        return links[: max(0, max_links)]
    return links


def render_rule_table_link_markdown(links: list[RuleTableLink]) -> str:
    lines = [
        "# Rule Table Link Report",
        "",
        f"- Linked current rows: {len(links)}",
        "- This is a read-only similarity report. It does not modify generated rule tables.",
        "- Reference matches are review hints only; they are not current-document evidence.",
        "",
    ]
    for index, link in enumerate(links, start=1):
        lines.append(f"## {index}. {' / '.join(link.current_path)}")
        lines.append(f"- current_row_id: `{link.current_row_id}`")
        lines.append(f"- current_description_source: `{link.current_description_source}`")
        for match in link.matches:
            shared = ", ".join(match.shared_terms)
            reference_file = match.reference_file or "(none)"
            lines.append(
                "- match: "
                f"{' / '.join(match.reference_path)} "
                f"(score={match.score}, reference={match.reference_name}, "
                f"type={match.reference_type}, file={reference_file}, "
                f"source={match.reference_description_source}, shared: {shared})"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
