from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data_classification_profile import classify_content_type, classify_field_role
from .table_segmenter import TableSegment


@dataclass(frozen=True)
class TableStructureItem:
    segment_id: str
    source_chunk_id: str
    doc_name: str
    section_title: str
    table_title: str
    hierarchy_header: str
    content_type: str
    line_span: dict[str, int | None]
    field_roles: list[dict[str, str]]
    flattened_row_hints_count: int
    review_notes: list[str]


@dataclass(frozen=True)
class TableStructureReport:
    total_segments: int
    items: list[TableStructureItem]
    segmentation_mode: str = "all_nonempty_chunks_as_table_candidates"


FILTER_MODE_REVIEWABLE_STRUCTURE_SIGNALS = "reviewable_structure_signals"


def _split_header_fields(header_text: str) -> list[str]:
    return [field for field in str(header_text or "").split() if field]


def _field_roles(header_text: str) -> list[dict[str, str]]:
    return [
        {"field": field, "role": classify_field_role(field)}
        for field in _split_header_fields(header_text)
    ]


def _review_notes(
    table_title: str,
    hierarchy_header: str,
    content_type: str,
    flattened_row_hints_count: int,
) -> list[str]:
    notes: list[str] = []
    if table_title:
        notes.append("detected table title")
    if hierarchy_header:
        notes.append("detected hierarchy header")
    else:
        notes.append("missing header text")
    if content_type == "unknown":
        notes.append("unknown content type")
    if flattened_row_hints_count:
        notes.append("has flattened parent-child code lines")
    return notes


def _structure_context(segment: TableSegment) -> dict[str, object]:
    context = segment.structure_context
    if isinstance(context, dict):
        return context
    return {}


def _table_title(segment: TableSegment) -> str:
    return str(_structure_context(segment).get("table_title") or "")


def _hierarchy_header(segment: TableSegment) -> str:
    return str(_structure_context(segment).get("hierarchy_header") or segment.header_text or "")


def build_table_structure_report(segments: list[TableSegment]) -> TableStructureReport:
    items: list[TableStructureItem] = []
    for segment in segments:
        table_title = _table_title(segment)
        hierarchy_header = _hierarchy_header(segment)
        content_type = classify_content_type(
            header_text=hierarchy_header,
            table_title=table_title,
            text=segment.text,
        )
        flattened_row_hints_count = len(segment.flattened_row_hints)
        items.append(
            TableStructureItem(
                segment_id=segment.segment_id,
                source_chunk_id=segment.source_chunk_id,
                doc_name=segment.doc_name,
                section_title=segment.section_title,
                table_title=table_title,
                hierarchy_header=hierarchy_header,
                content_type=content_type,
                line_span={"start": segment.line_start, "end": segment.line_end},
                field_roles=_field_roles(hierarchy_header),
                flattened_row_hints_count=flattened_row_hints_count,
                review_notes=_review_notes(
                    table_title=table_title,
                    hierarchy_header=hierarchy_header,
                    content_type=content_type,
                    flattened_row_hints_count=flattened_row_hints_count,
                ),
            )
        )
    return TableStructureReport(total_segments=len(items), items=items)


def report_to_dict(report: TableStructureReport) -> dict[str, Any]:
    return asdict(report)


def filter_reviewable_table_structure_items(
    report: TableStructureReport,
) -> list[TableStructureItem]:
    return [
        item
        for item in report.items
        if item.content_type in {"classification_catalog", "classification_grading_table"}
        or bool(item.hierarchy_header)
        or item.flattened_row_hints_count > 0
    ]


def filtered_report_to_dict(report: TableStructureReport) -> dict[str, Any]:
    items = filter_reviewable_table_structure_items(report)
    return {
        "total_segments": report.total_segments,
        "filtered_segments": len(items),
        "segmentation_mode": report.segmentation_mode,
        "filter_mode": FILTER_MODE_REVIEWABLE_STRUCTURE_SIGNALS,
        "items": [asdict(item) for item in items],
    }


def _append_item_markdown(
    lines: list[str],
    item: TableStructureItem,
    include_table: bool = False,
    include_field_roles: bool = False,
    placeholder_empty_header: bool = False,
) -> None:
    header = item.hierarchy_header or "-" if placeholder_empty_header else item.hierarchy_header
    lines.extend(
        [
            "",
            f"## {item.segment_id}",
            f"- segment_id: {item.segment_id}",
            f"- section: {item.section_title}",
        ]
    )
    if include_table:
        lines.append(f"- table: {item.table_title or '-'}")
    lines.extend(
        [
            f"- content_type: {item.content_type}",
            f"- line_span: {item.line_span['start']}-{item.line_span['end']}",
            f"- header: {header}",
        ]
    )
    if include_field_roles:
        lines.append("- field_roles:")
        for field_role in item.field_roles:
            lines.append(f"  - {field_role['field']} -> {field_role['role']}")
        if not item.field_roles:
            lines.append("  - (none)")
    lines.append(f"- flattened_row_hints: {item.flattened_row_hints_count}")
    lines.append("- review_notes:")
    for note in item.review_notes:
        lines.append(f"  - {note}")
    if not item.review_notes:
        lines.append("  - (none)")


def render_table_structure_markdown(report: TableStructureReport) -> str:
    lines = [
        "# Table Structure Report",
        f"- Total segments: {report.total_segments}",
        f"- Segmentation mode: {report.segmentation_mode}",
        "- This is a read-only diagnostic report. It does not modify row extraction or generated outputs.",
    ]
    for item in report.items:
        _append_item_markdown(lines, item, include_field_roles=True)
    return "\n".join(lines) + "\n"


def render_filtered_table_structure_markdown(report: TableStructureReport) -> str:
    items = filter_reviewable_table_structure_items(report)
    lines = [
        "# Filtered Table Structure Report",
        f"- Total segments: {report.total_segments}",
        f"- Filtered segments: {len(items)}",
        f"- Segmentation mode: {report.segmentation_mode}",
        f"- Filter mode: {FILTER_MODE_REVIEWABLE_STRUCTURE_SIGNALS}",
        "- This is a read-only diagnostic report. It does not modify row extraction or generated outputs.",
    ]
    for item in items:
        _append_item_markdown(lines, item, include_table=True, placeholder_empty_header=True)
    return "\n".join(lines) + "\n"
