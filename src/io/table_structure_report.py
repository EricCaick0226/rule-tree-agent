from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data_classification_profile import classify_content_type, classify_field_role
from .table_segmenter import TableSegment


@dataclass
class TableStructureItem:
    segment_id: str
    source_chunk_id: str
    doc_name: str
    section_title: str
    table_title: str
    hierarchy_header: str
    content_type: str
    line_span: dict[str, int | None]
    field_roles: dict[str, str]
    flattened_row_hints_count: int
    review_notes: list[str]


@dataclass
class TableStructureReport:
    total_segments: int
    items: list[TableStructureItem]


def _split_header_fields(header_text: str) -> list[str]:
    return [field for field in str(header_text or "").split() if field]


def _field_roles(header_text: str) -> dict[str, str]:
    return {field: classify_field_role(field) for field in _split_header_fields(header_text)}


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


def _table_title(segment: TableSegment) -> str:
    return str(segment.structure_context.get("table_title") or "")


def _hierarchy_header(segment: TableSegment) -> str:
    return str(segment.structure_context.get("hierarchy_header") or segment.header_text or "")


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


def render_table_structure_markdown(report: TableStructureReport) -> str:
    lines = [
        "# Table Structure Report",
        f"- Total segments: {report.total_segments}",
        "- This is a read-only diagnostic report. It does not modify row extraction or generated outputs.",
    ]
    for item in report.items:
        lines.extend(
            [
                "",
                f"## {item.segment_id}",
                f"- segment_id: {item.segment_id}",
                f"- section: {item.section_title}",
                f"- content_type: {item.content_type}",
                f"- line_span: {item.line_span['start']}-{item.line_span['end']}",
                f"- header: {item.hierarchy_header}",
                "- field_roles:",
            ]
        )
        for field, role in item.field_roles.items():
            lines.append(f"  - {field} -> {role}")
        lines.append(f"- flattened_row_hints: {item.flattened_row_hints_count}")
        lines.append("- review_notes:")
        for note in item.review_notes:
            lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"
