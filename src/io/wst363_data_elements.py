from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ENTRY_RE = re.compile(
    r"(?:^|\n)数据元标识符\s+(DE\d{2}\.\d{2}\.\d{3}\.\d{2})\s*\n"
    r"(?P<body>.*?)(?=\n数据元标识符\s+DE\d{2}\.\d{2}\.\d{3}\.\d{2}|\Z)",
    re.S,
)
FIELD_LABELS = [
    "数据元名称",
    "定义",
    "数据元值的数据类型",
    "表示格式",
    "数据元允许值",
]
VALUE_DOMAIN_RE = re.compile(r"WS/T\s+364\.(\d+)\s+(CV\d{2}\.\d{2}\.\d{3})")
STANDARD_RE = re.compile(r"WS/T\s+363\.(\d+)—(\d{4})")
TITLE_RE = re.compile(r"第\s*(\d+)\s*部分[:：]\s*([^\n\r]+)")
HEADER_LINE_RE = re.compile(r"^\s*(?:WS/T\s+363\.\d+—\d{4}|[IVX]+|\d+)\s*$")
INLINE_HEADER_RE = re.compile(r"\bWS/T\s+363\.\d+—\d{4}\s+\d+\b")


def _strip_noise_lines(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if HEADER_LINE_RE.fullmatch(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _clean_text(value: str) -> str:
    text = INLINE_HEADER_RE.sub("", value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _field_value(body: str, label: str) -> str:
    starts: list[tuple[int, str]] = []
    for field_label in FIELD_LABELS:
        match = re.search(rf"(?m)^({re.escape(field_label)})\s*", body)
        if match:
            starts.append((match.start(), field_label))
    starts.sort()

    for index, (start, field_label) in enumerate(starts):
        if field_label != label:
            continue
        label_end = start + len(field_label)
        end = starts[index + 1][0] if index + 1 < len(starts) else len(body)
        return _clean_text(body[label_end:end])
    return ""


def _value_domain_refs(allowed_values: str) -> list[str]:
    refs = []
    for part, code in VALUE_DOMAIN_RE.findall(allowed_values):
        refs.append(f"WS/T 364.{part}:{code}")
    return refs


def _part_string(part: str) -> str:
    return f"{int(part):02d}"


def parse_wst363_data_elements(text: str, source_path: str = "") -> list[dict[str, Any]]:
    cleaned_text = _strip_noise_lines(text)
    standard_match = STANDARD_RE.search(text)
    source_part = f"WS/T 363.{standard_match.group(1)}—{standard_match.group(2)}" if standard_match else ""

    elements: list[dict[str, Any]] = []
    for match in ENTRY_RE.finditer(cleaned_text):
        body = match.group("body")
        allowed_values = _field_value(body, "数据元允许值")
        elements.append(
            {
                "element_code": match.group(1),
                "element_name": _field_value(body, "数据元名称"),
                "definition": _field_value(body, "定义"),
                "data_type": _field_value(body, "数据元值的数据类型"),
                "display_format": _field_value(body, "表示格式"),
                "allowed_values": allowed_values,
                "value_domain_refs": _value_domain_refs(allowed_values),
                "source_part": source_part,
                "source_path": source_path,
                "extraction_status": "parsed",
            }
        )
    return elements


def build_wst363_payload(text: str, source_path: str = "") -> dict[str, Any]:
    standard_match = STANDARD_RE.search(text)
    title_match = TITLE_RE.search(text)
    part = _part_string(standard_match.group(1)) if standard_match else ""
    standard = f"WS/T 363.{standard_match.group(1)}—{standard_match.group(2)}" if standard_match else ""
    title = _clean_text(title_match.group(2)) if title_match else ""
    elements = parse_wst363_data_elements(text, source_path=source_path)
    return {
        "standard": standard,
        "part": part,
        "title": title,
        "source_path": source_path,
        "extraction_status": "parsed",
        "element_count": len(elements),
        "elements": elements,
    }


def write_wst363_payload(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = build_wst363_payload(
        input_path.read_text(encoding="utf-8"),
        source_path=str(input_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
