from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.agent_state import (  # noqa: E402
    AgentState,
    ClassificationRow,
    ClassificationSchema,
    EvidenceRef,
    GradeDefinition,
    SourceDocument,
    ValidationIssue,
)
from src.llm.client import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleLLMClient  # noqa: E402
from src.output.exporter import export_outputs  # noqa: E402
from src.steps.description_context_kb import enhance_descriptions_with_context  # noqa: E402


DESCRIPTION_CONTEXT_ENV = {
    "DESCRIPTION_CONTEXT_ENABLED",
    "DESCRIPTION_CONTEXT_MODE",
    "DESCRIPTION_CONTEXT_LIMIT",
    "DESCRIPTION_CONTEXT_BATCH_SIZE",
}


def _dataclass_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _evidence_refs(value: object) -> list[EvidenceRef]:
    if not isinstance(value, list):
        return []
    return [
        EvidenceRef(**_dataclass_kwargs(EvidenceRef, item))
        for item in value
        if isinstance(item, dict)
    ]


def _classification_schema(value: object) -> ClassificationSchema | None:
    if not isinstance(value, dict):
        return None
    data = _dataclass_kwargs(ClassificationSchema, value)
    data["evidence_refs"] = _evidence_refs(value.get("evidence_refs"))
    return ClassificationSchema(**data)


def _classification_rows(value: object) -> list[ClassificationRow]:
    if not isinstance(value, list):
        return []
    rows: list[ClassificationRow] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        data = _dataclass_kwargs(ClassificationRow, item)
        data["evidence_refs"] = _evidence_refs(item.get("evidence_refs"))
        rows.append(ClassificationRow(**data))
    return rows


def _grade_scheme(value: object) -> list[GradeDefinition]:
    if not isinstance(value, list):
        return []
    grades: list[GradeDefinition] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        data = _dataclass_kwargs(GradeDefinition, item)
        data["evidence_refs"] = _evidence_refs(item.get("evidence_refs"))
        grades.append(GradeDefinition(**data))
    return grades


def _validation_issues(value: object) -> list[ValidationIssue]:
    if not isinstance(value, list):
        return []
    return [
        ValidationIssue(**_dataclass_kwargs(ValidationIssue, item))
        for item in value
        if isinstance(item, dict)
    ]


def load_state_from_rule_table(txt_path: Path, rule_table_path: Path) -> AgentState:
    resolved_txt = txt_path.expanduser().resolve()
    resolved_rule_table = rule_table_path.expanduser().resolve()
    text = resolved_txt.read_text(encoding="utf-8")
    data = json.loads(resolved_rule_table.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"rule_table must contain a JSON object: {resolved_rule_table}")

    return AgentState(
        task="Rerun description context only from an existing rule_table.json.",
        task_type="rerun_description_context",
        input_files=[str(resolved_txt)],
        documents=[
            SourceDocument(
                doc_id="doc_1",
                doc_name=resolved_txt.name,
                file_path=str(resolved_txt),
                raw_text=text,
            )
        ],
        classification_schema=_classification_schema(data.get("classification_schema")),
        grade_scheme=_grade_scheme(data.get("grade_scheme")),
        classification_rows=_classification_rows(data.get("classification_rows")),
        validation_issues=_validation_issues(data.get("validation_issues")),
    )


@contextmanager
def _description_context_env(mode: str, limit: int, batch_size: int) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in DESCRIPTION_CONTEXT_ENV}
    os.environ["DESCRIPTION_CONTEXT_ENABLED"] = "true"
    os.environ["DESCRIPTION_CONTEXT_MODE"] = mode
    os.environ["DESCRIPTION_CONTEXT_LIMIT"] = str(max(0, int(limit)))
    os.environ["DESCRIPTION_CONTEXT_BATCH_SIZE"] = str(max(1, int(batch_size)))
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def rerun_description_context(
    txt_path: Path,
    rule_table_path: Path,
    output_dir: Path,
    llm_client: Any,
    mode: str = "v2",
    limit: int = 20,
    batch_size: int = 20,
) -> AgentState:
    if mode not in {"v1", "v2"}:
        raise ValueError("mode must be 'v1' or 'v2'")

    state = load_state_from_rule_table(txt_path, rule_table_path)
    state.llm_enabled = True
    state.llm_model = str(getattr(llm_client, "model", "") or DEFAULT_MODEL)
    state.llm_base_url = str(getattr(llm_client, "base_url", "") or DEFAULT_BASE_URL)

    with _description_context_env(mode, limit, batch_size):
        state = enhance_descriptions_with_context(state, llm_client, output_dir=str(output_dir))
    return export_outputs(state, str(output_dir))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun only description-context enhancement from an existing rule_table.json."
    )
    parser.add_argument("--txt", required=True, help="Source TXT document path.")
    parser.add_argument("--rule-table", required=True, help="Existing rule_table.json path.")
    parser.add_argument("--out", required=True, help="Output directory for updated artifacts.")
    parser.add_argument("--mode", choices=["v1", "v2"], default="v2", help="Description context mode.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum weak-description rows to regenerate.")
    parser.add_argument("--batch-size", type=int, default=20, help="Rows per description-generation LLM call.")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-model", default=None, help="LLM model name.")
    args = parser.parse_args()

    llm_client = OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=args.llm_model)
    state = rerun_description_context(
        txt_path=Path(args.txt),
        rule_table_path=Path(args.rule_table),
        output_dir=Path(args.out),
        llm_client=llm_client,
        mode=args.mode,
        limit=max(0, args.limit),
        batch_size=max(1, args.batch_size),
    )
    print(
        "Rerun complete.",
        f"rows={len(state.classification_rows)}",
        f"output_dir={Path(args.out).expanduser().resolve()}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
