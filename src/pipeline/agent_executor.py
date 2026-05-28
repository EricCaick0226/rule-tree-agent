from __future__ import annotations

from pathlib import Path

from ..core.agent_state import AgentState
from ..io.document_parser import chunk_documents, parse_documents
from ..io.evidence_index import build_evidence_index
from ..llm.client import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleLLMClient
from ..output.exporter import export_outputs
from ..steps.block_classifier import classify_document_blocks_with_llm
from ..steps.classification_row_extractor import extract_classification_rows_with_llm
from ..steps.classification_row_normalizer import normalize_classification_rows
from ..steps.evidence_claim_extractor import extract_evidence_claims_with_llm
from ..steps.grade_definition_extractor import extract_grade_definitions_with_llm
from ..steps.tree_projector import project_tree_from_rows
from ..validation.row_grounding_validator import validate_row_grounding


class LLMGenerationError(RuntimeError):
    pass


SUPPORTED_ROW_FIRST_SUFFIXES = {".txt", ".md"}


def _unsupported_row_first_inputs(input_files: list[str]) -> list[str]:
    return [
        file_path
        for file_path in input_files
        if Path(file_path).suffix.lower() not in SUPPORTED_ROW_FIRST_SUFFIXES
    ]


def _raise_for_unsupported_row_first_inputs(input_files: list[str]) -> None:
    unsupported_files = _unsupported_row_first_inputs(input_files)
    if unsupported_files:
        raise ValueError(
            "row-first MVP only supports .txt and .md inputs; unsupported files: "
            + ", ".join(unsupported_files)
        )


def create_plan(task_type: str) -> list[dict]:
    if task_type == "generate_rule_tree_from_docs":
        return [
            {"tool": "parse_documents", "label": "Parse source documents"},
            {"tool": "chunk_documents", "label": "Chunk documents"},
            {"tool": "build_evidence_index", "label": "Build evidence index"},
            {"tool": "extract_evidence_claims_with_llm", "label": "Extract evidence claims with LLM"},
            {"tool": "classify_document_blocks_with_llm", "label": "Classify document blocks with LLM"},
            {"tool": "extract_classification_rows_with_llm", "label": "Extract classification rows with LLM"},
            {"tool": "extract_grade_definitions_with_llm", "label": "Extract grade definitions with LLM"},
            {"tool": "normalize_classification_rows", "label": "Normalize classification rows"},
            {"tool": "validate_row_grounding", "label": "Validate row grounding"},
            {"tool": "project_tree_from_rows", "label": "Project tree from rows"},
            {"tool": "export_outputs", "label": "Export outputs"},
        ]
    return []


def _run_step(tool_name: str, state: AgentState, output_dir: str, llm_client) -> AgentState:
    if tool_name == "parse_documents":
        state.documents = parse_documents(state.input_files, enable_ocr=state.pdf_ocr_enabled)
    elif tool_name == "chunk_documents":
        state.chunks = chunk_documents(state.documents)
    elif tool_name == "build_evidence_index":
        state = build_evidence_index(state)
    elif tool_name == "extract_evidence_claims_with_llm":
        state = extract_evidence_claims_with_llm(state, llm_client, output_dir)
    elif tool_name == "classify_document_blocks_with_llm":
        state = classify_document_blocks_with_llm(state, llm_client, output_dir=output_dir)
    elif tool_name == "extract_classification_rows_with_llm":
        state = extract_classification_rows_with_llm(state, llm_client, output_dir=output_dir)
    elif tool_name == "extract_grade_definitions_with_llm":
        state = extract_grade_definitions_with_llm(state, llm_client)
    elif tool_name == "normalize_classification_rows":
        state = normalize_classification_rows(state)
    elif tool_name == "validate_row_grounding":
        state.validation_issues = validate_row_grounding(state)
    elif tool_name == "project_tree_from_rows":
        state = project_tree_from_rows(state)
    elif tool_name == "export_outputs":
        state = export_outputs(state, output_dir)
    else:
        raise ValueError(f"Unknown step: {tool_name}")
    return state


def _run_llm_steps(state: AgentState, output_dir: str, llm_client) -> AgentState:
    _raise_for_unsupported_row_first_inputs(state.input_files)

    plan = create_plan(state.task_type)
    print(f"Task type: {state.task_type}")
    print(f"LLM: required model={state.llm_model} base_url={state.llm_base_url}")
    for index, step in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {step['label']} ({step['tool']})")
        if step["tool"].endswith("_with_llm"):
            trace_count = len(state.step_traces)
            try:
                state = _run_step(step["tool"], state, output_dir, llm_client)
                if any(trace.raw_response for trace in state.step_traces[trace_count:]):
                    state.llm_used = True
            except LLMGenerationError:
                raise
            except Exception as exc:
                state.llm_error = str(exc)
                raise LLMGenerationError(str(exc)) from exc
        else:
            state = _run_step(step["tool"], state, output_dir, llm_client)
    return state


def run_agent(
    user_task: str,
    input_files: list[str],
    output_dir: str = "outputs",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    enable_ocr: bool = False,
) -> AgentState:
    resolved_inputs = [str(Path(file_path).expanduser().resolve()) for file_path in input_files]
    _raise_for_unsupported_row_first_inputs(resolved_inputs)
    base_url = llm_base_url or DEFAULT_BASE_URL
    model = llm_model or DEFAULT_MODEL
    state = AgentState(
        task=user_task,
        task_type="generate_rule_tree_from_docs",
        input_files=resolved_inputs,
        llm_enabled=True,
        llm_model=model,
        llm_base_url=base_url,
        pdf_ocr_enabled=enable_ocr,
    )

    llm_client = OpenAICompatibleLLMClient(base_url=base_url, model=model)
    return _run_llm_steps(state, output_dir, llm_client)
