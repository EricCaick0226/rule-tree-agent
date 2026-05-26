from __future__ import annotations

from pathlib import Path

from ..core.agent_state import AgentState
from ..io.document_parser import chunk_documents, parse_documents
from ..io.evidence_index import build_evidence_index
from ..llm.client import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleLLMClient
from ..output.exporter import export_outputs
from ..steps.concept_normalizer import normalize_concepts_with_llm
from ..steps.dimension_analyzer import discover_dimensions_with_llm
from ..steps.evidence_claim_extractor import extract_evidence_claims_with_llm
from ..steps.grading_analyzer import analyze_grading_with_llm
from ..steps.node_describer import describe_nodes_with_llm
from ..steps.rule_synthesizer import synthesize_rules_with_llm
from ..steps.taxonomy_synthesizer import synthesize_taxonomy_with_llm
from ..validation.grounding_validator import validate_grounding


class LLMGenerationError(RuntimeError):
    pass


def create_plan(task_type: str) -> list[dict]:
    if task_type == "generate_rule_tree_from_docs":
        return [
            {"tool": "parse_documents", "label": "Parse source documents"},
            {"tool": "chunk_documents", "label": "Chunk documents"},
            {"tool": "build_evidence_index", "label": "Build evidence index"},
            {"tool": "extract_evidence_claims_with_llm", "label": "Extract evidence claims with LLM"},
            {"tool": "normalize_concepts_with_llm", "label": "Normalize concepts with LLM"},
            {"tool": "discover_dimensions_with_llm", "label": "Discover classification dimensions with LLM"},
            {"tool": "synthesize_taxonomy_with_llm", "label": "Synthesize taxonomy with LLM"},
            {"tool": "describe_nodes_with_llm", "label": "Describe nodes with LLM"},
            {"tool": "analyze_grading_with_llm", "label": "Analyze grading with LLM"},
            {"tool": "synthesize_rules_with_llm", "label": "Synthesize matching rules with LLM"},
            {"tool": "validate_grounding", "label": "Validate grounding"},
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
        state = extract_evidence_claims_with_llm(state, llm_client)
    elif tool_name == "normalize_concepts_with_llm":
        state = normalize_concepts_with_llm(state, llm_client)
    elif tool_name == "discover_dimensions_with_llm":
        state = discover_dimensions_with_llm(state, llm_client)
    elif tool_name == "synthesize_taxonomy_with_llm":
        state = synthesize_taxonomy_with_llm(state, llm_client)
    elif tool_name == "describe_nodes_with_llm":
        state = describe_nodes_with_llm(state, llm_client)
    elif tool_name == "analyze_grading_with_llm":
        state = analyze_grading_with_llm(state, llm_client)
    elif tool_name == "synthesize_rules_with_llm":
        state = synthesize_rules_with_llm(state, llm_client)
    elif tool_name == "validate_grounding":
        state.validation_issues = validate_grounding(state)
    elif tool_name == "export_outputs":
        state = export_outputs(state, output_dir)
    else:
        raise ValueError(f"Unknown step: {tool_name}")
    return state


def _run_llm_steps(state: AgentState, output_dir: str, llm_client) -> AgentState:
    plan = create_plan(state.task_type)
    print(f"Task type: {state.task_type}")
    print(f"LLM: required model={state.llm_model} base_url={state.llm_base_url}")
    for index, step in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {step['label']} ({step['tool']})")
        if step["tool"].endswith("_with_llm"):
            try:
                state = _run_step(step["tool"], state, output_dir, llm_client)
                state.llm_used = True
            except LLMGenerationError:
                raise
            except Exception as exc:
                state.llm_error = str(exc)
                raise LLMGenerationError(str(exc)) from exc
            if step["tool"] == "synthesize_taxonomy_with_llm" and not state.nodes:
                raise LLMGenerationError("LLM did not return any candidate nodes.")
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
