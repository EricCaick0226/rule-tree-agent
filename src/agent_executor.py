from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_state import AgentState
from .llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleLLMClient
from .task_router import detect_task_type
from .tools import execute_tool


class LLMGenerationError(RuntimeError):
    pass


def create_plan(task_type: str) -> list[dict]:
    if task_type == "generate_rule_tree_from_docs":
        return [
            {"tool": "parse_documents", "label": "Parse source documents"},
            {"tool": "chunk_documents", "label": "Chunk documents"},
            {"tool": "generate_grounded_candidates_with_llm", "label": "Generate grounded candidates with LLM"},
            {"tool": "validate_grounding", "label": "Validate grounding"},
            {"tool": "export_outputs", "label": "Export outputs"},
        ]
    return []


def _run_llm_steps(state: AgentState, output_dir: str, llm_client: Any) -> AgentState:
    plan = create_plan(state.task_type)
    print(f"Task type: {state.task_type}")
    print(f"LLM: required model={state.llm_model} base_url={state.llm_base_url}")
    for index, step in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {step['label']} ({step['tool']})")
        if step["tool"] == "generate_grounded_candidates_with_llm":
            try:
                state = execute_tool(step["tool"], state, llm_client=llm_client)
                if not state.nodes:
                    raise LLMGenerationError("LLM did not return any candidate nodes.")
            except LLMGenerationError:
                raise
            except Exception as exc:
                raise LLMGenerationError(str(exc)) from exc
        elif step["tool"] == "export_outputs":
            state = execute_tool(step["tool"], state, output_dir=output_dir)
        else:
            state = execute_tool(step["tool"], state)
    return state


def run_agent(
    user_task: str,
    input_files: list[str],
    output_dir: str = "outputs",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> AgentState:
    task_type = detect_task_type(user_task)
    if task_type == "unknown":
        task_type = "generate_rule_tree_from_docs"

    resolved_inputs = [str(Path(file_path).expanduser().resolve()) for file_path in input_files]
    base_url = llm_base_url or DEFAULT_BASE_URL
    model = llm_model or DEFAULT_MODEL
    state = AgentState(
        task=user_task,
        task_type=task_type,
        input_files=resolved_inputs,
        llm_enabled=True,
        llm_model=model,
        llm_base_url=base_url,
    )
    if not create_plan(task_type):
        raise ValueError(f"Unsupported task type for MVP: {task_type}")

    llm_client = OpenAICompatibleLLMClient(base_url=base_url, model=model)
    return _run_llm_steps(state, output_dir, llm_client)
