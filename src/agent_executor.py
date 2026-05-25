from __future__ import annotations

from pathlib import Path

from .agent_state import AgentState
from .task_router import detect_task_type
from .tools import execute_tool


def create_plan(task_type: str) -> list[dict]:
    if task_type == "generate_rule_tree_from_docs":
        return [
            {"tool": "parse_documents", "label": "Parse source documents"},
            {"tool": "chunk_documents", "label": "Chunk documents"},
            {"tool": "extract_concepts", "label": "Extract candidate concepts"},
            {"tool": "discover_classification_dimensions", "label": "Discover classification dimensions"},
            {"tool": "build_taxonomy", "label": "Build candidate taxonomy"},
            {"tool": "extract_grade_scheme", "label": "Extract grade scheme"},
            {"tool": "generate_node_descriptions", "label": "Generate grounded descriptions"},
            {"tool": "assign_grades_to_nodes", "label": "Assign grades where supported"},
            {"tool": "generate_node_rules", "label": "Generate matching rules"},
            {"tool": "validate_grounding", "label": "Validate grounding"},
            {"tool": "export_outputs", "label": "Export outputs"},
        ]
    return []


def run_agent(user_task: str, input_files: list[str], output_dir: str = "outputs") -> AgentState:
    task_type = detect_task_type(user_task)
    if task_type == "unknown":
        task_type = "generate_rule_tree_from_docs"

    resolved_inputs = [str(Path(file_path).expanduser().resolve()) for file_path in input_files]
    state = AgentState(task=user_task, task_type=task_type, input_files=resolved_inputs)
    plan = create_plan(task_type)
    if not plan:
        raise ValueError(f"Unsupported task type for MVP: {task_type}")

    print(f"Task type: {task_type}")
    for index, step in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {step['label']} ({step['tool']})")
        if step["tool"] == "export_outputs":
            state = execute_tool(step["tool"], state, output_dir=output_dir)
        else:
            state = execute_tool(step["tool"], state)
    return state

