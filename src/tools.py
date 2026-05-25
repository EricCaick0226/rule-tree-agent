from __future__ import annotations

from .agent_state import AgentState
from .concept_extractor import extract_concepts
from .description_generator import generate_node_descriptions
from .dimension_discoverer import discover_classification_dimensions, select_primary_dimension
from .document_parser import chunk_documents, parse_documents
from .evidence_store import collect_unique_evidence
from .exporter import export_outputs
from .grade_assigner import assign_grades_to_nodes
from .grade_scheme_extractor import extract_grade_scheme
from .grounding_validator import validate_grounding
from .rule_generator import generate_node_rules
from .taxonomy_builder import build_taxonomy


TOOL_REGISTRY = {
    "parse_documents": parse_documents,
    "chunk_documents": chunk_documents,
    "extract_concepts": extract_concepts,
    "discover_classification_dimensions": discover_classification_dimensions,
    "build_taxonomy": build_taxonomy,
    "extract_grade_scheme": extract_grade_scheme,
    "generate_node_descriptions": generate_node_descriptions,
    "assign_grades_to_nodes": assign_grades_to_nodes,
    "generate_node_rules": generate_node_rules,
    "validate_grounding": validate_grounding,
    "export_outputs": export_outputs,
}


def _refresh_evidence_refs(state: AgentState) -> None:
    refs = []
    for concept in state.candidate_concepts:
        refs.extend(concept.evidence_refs)
    for dimension in state.classification_dimensions:
        refs.extend(dimension.evidence_refs)
    for grade in state.grade_scheme:
        refs.extend(grade.evidence_refs)
    for node in state.nodes:
        refs.extend(node.evidence_refs)
        refs.extend(node.description_evidence_refs)
        refs.extend(node.grade_evidence_refs)
        for rule in node.rules:
            refs.extend(rule.evidence_refs)
    state.evidence_refs = collect_unique_evidence(refs)


def execute_tool(tool_name: str, state: AgentState, **kwargs) -> AgentState:
    if tool_name == "parse_documents":
        state.documents = parse_documents(state.input_files)
    elif tool_name == "chunk_documents":
        state.chunks = chunk_documents(state.documents)
    elif tool_name == "extract_concepts":
        state.candidate_concepts = extract_concepts(state.chunks)
    elif tool_name == "discover_classification_dimensions":
        state.classification_dimensions = discover_classification_dimensions(
            state.candidate_concepts, state.chunks
        )
        state.selected_dimension = select_primary_dimension(state.classification_dimensions)
    elif tool_name == "build_taxonomy":
        state.nodes = build_taxonomy(
            state.candidate_concepts,
            state.selected_dimension,
            state.chunks,
        )
    elif tool_name == "extract_grade_scheme":
        state.grade_scheme = extract_grade_scheme(state.chunks)
    elif tool_name == "generate_node_descriptions":
        state.nodes = generate_node_descriptions(state.nodes, state.chunks)
    elif tool_name == "assign_grades_to_nodes":
        state.nodes = assign_grades_to_nodes(state.nodes, state.grade_scheme, state.chunks)
    elif tool_name == "generate_node_rules":
        state.nodes = generate_node_rules(state.nodes, state.chunks)
    elif tool_name == "validate_grounding":
        state.validation_issues = validate_grounding(state)
    elif tool_name == "export_outputs":
        output_dir = kwargs.get("output_dir", "outputs")
        state = export_outputs(state, output_dir)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    _refresh_evidence_refs(state)
    return state
