from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .agent_state import (
    AgentState,
    CandidateConcept,
    ClassificationDimension,
    DocumentChunk,
    EvidenceRef,
    GradeDefinition,
    MatchingRule,
    TreeNode,
)
from .evidence_store import create_evidence_ref, dedupe_evidence_refs


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object.")
    return data


def _chunk_payload(chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_name": chunk.doc_name,
            "section_title": chunk.section_title,
            "position": chunk.position,
            "text": chunk.text,
        }
        for chunk in chunks
    ]


def _system_prompt() -> str:
    return """你是一个企业文档证据驱动的分类与分级规则树生成 Agent。

必须遵守：
- 只能使用提供的 document_chunks。
- 不得发明分类名称、等级名称、层级、描述、规则、风险判断或业务背景。
- 每个 concept、dimension、grade、node、description、rule 都必须返回 evidence_chunk_ids。
- 如果证据不足，返回 needs_review=true，并用谨慎文本说明不能从当前文档确定。
- 不要使用任何文档外示例、行业常识、默认安全等级或默认分类。
- 只输出 JSON，不要输出 Markdown 或解释文字。
"""


def _user_prompt(chunks: list[DocumentChunk]) -> str:
    schema = {
        "concepts": [
            {
                "text": "文档原文中的概念名称",
                "concept_type": "heading|list_item|definition|grade|rule_term|other",
                "evidence_chunk_ids": ["doc_1_chunk_1"],
                "confidence": 0.0,
                "needs_review": False,
            }
        ],
        "classification_dimensions": [
            {
                "name": "分类维度名称，必须来自证据",
                "description": "只基于证据的说明",
                "reason": "为什么认为这是分类维度",
                "evidence_chunk_ids": ["doc_1_chunk_1"],
                "confidence": 0.0,
                "needs_review": False,
            }
        ],
        "selected_dimension_name": "如果无法确定则为 null",
        "grade_scheme": [
            {
                "grade_name": "等级名称，必须来自证据",
                "definition": "等级定义，必须来自证据",
                "criteria": ["只写证据支持的条件"],
                "evidence_chunk_ids": ["doc_1_chunk_1"],
                "confidence": 0.0,
                "needs_review": False,
            }
        ],
        "nodes": [
            {
                "name": "节点名称，必须来自证据",
                "path": "父节点 / 子节点；无父节点则为节点名称",
                "parent_path": "父节点路径；根节点为 null",
                "level": 1,
                "description": "节点说明；证据不足时写无法从当前文档确定",
                "description_evidence_chunk_ids": ["doc_1_chunk_1"],
                "grade": "等级名称；无法确定则为 null",
                "grade_evidence_chunk_ids": ["doc_1_chunk_1"],
                "grade_reason": "分级理由；无法确定时说明证据不足",
                "node_evidence_chunk_ids": ["doc_1_chunk_1"],
                "rules": [
                    {
                        "rule_type": "keyword_rule|phrase_rule|context_rule|negative_rule",
                        "conditions": ["必须来自证据的关键词或短语"],
                        "negative_conditions": ["只有文档明确排除时填写"],
                        "evidence_chunk_ids": ["doc_1_chunk_1"],
                        "confidence": 0.0,
                        "needs_review": False,
                    }
                ],
                "confidence": 0.0,
                "needs_review": False,
                "status": "evidence_supported|proposed|insufficient_evidence",
            }
        ],
        "validation_notes": ["可选：说明哪些地方证据不足"],
    }
    return json.dumps(
        {
            "task": "从输入文档生成新的分类与分级候选规则树。不要套用任何默认领域知识。",
            "output_schema": schema,
            "document_chunks": _chunk_payload(chunks),
        },
        ensure_ascii=False,
        indent=2,
    )


def _refs_from_chunk_ids(
    chunk_by_id: dict[str, DocumentChunk],
    chunk_ids: list[Any],
    used_for: str,
    score: float,
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for raw_id in chunk_ids:
        chunk = chunk_by_id.get(str(raw_id))
        if chunk is None:
            continue
        refs.append(create_evidence_ref(chunk, used_for, score))
    return dedupe_evidence_refs(refs)


def _confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return default


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _to_concepts(data: dict[str, Any], chunk_by_id: dict[str, DocumentChunk]) -> list[CandidateConcept]:
    concepts: list[CandidateConcept] = []
    seen: set[str] = set()
    for item in data.get("concepts") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        norm = _normalize(text)
        if norm in seen:
            continue
        seen.add(norm)
        refs = _refs_from_chunk_ids(
            chunk_by_id,
            item.get("evidence_chunk_ids") or [],
            f"llm_concept:{text}",
            0.86,
        )
        concepts.append(
            CandidateConcept(
                concept_id=_stable_id("concept", norm),
                text=text,
                normalized_text=norm,
                concept_type=str(item.get("concept_type") or "llm_extracted"),
                evidence_refs=refs,
                confidence=_confidence(item.get("confidence"), 0.65),
                needs_review=_bool(item.get("needs_review"), not bool(refs)),
            )
        )
    return concepts


def _to_dimensions(
    data: dict[str, Any], chunk_by_id: dict[str, DocumentChunk]
) -> tuple[list[ClassificationDimension], ClassificationDimension | None]:
    dimensions: list[ClassificationDimension] = []
    selected_name = data.get("selected_dimension_name")
    for item in data.get("classification_dimensions") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        refs = _refs_from_chunk_ids(
            chunk_by_id,
            item.get("evidence_chunk_ids") or [],
            f"llm_dimension:{name}",
            0.9,
        )
        dimensions.append(
            ClassificationDimension(
                dimension_id=_stable_id("dim", name),
                name=name,
                description=str(item.get("description") or ""),
                evidence_refs=refs,
                reason=str(item.get("reason") or ""),
                confidence=_confidence(item.get("confidence"), 0.65),
                needs_review=_bool(item.get("needs_review"), not bool(refs)),
            )
        )
    selected = None
    if selected_name:
        selected = next((dim for dim in dimensions if dim.name == selected_name), None)
    if selected is None and dimensions:
        selected = sorted(dimensions, key=lambda dim: (dim.needs_review, -dim.confidence))[0]
    return dimensions, selected


def _to_grades(data: dict[str, Any], chunk_by_id: dict[str, DocumentChunk]) -> list[GradeDefinition]:
    grades: list[GradeDefinition] = []
    seen: set[str] = set()
    for item in data.get("grade_scheme") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("grade_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        refs = _refs_from_chunk_ids(
            chunk_by_id,
            item.get("evidence_chunk_ids") or [],
            f"llm_grade_definition:{name}",
            0.9,
        )
        grades.append(
            GradeDefinition(
                grade_id=_stable_id("grade", name),
                grade_name=name,
                definition=str(item.get("definition") or ""),
                criteria=_string_list(item.get("criteria")),
                evidence_refs=refs,
                confidence=_confidence(item.get("confidence"), 0.65),
                needs_review=_bool(item.get("needs_review"), not bool(refs)),
                status="proposed" if _bool(item.get("needs_review"), not bool(refs)) else "evidence_supported",
            )
        )
    return grades


def _to_nodes(data: dict[str, Any], chunk_by_id: dict[str, DocumentChunk]) -> list[TreeNode]:
    raw_nodes = [item for item in data.get("nodes") or [] if isinstance(item, dict)]
    node_by_path: dict[str, TreeNode] = {}
    pending_parent_paths: dict[str, str | None] = {}

    for item in raw_nodes:
        name = str(item.get("name") or "").strip()
        path = str(item.get("path") or name).strip()
        if not name or not path:
            continue
        parent_path = item.get("parent_path")
        parent_path = str(parent_path).strip() if parent_path else None
        level = int(item.get("level") or max(1, path.count(" / ") + 1))
        node_refs = _refs_from_chunk_ids(
            chunk_by_id,
            item.get("node_evidence_chunk_ids") or item.get("evidence_chunk_ids") or [],
            f"llm_node:{name}",
            0.88,
        )
        description_refs = _refs_from_chunk_ids(
            chunk_by_id,
            item.get("description_evidence_chunk_ids") or [],
            f"llm_description:{name}",
            0.86,
        )
        grade_refs = _refs_from_chunk_ids(
            chunk_by_id,
            item.get("grade_evidence_chunk_ids") or [],
            f"llm_grade_assignment:{name}",
            0.88,
        )
        needs_review = _bool(
            item.get("needs_review"),
            not bool(node_refs) or not bool(description_refs),
        )
        node = TreeNode(
            node_id=_stable_id("node", path),
            name=name,
            path=path,
            level=level,
            parent_id=None,
            description=str(item.get("description") or ""),
            description_evidence_refs=description_refs,
            grade=str(item.get("grade")).strip() if item.get("grade") else None,
            grade_evidence_refs=grade_refs,
            grade_reason=str(item.get("grade_reason") or ""),
            rules=[],
            confidence=_confidence(item.get("confidence"), 0.65),
            needs_review=needs_review,
            status=str(item.get("status") or ("proposed" if needs_review else "evidence_supported")),
            evidence_refs=node_refs,
            description_evidence_level="B" if description_refs else "D",
        )
        for rule_item in item.get("rules") or []:
            if not isinstance(rule_item, dict):
                continue
            conditions = _string_list(rule_item.get("conditions"))
            negatives = _string_list(rule_item.get("negative_conditions"))
            rule_refs = _refs_from_chunk_ids(
                chunk_by_id,
                rule_item.get("evidence_chunk_ids") or [],
                f"llm_rule:{name}",
                0.84,
            )
            rule_review = _bool(rule_item.get("needs_review"), not bool(rule_refs))
            node.rules.append(
                MatchingRule(
                    rule_id=_stable_id("rule", f"{path}:{'|'.join(conditions)}:{'|'.join(negatives)}"),
                    target_node_id=node.node_id,
                    rule_type=str(rule_item.get("rule_type") or "keyword_rule"),
                    conditions=conditions,
                    negative_conditions=negatives,
                    evidence_refs=rule_refs,
                    confidence=_confidence(rule_item.get("confidence"), 0.6),
                    needs_review=rule_review,
                    status="proposed" if rule_review else "evidence_supported",
                )
            )
            if rule_review:
                node.needs_review = True
        node_by_path[path] = node
        pending_parent_paths[path] = parent_path

    for path, node in node_by_path.items():
        parent_path = pending_parent_paths.get(path)
        if parent_path and parent_path in node_by_path:
            node.parent_id = node_by_path[parent_path].node_id
        elif " / " in path:
            inferred_parent_path = path.rsplit(" / ", 1)[0]
            if inferred_parent_path in node_by_path:
                node.parent_id = node_by_path[inferred_parent_path].node_id
    return list(node_by_path.values())


def generate_grounded_candidates_with_llm(state: AgentState, llm_client: Any) -> AgentState:
    prompt_messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_prompt(state.chunks)},
    ]
    response = llm_client.chat(prompt_messages)
    data = _extract_json_object(response.content)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in state.chunks}

    state.candidate_concepts = _to_concepts(data, chunk_by_id)
    state.classification_dimensions, state.selected_dimension = _to_dimensions(data, chunk_by_id)
    state.grade_scheme = _to_grades(data, chunk_by_id)
    state.nodes = _to_nodes(data, chunk_by_id)
    state.llm_used = True
    state.llm_error = ""
    return state
