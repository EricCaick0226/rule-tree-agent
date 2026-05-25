from __future__ import annotations

import hashlib

from .agent_state import AgentState, ValidationIssue


def _issue_id(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"issue_{digest}"


def _add_issue(
    issues: list[ValidationIssue],
    issue_type: str,
    severity: str,
    target: str,
    problem: str,
    suggested_action: str,
) -> None:
    issues.append(
        ValidationIssue(
            issue_id=_issue_id(issue_type, target, problem),
            issue_type=issue_type,
            severity=severity,
            target=target,
            problem=problem,
            suggested_action=suggested_action,
            status="open",
        )
    )


def _corpus_text(state: AgentState) -> str:
    return "\n".join(doc.raw_text for doc in state.documents)


def validate_grounding(state: AgentState) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    corpus = _corpus_text(state)

    if state.selected_dimension is None:
        _add_issue(
            issues,
            "missing_classification_dimension",
            "high",
            "classification_dimension",
            "当前文档未提供可靠的分类维度。",
            "请人工补充分类依据，或提供包含分类原则的文档。",
        )
    elif not state.selected_dimension.evidence_refs:
        _add_issue(
            issues,
            "unsupported_generation",
            "high",
            state.selected_dimension.name,
            "分类维度缺少证据引用。",
            "回到原文确认分类依据，并补充证据。",
        )

    if not state.grade_scheme:
        _add_issue(
            issues,
            "missing_grade_scheme",
            "medium",
            "grade_scheme",
            "当前文档未定义可用分级方案。",
            "若需要分级，请提供包含等级名称、定义或标准的文档。",
        )

    paths: set[str] = set()
    for node in state.nodes:
        if node.path in paths:
            _add_issue(
                issues,
                "duplicated_path",
                "medium",
                node.path,
                "发现重复节点路径。",
                "请人工确认重复节点是否应合并或重命名。",
            )
        paths.add(node.path)

        if not node.evidence_refs:
            _add_issue(
                issues,
                "unsupported_generation",
                "high",
                node.path,
                "节点缺少创建依据。",
                "删除该节点或补充原文证据。",
            )

        if node.name not in corpus:
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                node.path,
                "节点名称未出现在输入文档中。",
                "不得保留未由文档支持的节点名称。",
            )

        if not node.description_evidence_refs and node.description_evidence_level != "D":
            _add_issue(
                issues,
                "unsupported_generation",
                "high",
                node.path,
                "节点描述缺少证据引用。",
                "将描述改为证据不足，或补充引用。",
            )

        if node.description_evidence_level in {"C", "D"}:
            _add_issue(
                issues,
                "needs_human_review",
                "medium",
                node.path,
                "节点描述证据不足或仅有弱上下文。",
                "请人工确认定义、范围和适用边界。",
            )

        if node.grade is not None and not node.grade_evidence_refs:
            _add_issue(
                issues,
                "unsupported_generation",
                "high",
                node.path,
                "节点分级缺少证据引用。",
                "移除分级建议，或补充分级映射证据。",
            )

        if node.grade is not None and node.grade not in corpus:
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                node.path,
                "节点分级名称未出现在输入文档中。",
                "不得保留未由文档支持的分级名称。",
            )

        if not node.rules:
            _add_issue(
                issues,
                "missing_rule",
                "medium",
                node.path,
                "节点未生成匹配规则。",
                "人工补充规则，或提供包含可匹配术语的文档。",
            )

        for rule in node.rules:
            if rule.conditions and not rule.evidence_refs:
                _add_issue(
                    issues,
                    "unsupported_generation",
                    "high",
                    node.path,
                    "匹配规则缺少证据引用。",
                    "移除规则或补充术语来源证据。",
                )
            for condition in [*rule.conditions, *rule.negative_conditions]:
                if condition and condition not in corpus:
                    _add_issue(
                        issues,
                        "hardcoded_or_ungrounded_content",
                        "high",
                        node.path,
                        f"规则条件“{condition}”未出现在输入文档中。",
                        "只能使用文档中出现过的术语、短语或格式模式。",
                    )

        if node.confidence < 0.6:
            _add_issue(
                issues,
                "low_confidence_node",
                "medium",
                node.path,
                "节点置信度较低。",
                "请人工确认是否保留该节点。",
            )

        if node.needs_review:
            _add_issue(
                issues,
                "needs_human_review",
                "low",
                node.path,
                "节点被标记为需要人工复核。",
                "请审核节点名称、层级、描述、分级和规则。",
            )

    for grade in state.grade_scheme:
        if not grade.evidence_refs:
            _add_issue(
                issues,
                "unsupported_generation",
                "high",
                grade.grade_name,
                "等级定义缺少证据引用。",
                "删除该等级或补充原文证据。",
            )
        if grade.grade_name not in corpus:
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                grade.grade_name,
                "等级名称未出现在输入文档中。",
                "不得保留未由文档支持的等级名称。",
            )

    return issues

