from __future__ import annotations

import hashlib
import re

from ..core.agent_state import AgentState, ValidationIssue


ALLOWED_CLAIM_TYPES = {
    "definition",
    "inclusion",
    "exclusion",
    "hierarchy",
    "classification_principle",
    "grade_definition",
    "grade_mapping",
    "rule_phrase",
    "insufficient_evidence",
}

ALLOWED_SUPPORT_LEVELS = {"explicit", "structural", "inferred", "weak", "ocr"}


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


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _contains_text(container: str, text: str) -> bool:
    if not text:
        return True
    return text in container or _normalize_for_match(text) in _normalize_for_match(container)


def validate_grounding(state: AgentState) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    corpus = _corpus_text(state)
    claim_ids = {claim.claim_id for claim in state.evidence_claims}

    if not state.evidence_claims:
        _add_issue(
            issues,
            "missing_evidence_claims",
            "high",
            "evidence_claims",
            "未抽取到 evidence claims，后续规则树缺少事实基础。",
            "请检查输入文档或 LLM claim 抽取步骤。",
        )

    for claim in state.evidence_claims:
        ref_text = "\n".join(ref.text for ref in claim.evidence_refs)
        if claim.claim_type not in ALLOWED_CLAIM_TYPES:
            _add_issue(
                issues,
                "schema_contract_violation",
                "high",
                claim.claim_id,
                f"Evidence claim 使用了不允许的 claim_type：{claim.claim_type}",
                "删除该 claim，或让 LLM 按允许的 claim_type 重新抽取。",
            )
        if claim.support_level not in ALLOWED_SUPPORT_LEVELS:
            _add_issue(
                issues,
                "schema_contract_violation",
                "high",
                claim.claim_id,
                f"Evidence claim 使用了不允许的 support_level：{claim.support_level}",
                "删除该 claim，或让 LLM 按允许的 support_level 重新抽取。",
            )
        if not claim.evidence_refs:
            _add_issue(
                issues,
                "unsupported_generation",
                "high",
                claim.claim_id,
                "Evidence claim 缺少证据引用。",
                "删除该 claim 或补充 chunk 证据。",
            )
        if any(ref.source_method == "ocr" for ref in claim.evidence_refs):
            pages = sorted(
                {ref.page_number for ref in claim.evidence_refs if ref.page_number is not None}
            )
            page_text = f"页码：{pages}" if pages else "页码未知"
            _add_issue(
                issues,
                "ocr_evidence_requires_review",
                "medium",
                claim.claim_id,
                f"Evidence claim 使用 OCR 证据，可能存在识别误差（{page_text}）。",
                "请对照 PDF 原件人工核验 OCR 证据后再采用该 claim。",
            )
        if claim.subject and not _contains_text(corpus, claim.subject):
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                claim.claim_id,
                f"Claim subject 未出现在输入文档中：{claim.subject}",
                "Claim 主体必须来自文档原文。",
            )
        if not claim.evidence_quote:
            _add_issue(
                issues,
                "weak_trace",
                "medium",
                claim.claim_id,
                "Evidence claim 缺少 evidence_quote 短原文片段。",
                "建议 claim 抽取阶段返回支持该 claim 的短原文片段。",
            )
        elif not _contains_text(ref_text or corpus, claim.evidence_quote):
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                claim.claim_id,
                "Evidence claim 的 evidence_quote 未出现在其引用证据中。",
                "删除该 claim，或改为引用文档中的直接原文片段。",
            )
        if claim.support_level in {"inferred", "weak", "ocr"} and not claim.needs_review:
            _add_issue(
                issues,
                "schema_contract_violation",
                "medium",
                claim.claim_id,
                "弱证据、推断证据或 OCR 证据没有标记 needs_review。",
                "将该 claim 标记为需要人工复核。",
            )
        if claim.needs_review and not claim.review_reason:
            _add_issue(
                issues,
                "weak_trace",
                "low",
                claim.claim_id,
                "Evidence claim 需要人工复核但缺少 review_reason。",
                "补充需要复核的具体原因，例如 OCR、证据弱、结构推断或证据不足。",
            )
        for value in [claim.object, claim.value]:
            if value and len(value) <= 80 and not _contains_text(corpus, value):
                _add_issue(
                    issues,
                    "hardcoded_or_ungrounded_content",
                    "medium",
                    claim.claim_id,
                    f"Claim 内容可能未出现在输入文档中：{value}",
                    "请人工确认该 claim 是否为证据内归纳，或改为直接原文片段。",
                )

    for concept in state.concept_profiles:
        if not concept.evidence_refs:
            _add_issue(
                issues,
                "unsupported_generation",
                "high",
                concept.name,
                "概念画像缺少证据引用。",
                "删除该概念或补充 claim 证据。",
            )
        if concept.name and not _contains_text(corpus, concept.name):
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                concept.name,
                "概念名称未出现在输入文档中。",
                "概念名称必须来自文档原文。",
            )
        for claim_id in concept.related_claim_ids:
            if claim_id not in claim_ids:
                _add_issue(
                    issues,
                    "broken_trace",
                    "high",
                    concept.name,
                    f"概念引用了不存在的 claim：{claim_id}",
                    "修复 related_claim_ids 或删除无效引用。",
                )

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
    elif not state.selected_dimension.evidence_claim_ids:
        _add_issue(
            issues,
            "weak_trace",
            "medium",
            state.selected_dimension.name,
            "分类维度缺少 evidence_claim_ids。",
            "建议维度直接引用支持它的 evidence claims。",
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
        if not node.evidence_claim_ids:
            _add_issue(
                issues,
                "weak_trace",
                "medium",
                node.path,
                "节点缺少 evidence_claim_ids。",
                "建议节点直接引用支持其存在或层级关系的 evidence claims。",
            )
        for claim_id in node.evidence_claim_ids:
            if claim_id not in claim_ids:
                _add_issue(
                    issues,
                    "broken_trace",
                    "high",
                    node.path,
                    f"节点引用了不存在的 claim：{claim_id}",
                    "修复节点 evidence_claim_ids。",
                )

        if not _contains_text(corpus, node.name):
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

        if node.grade is not None and not _contains_text(corpus, node.grade):
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
                if condition and not _contains_text(corpus, condition):
                    _add_issue(
                        issues,
                        "hardcoded_or_ungrounded_content",
                        "high",
                        node.path,
                        f"规则条件“{condition}”未出现在输入文档中。",
                        "只能使用文档中出现过的术语、短语或格式模式。",
                    )
            for claim_id in rule.evidence_claim_ids:
                if claim_id not in claim_ids:
                    _add_issue(
                        issues,
                        "broken_trace",
                        "high",
                        rule.rule_id,
                        f"规则引用了不存在的 claim：{claim_id}",
                        "修复规则 evidence_claim_ids。",
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
        if not grade.evidence_claim_ids:
            _add_issue(
                issues,
                "weak_trace",
                "medium",
                grade.grade_name,
                "等级定义缺少 evidence_claim_ids。",
                "建议等级定义直接引用 grade_definition claim。",
            )
        if not _contains_text(corpus, grade.grade_name):
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                grade.grade_name,
                "等级名称未出现在输入文档中。",
                "不得保留未由文档支持的等级名称。",
            )

    if not state.nodes:
        _add_issue(
            issues,
            "insufficient_evidence",
            "high",
            "taxonomy",
            "未生成候选规则树节点。",
            "请检查文档是否包含分类结构，或检查 LLM 建树步骤输出。",
        )

    return issues
