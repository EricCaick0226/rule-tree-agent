from __future__ import annotations

import hashlib
import re

from ..core.agent_state import AgentState, ValidationIssue


ALLOWED_ROW_SUPPORT_LEVELS = {"explicit", "structural", "weak"}
ALLOWED_DESCRIPTION_SOURCES = {"quoted", "summarized", "insufficient"}
INSUFFICIENT_DESCRIPTION = "证据不足，无法从当前文档确定"


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
    raw_texts = [doc.raw_text for doc in state.documents]
    page_texts = [page.text for doc in state.documents for page in doc.pages]
    return "\n".join([*raw_texts, *page_texts])


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _contains_text(container: str, text: str) -> bool:
    if not text:
        return True
    return text in container or _normalize_for_match(text) in _normalize_for_match(container)


def _contains_evidence_quote(container: str, quote: str) -> bool:
    return _contains_text(container, quote)


def _quote_parts(quote: str) -> list[str]:
    return [part.strip() for part in (quote or "").split("；") if part.strip()]


def _contains_all_quote_parts(container: str, quote: str) -> bool:
    return all(_contains_evidence_quote(container, part) for part in _quote_parts(quote))


def _quote_in_any_ref(row, quote: str) -> bool:
    if not quote:
        return True
    return any(_contains_evidence_quote(ref.text, quote) for ref in row.evidence_refs)


def _all_quote_parts_in_refs(row, quote: str) -> bool:
    return all(_quote_in_any_ref(row, part) for part in _quote_parts(quote))


def validate_row_grounding(state: AgentState) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    corpus = _corpus_text(state)
    seen_paths: set[str] = set()
    grade_names = {grade.grade_name for grade in state.grade_scheme}

    if not state.classification_rows:
        _add_issue(
            issues,
            "missing_classification_rows",
            "high",
            "classification_rows",
            "未抽取到 classification_rows，无法生成分类分级表。",
            "请检查输入文档是否包含分类分级明细，或检查 row 抽取步骤。",
        )

    for row in state.classification_rows:
        path = " / ".join(row.path_levels)
        target = path or row.row_id
        ref_text = "\n".join(ref.text for ref in row.evidence_refs)
        evidence_text = ref_text or corpus

        if not row.path_levels:
            _add_issue(
                issues,
                "schema_contract_violation",
                "high",
                target,
                "分类行缺少 path_levels。",
                "删除该行或重新抽取。",
            )

        if path in seen_paths:
            _add_issue(
                issues,
                "duplicated_path",
                "medium",
                target,
                "发现重复分类路径。",
                "合并重复行并保留证据更强的说明和分级。",
            )
        seen_paths.add(path)

        for level in row.path_levels:
            if not _contains_text(corpus, level):
                _add_issue(
                    issues,
                    "hardcoded_or_ungrounded_content",
                    "high",
                    target,
                    f"分类层级未出现在输入文档中：{level}",
                    "删除该层级或改为文档原文。",
                )

        if not row.evidence_refs:
            _add_issue(
                issues,
                "unsupported_generation",
                "high",
                target,
                "分类行缺少证据引用。",
                "补充 chunk 证据或删除该行。",
            )

        if not row.evidence_quote:
            _add_issue(
                issues,
                "weak_trace",
                "medium",
                target,
                "分类行缺少 evidence_quote。",
                "重新抽取并返回支持该行的短原文。",
            )
        elif not _contains_all_quote_parts(evidence_text, row.evidence_quote):
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                target,
                "分类行 evidence_quote 未出现在引用证据或原文中。",
                "删除该行或改为引用原文片段。",
            )

        if row.grade_evidence_quote and not _all_quote_parts_in_refs(
            row,
            row.grade_evidence_quote,
        ):
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                target,
                "grade_evidence_quote 未出现在引用证据或原文中。",
                "改为引用文档原文。",
            )

        if row.support_level not in ALLOWED_ROW_SUPPORT_LEVELS:
            _add_issue(
                issues,
                "schema_contract_violation",
                "high",
                target,
                f"不允许的 support_level：{row.support_level}",
                "使用 explicit、structural 或 weak。",
            )

        if row.description_source not in ALLOWED_DESCRIPTION_SOURCES:
            _add_issue(
                issues,
                "schema_contract_violation",
                "high",
                target,
                f"不允许的 description_source：{row.description_source}",
                "使用 quoted、summarized 或 insufficient。",
            )

        if row.description_source == "insufficient" and row.description != INSUFFICIENT_DESCRIPTION:
            _add_issue(
                issues,
                "schema_contract_violation",
                "medium",
                target,
                "证据不足说明未使用统一文案。",
                f"将分类说明改为：{INSUFFICIENT_DESCRIPTION}",
            )

        description_is_covered = row.description and _contains_evidence_quote(
            row.evidence_quote,
            row.description,
        )
        if (
            row.description_source != "insufficient"
            and not row.description_evidence_quote
            and not description_is_covered
        ):
            _add_issue(
                issues,
                "weak_trace",
                "medium",
                target,
                "分类说明缺少 description_evidence_quote。",
                "补充支持分类说明的原文片段。",
            )

        if row.description_evidence_quote and not _contains_all_quote_parts(
            evidence_text,
            row.description_evidence_quote,
        ):
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "high",
                target,
                "description_evidence_quote 未出现在引用证据或原文中。",
                "改为引用文档原文。",
            )

        if (
            row.recommended_grade
            and not _contains_text(row.evidence_quote, row.recommended_grade)
            and not _contains_text(row.grade_evidence_quote, row.recommended_grade)
        ):
            legality_note = "；该等级名称存在于分级定义中" if row.recommended_grade in grade_names else ""
            _add_issue(
                issues,
                "hardcoded_or_ungrounded_content",
                "medium",
                target,
                f"推荐分级未出现在该行证据中：{row.recommended_grade}{legality_note}",
                "人工确认该行是否应绑定该推荐分级，或补充包含该分级映射的行级证据。",
            )

        if row.support_level in {"structural", "weak"} and not row.needs_review:
            _add_issue(
                issues,
                "schema_contract_violation",
                "medium",
                target,
                "结构推断或弱证据行没有标记 needs_review。",
                "将该行标记为需要人工复核。",
            )

    return issues
