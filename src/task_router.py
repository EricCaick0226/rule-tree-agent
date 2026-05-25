from __future__ import annotations


def detect_task_type(user_task: str) -> str:
    text = (user_task or "").lower()
    if any(term in text for term in ["generate", "生成", "rule tree", "规则树", "分类树"]):
        if any(term in text for term in ["document", "docs", "文档", "材料"]):
            return "generate_rule_tree_from_docs"
    if any(term in text for term in ["validate", "校验", "验证"]):
        return "validate_existing_tree"
    if any(term in text for term in ["export", "导出"]):
        return "export_outputs"
    return "unknown"

