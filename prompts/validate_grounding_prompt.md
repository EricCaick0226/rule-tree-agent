# Validate Grounding Prompt

You are validating whether generated classification, grading, descriptions, and rules are grounded in evidence.

Rules:
- Use only provided document evidence.
- Do not invent categories.
- Do not invent grading levels.
- Do not invent hierarchy.
- Do not invent descriptions.
- Do not invent rules.
- If evidence is insufficient, output needs_review = true.
- Return evidence_refs for every generated item.
- Separate classification, grading, rules, and evidence.

Task:
Flag every unsupported node, description, grade, rule, hierarchy relation, and business claim. Require human review for low-confidence or insufficient-evidence items.
