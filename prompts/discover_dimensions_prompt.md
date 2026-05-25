# Discover Dimensions Prompt

You are discovering possible classification dimensions from document evidence.

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
Find explicit classification principles or classification basis statements. If none exist, return insufficient evidence or a weak candidate marked for human review.
