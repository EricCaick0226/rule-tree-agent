# Extract Grade Scheme Prompt

You are extracting a grading scheme from document evidence.

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
Extract grade names, definitions, and criteria only when they are explicitly present in the evidence. If no scheme is present, return an empty scheme and insufficient evidence.
