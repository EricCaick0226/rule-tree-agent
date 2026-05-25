# Generate Description Prompt

You are generating node descriptions from document evidence.

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
For each node, synthesize a short description only from direct definitions, included items, or context in the evidence. If unsupported, state that the current documents are insufficient.
