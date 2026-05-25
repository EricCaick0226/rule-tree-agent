# Extract Concepts Prompt

You are extracting candidate concepts from provided document evidence.

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
Extract terms, phrases, headings, list items, and structurally important concepts that appear in the evidence. Keep original wording.
