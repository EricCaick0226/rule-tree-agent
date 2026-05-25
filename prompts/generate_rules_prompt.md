# Generate Rules Prompt

You are generating matching rules for candidate classification nodes.

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
Use only terms, phrases, aliases, examples, exclusions, or explicit patterns found in evidence. Do not create regular expressions unless the evidence contains explicit formats.
