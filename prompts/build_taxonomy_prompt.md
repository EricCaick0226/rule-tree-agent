# Build Taxonomy Prompt

You are building a candidate classification tree from evidence.

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
Build only the hierarchy supported by headings, nested lists, parent-child statements, or other explicit evidence. Do not force a fixed depth.
