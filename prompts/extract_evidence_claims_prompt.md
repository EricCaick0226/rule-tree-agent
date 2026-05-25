# Extract Evidence Claims Prompt

Task: Extract evidence claims from document chunks.

Rules:
- Use only provided document chunks.
- Do not build a taxonomy.
- Do not invent categories.
- Do not invent grading levels.
- Do not invent hierarchy.
- Do not invent descriptions.
- Do not invent rules.
- If evidence is insufficient, output needs_review = true.
- Return evidence_chunk_ids for every claim.
- Separate classification, grading, rules, and evidence.

Allowed claim types:
- definition
- inclusion
- exclusion
- hierarchy
- classification_principle
- grade_definition
- grade_mapping
- rule_phrase
- insufficient_evidence

Output JSON object only.
