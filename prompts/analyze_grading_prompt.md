# Analyze Grading Prompt

Task: Extract grading scheme and assign grades where supported.

Rules:
- Use only provided evidence claims and candidate nodes.
- Do not invent grading levels.
- Do not create default levels.
- Do not assign grades from prior assumptions.
- Extract grade definitions only if documents define them.
- Assign node grades only if explicit mappings or explicit criteria support them.
- Return evidence_claim_ids for every grade definition and assignment.
- If evidence is insufficient, set grade = null and needs_review = true.

Output JSON object only.
