# Normalize Concepts Prompt

Task: Build concept profiles from evidence claims.

Rules:
- Use only provided evidence claims.
- Do not invent categories.
- Do not invent grading levels.
- Do not invent hierarchy.
- Do not invent descriptions.
- Do not invent rules.
- Concept names, aliases, definitions, included items, and excluded items must be traceable to claim IDs.
- If evidence is insufficient, output needs_review = true.
- Return related_claim_ids for every concept profile.
- Separate classification, grading, rules, and evidence.

Output JSON object only.
