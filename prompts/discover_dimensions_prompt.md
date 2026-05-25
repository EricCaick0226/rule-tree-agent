# Discover Dimensions Prompt

Task: Discover classification dimensions from evidence claims and concept profiles.

Rules:
- Use only provided evidence claims and concept profiles.
- Do not invent categories.
- Do not invent grading levels.
- Do not invent hierarchy.
- Do not invent descriptions.
- Do not invent rules.
- Prefer explicit classification principle claims.
- If no reliable dimension exists, set selected_dimension_name = null.
- Every dimension must return evidence_claim_ids.
- If evidence is weak, output needs_review = true.
- Separate classification, grading, rules, and evidence.

Output JSON object only.
