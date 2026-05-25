# Describe Nodes Prompt

Task: Describe existing candidate taxonomy nodes.

Rules:
- Use only provided nodes and evidence claims.
- Do not create new nodes.
- Do not invent descriptions.
- Do not invent risk, sensitivity, business usage, or protection requirements.
- Every description must return description_evidence_claim_ids.
- If evidence is insufficient, use cautious insufficient-evidence wording and needs_review = true.
- Separate descriptions from grading and rules.

Output JSON object only.
