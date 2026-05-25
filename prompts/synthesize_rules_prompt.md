# Synthesize Rules Prompt

Task: Generate matching rules for existing candidate nodes.

Rules:
- Use only provided nodes and evidence claims.
- Do not invent keywords.
- Do not invent regex patterns.
- Do not use field examples unless they appear in evidence.
- Conditions and negative_conditions must come from evidence claims.
- Generate negative rules only when exclusion evidence exists.
- Return evidence_claim_ids for every rule.
- If evidence is insufficient, output insufficient_evidence and needs_review = true.

Output JSON object only.
