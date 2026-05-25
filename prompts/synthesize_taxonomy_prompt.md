# Synthesize Taxonomy Prompt

Task: Build a candidate taxonomy from evidence claims and concept profiles.

Rules:
- Use only provided evidence claims and concept profiles.
- Do not invent root categories.
- Do not invent hierarchy.
- Do not force a fixed depth.
- Do not generate grading or matching rules in this step.
- Every node and parent-child relationship must return evidence_claim_ids.
- If hierarchy evidence is weak, mark needs_review = true.
- If evidence is insufficient, return an empty node list or insufficient_evidence items.

Output JSON object only.
