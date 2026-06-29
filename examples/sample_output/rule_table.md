# Candidate Rule Table

Synthetic example output. This file is not generated from a real policy.

| path | grade | description | evidence_quote | needs_review |
| --- | --- | --- | --- | --- |
| Customer Support Records > Public FAQ Records | Level 1 | Approved FAQ text that can be published externally. | "Public FAQ records contain approved question-and-answer text that can be published externally. These records are classified as Level 1." | false |
| Customer Support Records > Internal Support Tickets | Level 2 | Issue descriptions, troubleshooting notes, and assigned support owner information. | "Internal support tickets contain issue descriptions, troubleshooting notes, and assigned support owner information. These records are classified as Level 2." | false |
| Customer Support Records > Sensitive Escalation Records | Level 3 | Payment dispute, account recovery, or security investigation records requiring review before external sharing. | "Sensitive escalation records contain payment dispute details, account recovery evidence, or security investigation notes. These records are classified as Level 3 and require human review before any external sharing." | false |
| Customer Support Records > Unclear Record Type | insufficient_evidence | The source does not provide enough evidence to assign a supported category or grade. | "If a record description does not contain enough evidence to determine the record type or level, mark it for human review instead of inferring a category." | true |
