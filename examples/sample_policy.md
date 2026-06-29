# Synthetic Support Record Classification Policy

This sample is artificial and exists only to demonstrate repository output shape.

## Customer Support Records

Customer support records include service tickets, chat transcripts, and follow-up notes created while resolving a customer issue.

### Public FAQ Records

Public FAQ records contain approved question-and-answer text that can be published externally. These records are classified as Level 1.

### Internal Support Tickets

Internal support tickets contain issue descriptions, troubleshooting notes, and assigned support owner information. These records are classified as Level 2.

### Sensitive Escalation Records

Sensitive escalation records contain payment dispute details, account recovery evidence, or security investigation notes. These records are classified as Level 3 and require human review before any external sharing.

## Review Rule

If a record description does not contain enough evidence to determine the record type or level, mark it for human review instead of inferring a category.
