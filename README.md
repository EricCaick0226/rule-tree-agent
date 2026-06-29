# Rule Tree Agent

`rule-tree-agent` is a local Python agent for turning source policy documents into evidence-grounded candidate classification rows and a derived rule tree.

It is designed for review workflows where every meaningful row should be traceable to source text. The agent proposes candidate structure; humans still approve the final standard.

## Pipeline

```text
Documents
  -> chunks with source spans
  -> evidence claims
  -> document block signals
  -> classification rows
  -> grade definitions
  -> grounding validation
  -> derived rule tree
  -> review artifacts
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` in `.env`, then run:

```bash
python3 -m src.agent_demo --docs data/sample_docs/sample_row_policy.md --out outputs
```

## Example Output

For a public-safe synthetic example, see:

- `examples/sample_policy.md`
- `examples/sample_output/rule_table.md`
- `examples/sample_output/rule_tree.md`
- `examples/sample_output/run_quality.json`

## What It Does

- Parses local `.txt` and `.md` files in the default row-first path.
- Chunks source documents with source-span metadata.
- Extracts evidence claims before proposing classification rows.
- Classifies document blocks into source-backed signals.
- Extracts candidate classification rows and grade definitions from document evidence.
- Validates row grounding against source-backed support.
- Projects a derived rule tree from candidate row paths.
- Exports review artifacts for human inspection.

## Evidence Rules

The tool does not invent categories, grading levels, risk rules, or domain assumptions. Meaningful output should be backed by source text, and weak or unsupported items should be treated as review material rather than final truth.

Outputs are candidate artifacts for human review. They are not approved standards, production classifications, or compliance decisions.

## Outputs

A typical run writes reviewable artifacts under the selected output directory:

- `rule_table.json`
- `rule_table.md`
- `rule_tree.json`
- `rule_tree.md`
- `run_quality.json`
- `traces/` when raw LLM responses are available

`rule_table.json` and `rule_tree.json` store structured state and trace references. Raw LLM responses, when exported, are kept separately for audit and debugging.

## Testing

Run focused tests for the current public demo path:

```bash
python3 -m unittest tests.test_wps_txt_cleaner tests.test_document_parser_lines -v
```

## Scope And Limitations

- Default row-first inputs are `.txt` and `.md`.
- PDF/OCR parser code exists in the repository, but it is non-default legacy infrastructure.
- Non-text document parsing, complex layout reconstruction, low-quality scans, and ambiguous source policies require extra review.
- LLM JSON quality depends on the configured model and endpoint.
- The project does not include vector search, LangChain, LangGraph, a vector database, a frontend, Docker, a database, or a multi-agent architecture.
