# Rule Tree Agent

`rule-tree-agent` is a local Python MVP for generating evidence-grounded classification and grading rows from source documents, then deriving a candidate rule tree from those rows.

The project is intentionally simple. It always uses an OpenAI-compatible LLM endpoint for candidate generation. It does not use LangChain, LangGraph, vector databases, RAG frameworks, a frontend, Docker, a database, or a multi-agent architecture.

## Core Principle

Documents → evidence index → evidence claims → block signals → classification rows → grade definitions → normalized rows → row grounding validation → derived tree → human review.

Everything meaningful must come from the input documents:

- Category names must appear in the documents.
- Grade names must appear in the documents.
- Row paths must be supported by headings, tables, lists, or explicit parent-child text.
- Row descriptions must be extracted from evidence or marked insufficient.
- Unsupported or weak items are marked `needs_review = true`.

This agent proposes. Humans review. It does not create final approved enterprise standards.

## What It Does

- Parses local `.md` and `.txt` files in the default row-first MVP.
- Chunks documents using headings, numbered sections, blank lines, and list blocks.
- Builds a local evidence index over source chunks.
- Calls the configured OpenAI-compatible LLM in narrow stages using prompt files under `prompts/`.
- Extracts evidence claims in batches for larger documents.
- Stores evidence claim support level, short source quote, and human-review reason.
- Retries invalid LLM JSON output once with schema error feedback.
- Extracts evidence claims before extracting classification rows.
- Fails clearly if the API key or LLM endpoint is unavailable.
- Classifies document blocks before row extraction.
- Extracts classification rows and grade definitions from source evidence.
- Normalizes classification rows without inventing categories or grades.
- Validates row grounding strictly.
- Projects a derived tree deterministically from row `path_levels`.
- Exports a candidate table, derived tree, human review report, and raw LLM traces.

The row-first MVP supports .txt and .md inputs. PDF/OCR parsing code remains in the repository but is not part of the default row-first pipeline.

## What It Does Not Do

- It does not invent categories.
- It does not invent grading levels.
- It does not invent risk rules.
- It does not assume a business domain.
- It does not use hidden examples or built-in enterprise taxonomies.
- It does not treat generated output as final truth.

## Folder Structure

```text
rule-tree-agent/
├── README.md
├── requirements.txt
├── data/sample_docs/sample_policy.md
├── data/sample_docs/sample_row_policy.md
├── outputs/.gitkeep
├── prompts/
├── scripts/
├── src/
│   ├── agent_demo.py
│   ├── core/
│   ├── io/
│   ├── llm/
│   ├── output/
│   ├── pipeline/
│   ├── steps/
│   └── validation/
└── notes/architecture.md
```

## Agent Workflow

1. Parse txt/md documents
2. Chunk documents with source spans
3. Build evidence index
4. Extract evidence claims with LLM
5. Classify document blocks with LLM
6. Extract classification rows with LLM
7. Extract grade definitions with LLM
8. Normalize classification rows
9. Validate row grounding
10. Project tree from rows
11. Export candidate table, derived tree, review report, and traces

## Run

From the project directory:

```bash
python3 -m src.agent_demo --docs data/sample_docs/sample_row_policy.md --out outputs
```

By default the agent calls:

- Base URL: `https://api.example.com/v1`
- Model: `your-model-name`

Create `.env` from `.env.example` and set `LLM_API_KEY`.

Useful options:

```bash
# Override endpoint or model
python3 -m src.agent_demo \
  --docs data/sample_docs/sample_row_policy.md \
  --out outputs \
  --llm-base-url https://api.example.com/v1 \
  --llm-model your-model-name
```

Generated files:

- `outputs/rule_table.json`
- `outputs/rule_table.md`
- `outputs/rule_tree.json`
- `outputs/rule_tree.md`
- `outputs/review_report.md`
- `outputs/traces/` when LLM raw responses are available

Optional tuning:

```bash
# Evidence-claim batching. CLAIM_BATCH_SIZE is the max chunks per call;
# CLAIM_BATCH_MAX_CHARS is the max source-text budget per call. Checkpoints
# are written under <output_dir>/checkpoints/evidence_claim_batches.jsonl.
CLAIM_BATCH_SIZE=8 CLAIM_BATCH_MAX_CHARS=6000 \
  python3 -m src.agent_demo --docs data/sample_docs/sample_row_policy.md --out outputs
```

For long table-like `.txt`/`.md` inputs, row extraction is segmented and checkpointed:

- `ROW_SEGMENT_MAX_CHARS` controls deterministic table segment size.
- `ROW_BATCH_MAX_CHARS` controls LLM batch payload size.
- `ROW_CHECKPOINT_ENABLED=true` writes row batch checkpoints under `<output_dir>/checkpoints/classification_row_batches.jsonl`.
- `ROW_RESUME=true` resumes completed row batches after interruption.

`rule_table.json` and `rule_tree.json` intentionally store structured state and trace file paths only. Full raw LLM responses are written under `outputs/traces/` for debugging and audit review.

Legacy parser notes, not part of the default row-first MVP:

- PDF/OCR parser code remains in the repository as legacy/non-default infrastructure.
- The default row-first `agent_demo` / `run_agent` path rejects non-`.txt`/`.md` inputs, including `.pdf`.
- The `--ocr` CLI option is retained only for compatibility and is ignored by the row-first txt/md MVP.
- Legacy OCR code uses macOS Vision through `scripts/vision_ocr_pages.swift`; restoring PDF/OCR to the default pipeline would require a separate design.

## MVP Limitations

- No vector search.
- LLM JSON quality depends on the configured model and endpoint, even with one repair retry.
- Requires a valid API key and network access to the configured LLM gateway.
- Default row-first MVP input is Markdown and text only.
- Complex tables, PDF/OCR inputs, low-quality scans, layout reconstruction, and ambiguous policies are not handled by the default row-first MVP.

## Future Improvements

- Add deeper schema validation for nested objects and cross-step contracts.
- Add stronger table parsing.
- Add richer evidence scoring.
- Add reviewer feedback loops.
- Add export formats for enterprise review workflows.

Even in future LLM-enabled versions, all business content must remain traceable to document evidence.
