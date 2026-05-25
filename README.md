# Rule Tree Agent

`rule-tree-agent` is a local Python MVP for generating a candidate classification and grading rule tree from source documents.

The project is intentionally simple. It always uses an OpenAI-compatible LLM endpoint for candidate generation. It does not use LangChain, LangGraph, vector databases, RAG frameworks, a frontend, Docker, a database, or a multi-agent architecture.

## Core Principle

Documents → evidence index → evidence claims → concept profiles → possible classification dimensions → candidate taxonomy → grounded descriptions → optional grading scheme → grounded matching rules → validation issues → human review.

Everything meaningful must come from the input documents:

- Category names must appear in the documents.
- Grade names must appear in the documents.
- Hierarchy must be supported by headings, lists, or explicit parent-child text.
- Node descriptions must be extracted from evidence or marked insufficient.
- Matching rules must use terms, phrases, examples, or exclusions found in evidence.
- Unsupported or weak items are marked `needs_review = true`.

This agent proposes. Humans review. It does not create final approved enterprise standards.

## What It Does

- Parses local `.md` and `.txt` files.
- Chunks documents using headings, numbered sections, blank lines, and list blocks.
- Builds a local evidence index over source chunks.
- Calls the configured OpenAI-compatible LLM in narrow stages using prompt files under `prompts/`.
- Extracts evidence claims in batches for larger documents.
- Retries invalid LLM JSON output once with schema error feedback.
- Extracts evidence claims before building any rule tree.
- Normalizes concepts into concept profiles.
- Fails clearly if the API key or LLM endpoint is unavailable.
- Discovers possible classification dimensions from evidence claims.
- Builds a candidate taxonomy from evidence-backed hierarchy claims.
- Describes nodes only from supporting evidence.
- Extracts grading definitions only when evidence claims support them.
- Assigns grades only when documents map nodes to grades or criteria are explicitly supported.
- Generates evidence-based keyword, phrase, context, or negative rules.
- Validates grounding strictly.
- Exports JSON, Markdown tree, human review report, and raw LLM traces.

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
├── outputs/.gitkeep
├── prompts/
├── src/
└── notes/architecture.md
```

## Agent Workflow

1. Parse documents
2. Chunk documents
3. Build evidence index
4. Extract evidence claims with LLM
5. Normalize concept profiles with LLM
6. Discover classification dimensions with LLM
7. Synthesize candidate taxonomy with LLM
8. Describe nodes with LLM
9. Analyze grading with LLM
10. Synthesize matching rules with LLM
11. Validate grounding
12. Export human review package

## Run

From the project directory:

```bash
python3 -m src.agent_demo --docs data/sample_docs/sample_policy.md --out outputs
```

By default the agent calls:

- Base URL: `https://api.example.com/v1`
- Model: `your-model-name`

Create `.env` from `.env.example` and set `LLM_API_KEY`.

Useful options:

```bash
# Override endpoint or model
python3 -m src.agent_demo \
  --docs data/sample_docs/sample_policy.md \
  --out outputs \
  --llm-base-url https://api.example.com/v1 \
  --llm-model your-model-name
```

Generated files:

- `outputs/rule_tree.json`
- `outputs/rule_tree.md`
- `outputs/review_report.md`
- `outputs/traces/` when LLM raw responses are available

Optional tuning:

```bash
# Number of document chunks per evidence-claim LLM call.
CLAIM_BATCH_SIZE=8 python3 -m src.agent_demo --docs data/sample_docs/sample_policy.md --out outputs
```

`rule_tree.json` intentionally stores structured state and trace file paths only. Full raw LLM responses are written under `outputs/traces/` for debugging and audit review.

## MVP Limitations

- No vector search.
- LLM JSON quality depends on the configured model and endpoint, even with one repair retry.
- Requires a valid API key and network access to the configured LLM gateway.
- Markdown and text input only.
- Complex tables, scanned PDFs, and ambiguous policies are not handled in v0.1.

## Future Improvements

- Add deeper schema validation for nested objects and cross-step contracts.
- Add stronger table parsing.
- Add richer evidence scoring.
- Add reviewer feedback loops.
- Add export formats for enterprise review workflows.

Even in future LLM-enabled versions, all business content must remain traceable to document evidence.
