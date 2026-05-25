# Rule Tree Agent

`rule-tree-agent` is a local Python MVP for generating a candidate classification and grading rule tree from source documents.

The project is intentionally simple. It always uses an OpenAI-compatible LLM endpoint for candidate generation. It does not use LangChain, LangGraph, vector databases, RAG frameworks, a frontend, Docker, a database, or a multi-agent architecture.

## Core Principle

Documents → evidence → candidate concepts → possible classification dimensions → candidate tree → optional grading scheme → grounded descriptions → grounded matching rules → validation issues → human review.

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
- Calls the configured OpenAI-compatible LLM to propose grounded JSON candidates.
- Fails clearly if the API key or LLM endpoint is unavailable.
- Extracts candidate concepts from document structure and text.
- Discovers possible classification dimensions from explicit document wording.
- Builds a candidate taxonomy from evidence-backed hierarchy.
- Extracts grading definitions only when the documents define them.
- Assigns grades only when the documents map nodes to grades.
- Generates simple evidence-based keyword or phrase rules.
- Validates grounding strictly.
- Exports JSON, Markdown tree, and human review report.

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

## MVP Limitations

- No vector search.
- LLM JSON quality depends on the configured model and endpoint.
- Requires a valid API key and network access to the configured LLM gateway.
- Markdown and text input only.
- Complex tables, scanned PDFs, and ambiguous policies are not handled in v0.1.

## Future Improvements

- Add stricter LLM output repair and retry loops.
- Add stronger table parsing.
- Add richer evidence scoring.
- Add reviewer feedback loops.
- Add export formats for enterprise review workflows.

Even in future LLM-enabled versions, all business content must remain traceable to document evidence.
