from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline.agent_executor import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the document-grounded row-first rule tree agent demo.")
    parser.add_argument("--docs", nargs="+", required=True, help="Markdown or text documents.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults to https://api.example.com/v1.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name. Defaults to your-model-name.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Legacy option retained for compatibility; row-first MVP supports .txt and .md only.",
    )
    args = parser.parse_args()

    docs = [str(Path(doc).expanduser().resolve()) for doc in args.docs]
    out_dir = str(Path(args.out).expanduser().resolve())
    if args.ocr:
        print("Warning: --ocr is ignored by the row-first txt/md MVP.")
    try:
        state = run_agent(
            user_task="Generate a new document-grounded classification and grading rule tree from documents.",
            input_files=docs,
            output_dir=out_dir,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            enable_ocr=args.ocr,
        )
    except Exception as exc:
        print(f"Agent failed: {exc}")
        raise SystemExit(1) from exc

    review_count = sum(1 for row in state.classification_rows if row.needs_review)
    classification_depth = (
        state.classification_schema.max_depth if state.classification_schema else "insufficient evidence"
    )
    print("")
    print("Run complete.")
    print("Classification rows:", len(state.classification_rows))
    print("Classification depth:", classification_depth)
    print("Grade scheme found:", "yes" if state.grade_scheme else "no")
    print("LLM used:", "yes" if state.llm_used else "no")
    if state.llm_error:
        print("LLM error:", state.llm_error)
    print("Derived tree nodes:", len(state.nodes))
    print("Review-required rows:", review_count)
    print("Output paths:")
    for name, path in state.output_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
