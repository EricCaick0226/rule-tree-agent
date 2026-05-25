from __future__ import annotations

import argparse
from pathlib import Path

from .agent_executor import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the document-grounded rule tree agent demo.")
    parser.add_argument("--docs", nargs="+", required=True, help="Markdown or text documents.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    args = parser.parse_args()

    docs = [str(Path(doc).expanduser().resolve()) for doc in args.docs]
    out_dir = str(Path(args.out).expanduser().resolve())
    state = run_agent(
        user_task="Generate a new document-grounded classification and grading rule tree from documents.",
        input_files=docs,
        output_dir=out_dir,
    )

    review_count = sum(1 for node in state.nodes if node.needs_review)
    print("")
    print("Run complete.")
    print(
        "Selected classification dimension:",
        state.selected_dimension.name if state.selected_dimension else "insufficient evidence",
    )
    print("Grade scheme found:", "yes" if state.grade_scheme else "no")
    print("Number of nodes:", len(state.nodes))
    print("Review-required nodes:", review_count)
    print("Output paths:")
    for name, path in state.output_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

