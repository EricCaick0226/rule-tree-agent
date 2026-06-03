from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.steps.description_context_kb import (
    build_context_units,
    build_row_query_terms,
    flag_description_quality,
    generate_description_candidates_batched,
    retrieve_contexts,
)
from src.llm.client import OpenAICompatibleLLMClient


def _load_rule_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("classification_rows") if isinstance(data, dict) else None
    return [row for row in rows or [] if isinstance(row, dict)]


def build_description_context_report(
    txt_path: Path,
    rule_table_path: Path,
    limit: int,
    generate: bool = False,
    llm_client: Any | None = None,
    generation_batch_size: int = 20,
) -> dict[str, Any]:
    text = txt_path.read_text(encoding="utf-8")
    rows = _load_rule_rows(rule_table_path)
    units = build_context_units(text, window_lines=3)
    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []

    for row in rows:
        flags = flag_description_quality(row)
        if not flags:
            continue
        query_terms = build_row_query_terms(row)
        report_row = {
            "row_id": row.get("row_id", ""),
            "path": " / ".join(str(level) for level in row.get("path_levels") or []),
            "current_description": row.get("description", ""),
            "description_quality_flags": flags,
            "query_terms": query_terms,
            "retrieved_contexts": retrieve_contexts(units, query_terms, top_k=5),
        }
        priority = (
            1 if row.get("recommended_grade") else 0,
            1 if row.get("data_range_examples") else 0,
            len(row.get("path_levels") or []),
        )
        candidates.append((priority, report_row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    sampled = [
        report_row
        for _priority, report_row in candidates[: max(0, limit)]
    ]
    generation = {"status": "not_requested", "raw_response_excerpt": ""}
    if generate:
        if llm_client is None:
            raise ValueError("llm_client is required when generate=True")
        try:
            generated_candidates, raw_response = generate_description_candidates_batched(
                llm_client,
                sampled,
                batch_size=generation_batch_size,
            )
        except Exception as exc:
            generation = {
                "status": "failed",
                "error": str(exc),
                "candidate_count": 0,
                "batch_size": max(1, generation_batch_size),
                "batch_count": 0,
                "raw_response_excerpt": "",
            }
        else:
            generated_by_row_id = {
                candidate.get("row_id", ""): candidate
                for candidate in generated_candidates
                if candidate.get("row_id")
            }
            for row in sampled:
                row["generated_description"] = generated_by_row_id.get(row.get("row_id", ""), {})
            generation = {
                "status": "success",
                "candidate_count": len(generated_candidates),
                "batch_size": max(1, generation_batch_size),
                "batch_count": (len(sampled) + max(1, generation_batch_size) - 1) // max(1, generation_batch_size),
                "raw_response_excerpt": raw_response[:2000],
            }

    return {
        "txt_path": str(txt_path),
        "rule_table_path": str(rule_table_path),
        "total_row_count": len(rows),
        "context_unit_count": len(units),
        "sampled_row_count": len(sampled),
        "generation": generation,
        "rows": sampled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a read-only description-context POC report for weak classification descriptions."
    )
    parser.add_argument("--txt", required=True, help="Source TXT document path.")
    parser.add_argument("--rule-table", required=True, help="rule_table.json path.")
    parser.add_argument("--out", required=True, help="Output JSON report path.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum weak-description rows to include.")
    parser.add_argument("--generate", action="store_true", help="Call the LLM to generate candidate descriptions.")
    parser.add_argument("--generate-batch-size", type=int, default=20, help="Rows per LLM call when generating.")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-model", default=None, help="LLM model name.")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    llm_client = (
        OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=args.llm_model)
        if args.generate
        else None
    )
    report = build_description_context_report(
        txt_path=Path(args.txt).expanduser().resolve(),
        rule_table_path=Path(args.rule_table).expanduser().resolve(),
        limit=max(0, args.limit),
        generate=args.generate,
        llm_client=llm_client,
        generation_batch_size=max(1, args.generate_batch_size),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} rows={report['sampled_row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
