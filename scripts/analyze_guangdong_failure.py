from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CATEGORY_TEXT_STRUCTURE = "Text Structure Damage"
CATEGORY_QUOTE_MATCHING = "Quote Matching Fragility"
CATEGORY_NON_CATALOG = "Non-Catalog Content Entered Tree"
CATEGORY_TABLE_HIERARCHY = "Table Hierarchy Inheritance Risk"
CATEGORY_LLM_OUTPUT_PRESSURE = "LLM Output Pressure"
CATEGORY_OTHER = "Other Review Burden"

CATEGORY_ORDER = [
    CATEGORY_TEXT_STRUCTURE,
    CATEGORY_QUOTE_MATCHING,
    CATEGORY_NON_CATALOG,
    CATEGORY_TABLE_HIERARCHY,
    CATEGORY_LLM_OUTPUT_PRESSURE,
    CATEGORY_OTHER,
]

NON_CATALOG_MARKERS = (
    "影响事项",
    "重要民生保障",
    "重大突发事件",
    "满足重要、核心数据判定条件但未认定的数据",
    "定量判定数据类型",
)

PAGE_NOISE_RE = re.compile(
    r"(^|\n)(?:-\s*\d+\s*-|一级类别\s+二级类别\s+三级类别\s+四级类别\s+数据说明\s+建议级别|表\s*[A-ZＡ-Ｚ].*?\n)"
)


@dataclass
class ClassificationResult:
    primary_category: str
    categories: list[str]
    reason: str
    path: str
    issue_text: str = ""


@dataclass
class FailureExample:
    path: str
    issue_text: str
    reason: str


@dataclass
class FailureAnalysis:
    total_rows: int
    needs_review_rows: int
    validation_issue_count: int
    debug_failure_count: int
    category_counts: Counter[str] = field(default_factory=Counter)
    validation_issue_type_counts: Counter[str] = field(default_factory=Counter)
    support_level_counts: Counter[str] = field(default_factory=Counter)
    description_source_counts: Counter[str] = field(default_factory=Counter)
    top_evidence_refs: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[FailureExample]] = field(default_factory=dict)
    debug_files: list[str] = field(default_factory=list)


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def remove_page_noise(value: str) -> str:
    return PAGE_NOISE_RE.sub("\n", value or "")


def normalized_contains(source_text: str, needle: str) -> bool:
    if not needle:
        return False
    compact_source = compact_text(remove_page_noise(source_text))
    compact_needle = compact_text(needle)
    if compact_needle in compact_source:
        return True
    if len(compact_needle) > 40:
        return False
    source_index = 0
    first_match = -1
    last_match = -1
    for char in compact_needle:
        found_at = compact_source.find(char, source_index)
        if found_at < 0:
            return False
        if first_match < 0:
            first_match = found_at
        last_match = found_at
        source_index = found_at + 1
    return first_match >= 0 and (last_match - first_match) <= max(80, len(compact_needle) * 6)


def path_to_text(path_value: Any) -> str:
    if isinstance(path_value, list):
        return " / ".join(str(item) for item in path_value if str(item).strip())
    return str(path_value or "")


def row_path(row: dict[str, Any]) -> str:
    return path_to_text(row.get("path_levels") or row.get("path") or "")


def issue_path(issue: dict[str, Any]) -> str:
    return path_to_text(issue.get("path") or issue.get("path_levels") or issue.get("target") or "")


def issue_problem(issue: dict[str, Any]) -> str:
    return str(issue.get("problem") or issue.get("type") or issue.get("message") or "")


def strip_leading_code(value: str) -> str:
    return re.sub(r"^\s*(?:\d+(?:\.\d+)*|[A-ZＡ-Ｚ]、?)\s*", "", value or "").strip()


def add_unique(categories: list[str], category: str) -> None:
    if category not in categories:
        categories.append(category)


def has_non_catalog_marker(text: str) -> bool:
    return any(marker in text for marker in NON_CATALOG_MARKERS)


def classify_validation_issue(issue: dict[str, Any], source_text: str) -> ClassificationResult:
    path = issue_path(issue)
    problem = issue_problem(issue)
    combined = f"{path} {problem}"
    categories: list[str] = []
    reason = "validation issue did not match a more specific observable pattern"

    if has_non_catalog_marker(combined):
        add_unique(categories, CATEGORY_NON_CATALOG)
        reason = "path/problem looks like impact criteria or judgment-condition content, not a catalog row"

    missing_level_match = re.search(r"分类层级未出现在输入文档中：(.+)$", problem)
    if missing_level_match:
        missing_level = missing_level_match.group(1).strip()
        missing_label = strip_leading_code(missing_level)
        if normalized_contains(source_text, missing_level) or normalized_contains(source_text, missing_label):
            add_unique(categories, CATEGORY_TEXT_STRUCTURE)
            reason = "missing level appears in source after whitespace/page-noise normalization"

    if "quote 未出现在引用证据或原文中" in problem or "evidence_quote 未出现在" in problem:
        add_unique(categories, CATEGORY_QUOTE_MATCHING)
        reason = "evidence quote mismatch is visible in validation issue"

    if not categories:
        add_unique(categories, CATEGORY_OTHER)

    return ClassificationResult(
        primary_category=categories[0],
        categories=categories,
        reason=reason,
        path=path,
        issue_text=problem,
    )


def classify_row(row: dict[str, Any], source_text: str) -> ClassificationResult:
    path = row_path(row)
    review_reason = str(row.get("review_reason") or "")
    evidence_quote = str(row.get("evidence_quote") or "")
    support_level = str(row.get("support_level") or "")
    description_source = str(row.get("description_source") or "")
    combined = f"{path} {review_reason} {evidence_quote}"
    categories: list[str] = []
    reason = "review row did not match a more specific observable pattern"

    if has_non_catalog_marker(combined):
        add_unique(categories, CATEGORY_NON_CATALOG)
        reason = "row path or evidence looks like judgment-condition content"

    if description_source == "summarized" and evidence_quote:
        add_unique(categories, CATEGORY_QUOTE_MATCHING)
        reason = "summarized description plus evidence quote suggests quote reconstruction risk"

    if evidence_quote and normalized_contains(source_text, evidence_quote):
        if "\n" not in evidence_quote and ("结构" in review_reason or support_level == "structural"):
            add_unique(categories, CATEGORY_TABLE_HIERARCHY)
            if reason.startswith("review row did not"):
                reason = "row is evidence-backed but depends on structural table inheritance"
    elif evidence_quote and normalized_contains(source_text, path):
        add_unique(categories, CATEGORY_QUOTE_MATCHING)
        if reason.startswith("review row did not"):
            reason = "path is present after normalization but full evidence quote is fragile"
    elif any(level and normalized_contains(source_text, level) for level in path.split(" / ")):
        add_unique(categories, CATEGORY_TEXT_STRUCTURE)
        if reason.startswith("review row did not"):
            reason = "some path levels appear only after source normalization"

    if "继承" in review_reason or "表格结构" in review_reason or support_level == "structural":
        add_unique(categories, CATEGORY_TABLE_HIERARCHY)
        if reason.startswith("review row did not"):
            reason = "review reason/support level explicitly depends on table structure"

    if not categories:
        add_unique(categories, CATEGORY_OTHER)

    return ClassificationResult(
        primary_category=categories[0],
        categories=categories,
        reason=reason,
        path=path,
        issue_text=review_reason,
    )


def load_rule_table(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_debug_failures(debug_dir: Path) -> list[Path]:
    if not debug_dir.exists():
        return []
    return sorted(debug_dir.glob("failed_row_batch_*.txt"))


def add_example(analysis: FailureAnalysis, result: ClassificationResult) -> None:
    examples = analysis.examples.setdefault(result.primary_category, [])
    if len(examples) >= 5:
        return
    examples.append(
        FailureExample(
            path=result.path,
            issue_text=result.issue_text,
            reason=result.reason,
        )
    )


def analyze_failure_artifacts(out_dir: Path, source_txt: Path) -> FailureAnalysis:
    rule_table = load_rule_table(out_dir / "rule_table.json")
    source_text = source_txt.read_text(encoding="utf-8")
    rows = rule_table.get("classification_rows") or []
    issues = rule_table.get("validation_issues") or []
    debug_files = collect_debug_failures(out_dir / "debug")

    analysis = FailureAnalysis(
        total_rows=len(rows),
        needs_review_rows=sum(1 for row in rows if row.get("needs_review")),
        validation_issue_count=len(issues),
        debug_failure_count=len(debug_files),
        debug_files=[str(path.relative_to(out_dir)) for path in debug_files],
    )

    for issue in issues:
        problem = issue_problem(issue)
        analysis.validation_issue_type_counts[problem] += 1
        result = classify_validation_issue(issue, source_text)
        analysis.category_counts[result.primary_category] += 1
        add_example(analysis, result)

    for row in rows:
        if not row.get("needs_review"):
            continue
        analysis.support_level_counts[str(row.get("support_level") or "unknown")] += 1
        analysis.description_source_counts[str(row.get("description_source") or "unknown")] += 1
        for ref in row.get("evidence_refs") or []:
            analysis.top_evidence_refs[str(ref.get("chunk_id") or "unknown")] += 1
        result = classify_row(row, source_text)
        analysis.category_counts[result.primary_category] += 1
        add_example(analysis, result)

    if debug_files:
        analysis.category_counts[CATEGORY_LLM_OUTPUT_PRESSURE] += len(debug_files)
        for debug_file in debug_files[:5]:
            add_example(
                analysis,
                ClassificationResult(
                    primary_category=CATEGORY_LLM_OUTPUT_PRESSURE,
                    categories=[CATEGORY_LLM_OUTPUT_PRESSURE],
                    reason="row extraction produced a debug failure file for malformed or invalid JSON",
                    path=debug_file.name,
                    issue_text=debug_file.read_text(encoding="utf-8", errors="replace").splitlines()[0],
                ),
            )

    return analysis


def render_counter(counter: Counter[str], limit: int = 12) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in counter.most_common(limit)]


def render_markdown(analysis: FailureAnalysis) -> str:
    lines: list[str] = [
        "# Guangdong Failure Analysis",
        "",
        "## Summary",
        f"- Classification rows: {analysis.total_rows}",
        f"- Rows needing review: {analysis.needs_review_rows}",
        f"- Validation issues: {analysis.validation_issue_count}",
        f"- Debug failure files: {analysis.debug_failure_count}",
        "",
        "## Category Counts",
    ]
    for category in CATEGORY_ORDER:
        lines.append(f"- {category}: {analysis.category_counts.get(category, 0)}")

    lines.extend(["", "## Review Row Signals", "### Support Levels"])
    lines.extend(render_counter(analysis.support_level_counts))
    lines.extend(["", "### Description Sources"])
    lines.extend(render_counter(analysis.description_source_counts))
    lines.extend(["", "### Top Evidence References"])
    lines.extend(render_counter(analysis.top_evidence_refs))
    lines.extend(["", "## Validation Issue Types"])
    lines.extend(render_counter(analysis.validation_issue_type_counts))

    lines.extend(["", "## Examples"])
    for category in CATEGORY_ORDER:
        examples = analysis.examples.get(category) or []
        lines.extend(["", f"### {category}"])
        if not examples:
            lines.append("- none")
            continue
        for example in examples:
            lines.append(f"- `{example.path}`")
            if example.issue_text:
                lines.append(f"  - issue: {example.issue_text}")
            lines.append(f"  - reason: {example.reason}")

    lines.extend(
        [
            "",
            "## Conclusion",
            "- Treat this report as artifact diagnosis, not a business correctness judgment.",
            "- If Text Structure Damage and Quote Matching Fragility dominate, prioritize table/text reconstruction diagnostics before changing prompts.",
            "- If Non-Catalog Content Entered Tree dominates, tighten catalog-vs-condition separation before projecting rows into a tree.",
            "- Keep high-risk structural rows reviewable instead of auto-fixing them with document-specific rules.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_failure_analysis(out_dir: Path, source_txt: Path) -> Path:
    analysis = analyze_failure_artifacts(out_dir, source_txt)
    output_path = out_dir / "guangdong_failure_analysis.md"
    output_path.write_text(render_markdown(analysis), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a read-only Guangdong failure analysis report.")
    parser.add_argument("--out", required=True, type=Path, help="Existing Guangdong output directory.")
    parser.add_argument("--source-txt", required=True, type=Path, help="Original Guangdong txt source.")
    args = parser.parse_args()

    try:
        output_path = write_failure_analysis(args.out, args.source_txt)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
