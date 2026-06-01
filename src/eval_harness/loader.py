from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedFile:
    path: Path
    exists: bool
    data: Any = None
    error: str = ""


@dataclass(frozen=True)
class LoadedJsonl:
    path: Path
    exists: bool
    records: list[dict[str, Any]] = field(default_factory=list)
    corrupt_records: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LoadedTextFile:
    path: Path
    text: str


@dataclass(frozen=True)
class EvalInputs:
    output_dir: Path
    files: dict[str, LoadedFile]
    row_checkpoint: LoadedJsonl
    evidence_checkpoint: LoadedJsonl
    block_checkpoint: LoadedJsonl
    debug_files: list[LoadedTextFile]
    trace_files: list[LoadedTextFile]


def load_output_dir(output_dir: str | Path) -> EvalInputs:
    root = Path(output_dir)
    files = {
        "rule_table_json": _read_json_file(root / "rule_table.json"),
        "rule_tree_json": _read_json_file(root / "rule_tree.json"),
        "review_report_md": _read_text_as_file(root / "review_report.md"),
    }

    return EvalInputs(
        output_dir=root,
        files=files,
        row_checkpoint=_read_jsonl_candidates(
            root,
            [
                "checkpoints/classification_row_batches.jsonl",
                "checkpoints/row_checkpoint.jsonl",
                "checkpoints/classification_rows_checkpoint.jsonl",
                "checkpoints/classification_rows.jsonl",
                "classification_row_batches.jsonl",
                "row_checkpoint.jsonl",
            ],
            ["*row*checkpoint*.jsonl", "*classification*row*.jsonl"],
        ),
        evidence_checkpoint=_read_jsonl_candidates(
            root,
            [
                "checkpoints/evidence_claim_batches.jsonl",
                "checkpoints/evidence_checkpoint.jsonl",
                "checkpoints/evidence_claims_checkpoint.jsonl",
                "checkpoints/evidence_claims.jsonl",
                "evidence_claim_batches.jsonl",
                "evidence_checkpoint.jsonl",
            ],
            ["*evidence*checkpoint*.jsonl", "*evidence*claim*.jsonl"],
        ),
        block_checkpoint=_read_jsonl_candidates(
            root,
            [
                "checkpoints/block_signal_batches.jsonl",
                "checkpoints/block_checkpoint.jsonl",
                "checkpoints/document_blocks_checkpoint.jsonl",
                "checkpoints/document_blocks.jsonl",
                "block_signal_batches.jsonl",
                "block_checkpoint.jsonl",
            ],
            ["*block*checkpoint*.jsonl", "*document*block*.jsonl"],
        ),
        debug_files=_read_text_glob(root / "debug", "*.txt"),
        trace_files=_read_text_glob(root / "traces", "*.txt"),
    )


def _read_json_file(path: Path) -> LoadedFile:
    if not path.exists():
        return LoadedFile(path=path, exists=False)
    try:
        return LoadedFile(
            path=path,
            exists=True,
            data=json.loads(path.read_text(encoding="utf-8")),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return LoadedFile(path=path, exists=True, error=str(exc))


def _read_text_as_file(path: Path) -> LoadedFile:
    if not path.exists():
        return LoadedFile(path=path, exists=False)
    try:
        return LoadedFile(
            path=path,
            exists=True,
            data=path.read_text(encoding="utf-8"),
        )
    except (OSError, UnicodeDecodeError) as exc:
        return LoadedFile(path=path, exists=True, error=str(exc))


def _read_jsonl_candidates(
    root: Path,
    relative_paths: list[str],
    glob_patterns: list[str],
) -> LoadedJsonl:
    for relative_path in relative_paths:
        path = root / relative_path
        if path.exists():
            return _read_jsonl_file(path)

    checkpoints_dir = root / "checkpoints"
    if checkpoints_dir.exists():
        for pattern in glob_patterns:
            matches = sorted(checkpoints_dir.glob(pattern))
            if matches:
                return _read_jsonl_file(matches[0])

    return LoadedJsonl(path=root / relative_paths[0], exists=False)


def _read_jsonl_file(path: Path) -> LoadedJsonl:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    corrupt_records = 0

    if not path.exists():
        return LoadedJsonl(path=path, exists=False)

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return LoadedJsonl(path=path, exists=True, errors=[str(exc)])

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            corrupt_records += 1
            errors.append(f"line {line_number}: {exc.msg}")
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            corrupt_records += 1
            errors.append(f"line {line_number}: expected object")

    return LoadedJsonl(
        path=path,
        exists=True,
        records=records,
        corrupt_records=corrupt_records,
        errors=errors,
    )


def _read_text_glob(directory: Path, pattern: str) -> list[LoadedTextFile]:
    if not directory.exists():
        return []

    loaded: list[LoadedTextFile] = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            text = f"READ_ERROR: {exc}"
        loaded.append(LoadedTextFile(path=path, text=text))
    return loaded
