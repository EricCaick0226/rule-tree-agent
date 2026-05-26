from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from ..core.agent_state import DocumentPage, SourceDocument


PDF_MIN_TEXT_CHARS = 30


def parse_pdf_document(path: Path, doc_id: str, enable_ocr: bool) -> SourceDocument:
    pages: list[DocumentPage] = []
    raw_parts: list[str] = []

    for page_number, text, warning in _extract_pdf_text_pages(path):
        page_text = text.strip()
        source_method = "pdf_text"
        needs_review = False
        page_warning = warning

        if _visible_text_length(page_text) < PDF_MIN_TEXT_CHARS:
            needs_review = True
            if enable_ocr:
                page_text = _ocr_pdf_page(path, page_number - 1)
                source_method = "ocr"
                page_warning = (
                    "PDF text layer was empty or too short; OCR was used. "
                    "OCR-derived evidence requires human review."
                )
                if not page_text:
                    page_warning = "OCR produced no usable text for this page."
            else:
                page_warning = (
                    page_warning
                    or "PDF page has little or no extractable text. Re-run with --ocr for scanned pages."
                )

        pages.append(
            DocumentPage(
                page_number=page_number,
                text=page_text,
                source_method=source_method,
                needs_review=needs_review,
                warning=page_warning,
            )
        )
        if page_text:
            raw_parts.append(f"[Page {page_number}]\n{page_text}")

    raw_text = "\n\n".join(raw_parts).strip()
    if not raw_text:
        if enable_ocr:
            raise ValueError(f"PDF has no extractable text after OCR: {path}")
        raise ValueError(f"PDF has no extractable text: {path}. Re-run with --ocr for scanned pages.")

    return SourceDocument(
        doc_id=doc_id,
        doc_name=path.name,
        file_path=str(path),
        raw_text=raw_text,
        pages=pages,
    )


def _visible_text_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _extract_pdf_text_pages(path: Path) -> list[tuple[int, str, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Install project dependencies first.") from exc

    reader = PdfReader(str(path))
    pages: list[tuple[int, str, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        warning = ""
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - depends on malformed PDFs.
            text = ""
            warning = f"pypdf failed to extract text from page {page_index}: {exc}"
        pages.append((page_index, text.strip(), warning))
    return pages


def _ocr_pdf_page(path: Path, zero_based_page_index: int) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("OCR requires PyMuPDF to render PDF pages before Vision OCR.") from exc

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "vision_ocr_pages.swift"
    if not script_path.exists():
        raise RuntimeError(f"Vision OCR helper script not found: {script_path}")

    dpi = int(os.getenv("OCR_DPI", "200"))
    zoom = dpi / 72.0
    timeout = float(os.getenv("OCR_TIMEOUT_SECONDS", "120"))

    with tempfile.TemporaryDirectory(prefix="rule_tree_agent_ocr_") as temp_dir:
        temp_path = Path(temp_dir)
        image_path = temp_path / f"page_{zero_based_page_index + 1}.png"
        output_dir = temp_path / "ocr"

        pdf = fitz.open(str(path))
        try:
            page = pdf.load_page(zero_based_page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pixmap.save(str(image_path))
        finally:
            pdf.close()

        swift_env = os.environ.copy()
        swift_cache = Path(tempfile.gettempdir()) / "rule_tree_agent_swift_module_cache"
        swift_cache.mkdir(parents=True, exist_ok=True)
        swift_env.setdefault("CLANG_MODULE_CACHE_PATH", str(swift_cache))

        result = subprocess.run(
            ["swift", str(script_path), str(image_path), str(output_dir)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=swift_env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Vision OCR failed for "
                f"{path} page {zero_based_page_index + 1}: {result.stderr.strip() or result.stdout.strip()}"
            )

        return _read_ocr_text(output_dir, zero_based_page_index + 1)


def _read_ocr_text(output_dir: Path, page_number: int) -> str:
    json_path = output_dir / f"page_{page_number}.json"
    txt_path = output_dir / f"page_{page_number}.txt"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        lines = data.get("lines") or []
        return "\n".join(
            str(line.get("text") or "").strip()
            for line in lines
            if line.get("text")
        ).strip()
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return ""
