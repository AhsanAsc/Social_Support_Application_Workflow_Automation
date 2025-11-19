from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pdfplumber


class PDFParseError(Exception):
    pass


def extract_pages_text(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        raise PDFParseError(f"file not found: {pdf_path}")
    try:
        results: list[dict] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                results.append({"page": idx, "text": text})
        return results
    except Exception as e:  # noqa: BLE001
        raise PDFParseError(str(e))


# trivial chunker; improve later


def chunk_pages(pages: Iterable[dict], max_chars: int = 1200) -> list[dict]:
    chunks: list[dict] = []
    for p in pages:
        t = p.get("text", "")
        for i in range(0, len(t), max_chars):
            segment = t[i : i + max_chars]
            if segment.strip():
                chunks.append({"page": p["page"], "text": segment})
    return chunks
