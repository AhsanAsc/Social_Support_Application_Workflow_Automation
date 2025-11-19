from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image


class ImageOCRError(Exception):
    pass


def ocr_image(img_path: Path, lang: str = "eng") -> list[dict]:
    if not img_path.exists():
        raise ImageOCRError(f"file not found: {img_path}")
    try:
        # Basic load; PIL handles JPEG/PNG
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img, lang=lang) or ""
        # Return a single pseudo-page (page=1) to match PDF shape
        return [{"page": 1, "text": text}]
    except Exception as e:  # noqa: BLE001
        raise ImageOCRError(str(e))
