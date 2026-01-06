"""
pdf_utils.py
------------
Render PDF pages to in-memory PNG bytes (no disk I/O).
"""

from pathlib import Path
from typing import List, Dict
import io
import fitz  # PyMuPDF
from PIL import Image


def pdf_to_png_bytes(pdf_path: Path, dpi: int = 200) -> List[Dict]:
    """
    Render a PDF into a list of dicts:
      [{"png": bytes, "width": int, "height": int}, ...]
    """
    out: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)  # raw RGB
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            bio = io.BytesIO()
            im.save(bio, format="PNG", optimize=True)
            out.append({"png": bio.getvalue(), "width": pix.width, "height": pix.height})
    return out