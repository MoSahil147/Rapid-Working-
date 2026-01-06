"""
annotate.py
-----------
Annotated PDF with only red rectangles for final blocks.
We draw a red rectangle for every block on every page.
"""

from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw
import io

def annotate_pages_to_pdf_from_bytes(
    page_png_bytes: List[bytes],
    pages_blocks: List[Dict[str, Any]],
    out_pdf: Path,
):
    images=[]
    for png, pb in zip(page_png_bytes, pages_blocks):
        im=Image.open(io.BytesIO(png)).convert("RGB")
        dr=ImageDraw.Draw(im)

        for b in pb.get("blocks", []):
            x0,y0,x1,y1=b["bbox"]
            dr.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=3)

        images.append(im)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if images:
        first, rest = images[0], images[1:]
        first.save(out_pdf.as_posix(), save_all=True, append_images=rest, resolution=300)