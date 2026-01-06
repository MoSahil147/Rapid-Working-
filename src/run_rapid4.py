"""
run_rapid4.py
-------------
RapidOCR-only (4 engines), detect all drawn boxes and assign words into those boxes.
Outside-of-box words are merged into smaller flow blocks.

Output per PDF (single JSON):
  - <stem>_blocks.json
    pages[].blocks[] = {text, bbox, in_box}

Optional: annotated PDF with only red rectangles for every block.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from pdf_utils import pdf_to_png_bytes
from ocr_rapid import init_rapidocr_once, decode_png_bytes_to_rgb, extract_words_from_rgb
from box_detector import detect_visual_boxes_from_png
from blocks import words_to_flow_blocks
from annotate import annotate_pages_to_pdf_from_bytes


def _engine_initializer():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    init_rapidocr_once()

def _ocr_page_task(page_idx: int, png_bytes: bytes, min_conf: float | None) -> Tuple[int, List[Dict[str, Any]]]:
    rgb = decode_png_bytes_to_rgb(png_bytes)
    words = extract_words_from_rgb(rgb)
    if min_conf is not None:
        words = [w for w in words if (w.get("confidence") is None or w["confidence"] >= min_conf)]
    return page_idx, words


def _point_in_box(px: float, py: float, box: List[float]) -> bool:
    x0,y0,x1,y1 = box
    return (x0 <= px <= x1) and (y0 <= py <= y1)

def _concat_words(words: List[Dict[str, Any]]) -> str:
    ws = sorted(words, key=lambda w: ((w["bbox"][1]+w["bbox"][3])/2.0, w["bbox"][0]))
    return " ".join(" ".join(w["text"].split()) for w in ws if w.get("text"))

def _assign_words_to_boxes(words: List[Dict[str, Any]], boxes: List[List[float]]):
    per_box: List[List[Dict[str, Any]]] = [[] for _ in boxes]
    leftovers: List[Dict[str, Any]] = []

    for w in words:
        x0,y0,x1,y1 = w["bbox"]
        cx, cy = ((x0+x1)/2.0, (y0+y1)/2.0)
        assigned = False
        for bi, box in enumerate(boxes):
            if _point_in_box(cx, cy, box):
                per_box[bi].append(w)
                assigned = True
                break
        if not assigned:
            leftovers.append(w)
    return per_box, leftovers


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    min_conf: float | None = None,
    annotate_pdf: bool = False,
) -> float:
    t0 = time.time()
    stem = pdf_path.stem
    out_root = output_dir / stem
    out_root.mkdir(parents=True, exist_ok=True)

    pages = pdf_to_png_bytes(pdf_path, dpi=dpi)
    num_pages = len(pages)
    if num_pages == 0:
        return 0.0

    TOTAL_ENGINES = 4
    engines = [ProcessPoolExecutor(max_workers=1, initializer=_engine_initializer) for _ in range(TOTAL_ENGINES)]

    futures = {}
    for i, p in enumerate(pages):
        eng_id = i % TOTAL_ENGINES
        fut = engines[eng_id].submit(_ocr_page_task, i, p["png"], min_conf)
        futures[fut] = (eng_id, i)

    results: Dict[int, Dict[str, Any]] = {}
    for fut in tqdm(as_completed(list(futures.keys())), total=len(futures), desc=f"OCR {stem} (4 engines)"):
        _, page_idx = futures[fut]
        page_idx_ret, words = fut.result()
        w, h = pages[page_idx]["width"], pages[page_idx]["height"]
        results[page_idx_ret] = {"texts": words, "width": w, "height": h}

    for ex in engines:
        ex.shutdown(wait=True)

    pages_blocks: List[Dict[str, Any]] = []

    for i in range(num_pages):
        pj = results.get(i, {"texts": [], "width": pages[i]["width"], "height": pages[i]["height"]})
        page_png = pages[i]["png"]

        visual_boxes = detect_visual_boxes_from_png(page_png)
        per_box_words, leftovers = _assign_words_to_boxes(pj["texts"], visual_boxes)

        visual_blocks = []
        for b, wlist in zip(visual_boxes, per_box_words):
            if not wlist:
                continue
            txt = _concat_words(wlist)
            if txt.strip():
                visual_blocks.append({"text": txt.strip(), "bbox": b, "in_box": True})

        flow_blocks = words_to_flow_blocks(leftovers, gap_x=25.0, gap_y=18.0)
        for fb in flow_blocks:
            fb["in_box"] = False

        blocks = visual_blocks + flow_blocks

        pages_blocks.append({
            "page_num": i + 1,
            "width": pj["width"],
            "height": pj["height"],
            "blocks": blocks
        })

    blocks_json = {
        "document": pdf_path.name,
        "dpi": dpi,
        "engine": "rapidocr (4 engines, box-aware)",
        "pages": pages_blocks
    }
    (out_root / f"{stem}_blocks.json").write_text(json.dumps(blocks_json, ensure_ascii=False, indent=2), encoding="utf-8")

    if annotate_pdf:
        page_png_bytes = [p["png"] for p in pages]
        annotate_pages_to_pdf_from_bytes(page_png_bytes, pages_blocks, out_root / f"{stem}_annotated.pdf")

    elapsed = time.time() - t0
    print(f"{stem}: {elapsed:.2f}s | pages={num_pages}")
    return elapsed


def main():
    import argparse
    ap = argparse.ArgumentParser(description="RapidOCR (4 engines) with visual boxes; red-only annotation")
    ap.add_argument("--input", type=str, required=True, help="PDF file or folder")
    ap.add_argument("--output", type=str, required=True, help="Output directory")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI")
    ap.add_argument("--min-conf", type=float, default=None, help="Drop words below this confidence")
    ap.add_argument("--annotate", action="store_true", help="Also write annotated PDF (only red boxes)")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)

    total = 0.0
    if input_path.is_file() and input_path.suffix.lower()==".pdf":
        total += process_pdf(input_path, output_dir, dpi=args.dpi, min_conf=args.min_conf, annotate_pdf=args.annotate)
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {input_path}")
            return
        for pdf in pdfs:
            total += process_pdf(pdf, output_dir, dpi=args.dpi, min_conf=args.min_conf, annotate_pdf=args.annotate)
    else:
        raise ValueError("Input must be a PDF file or a folder containing PDFs")

    print(f"Total: {total:.2f}s")


if __name__ == "__main__":
    main()