"""
ocr_rapid.py
------------
RapidOCR wrapper that normalizes outputs across versions/odd formats.

Return per word:
  {"text": str, "bbox": [x0,y0,x1,y1], "confidence": float|None}
"""

from typing import List, Dict, Any, Iterable
import numbers
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR


def _is_number(x) -> bool:
    return isinstance(x, numbers.Number) and not isinstance(x, bool)

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _looks_like_xyxy(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 4 and all(_is_number(a) for a in v)

def _looks_like_flat8(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 8 and all(_is_number(a) for a in v)

def _looks_like_quad_points(v) -> bool:
    return (
        isinstance(v, (list, tuple)) and len(v) == 4 and
        all(isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_number(a) for a in p) for p in v)
    )

def _as_quad_points(v):
    if v is None:
        return None
    if _looks_like_quad_points(v):
        return [[float(p[0]), float(p[1])] for p in v]
    if _looks_like_flat8(v):
        x0,y0,x1,y1,x2,y2,x3,y3 = [float(a) for a in v]
        return [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    if _looks_like_xyxy(v):
        x0,y0,x1,y1 = [float(a) for a in v]
        return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    if isinstance(v, dict):
        cand = v.get("box") or v.get("points") or v.get("bbox")
        return _as_quad_points(cand)
    return None

def _quad_to_xyxy(quad) -> List[float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]

def _pick_text_score_from_iter(seq: Iterable[Any]):
    text, score = None, None
    for el in seq:
        if text is None and isinstance(el, str):
            text = el
        if score is None and (isinstance(el, (int, float)) or (isinstance(el, str) and _to_float(el) is not None)):
            score = _to_float(el)
        if text is not None and score is not None:
            break
    return text, score


_EXTRACTOR = None

def init_rapidocr_once():
    """Called once per process (worker)."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = RapidOCR()

def extract_words_from_rgb(img_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run RapidOCR on an RGB np.uint8 image (H,W,3). Returns a list of words.
    """
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = RapidOCR()

    out: List[Dict[str, Any]] = []
    if img_rgb is None or img_rgb.size == 0:
        return out

    result = _EXTRACTOR(img_rgb)  # some versions return (result, elapsed)
    if isinstance(result, (list, tuple)) and len(result) == 2 and not isinstance(result[0], (str, bytes)):
        result = result[0]
    if not result:
        return out

    for item in result:
        text, score, quad = None, None, None

        if isinstance(item, dict):
            quad = item.get("box") or item.get("points") or item.get("bbox")
            text = item.get("text")
            score = _to_float(item.get("score"))

        elif isinstance(item, (list, tuple)):
            if len(item) == 3 and isinstance(item[0], (str, bytes)):             # [text, score, quad]
                text = item[0].decode("utf-8") if isinstance(item[0], bytes) else str(item[0])
                score = _to_float(item[1]); quad  = item[2]
            elif len(item) == 2 and isinstance(item[1], (list, tuple)) and len(item[1]) == 2:  # [quad, (text, score)]
                quad, (t, s) = item; text = t; score = _to_float(s)
            elif len(item) == 3 and (_looks_like_quad_points(item[0]) or _looks_like_flat8(item[0]) or _looks_like_xyxy(item[0])):  # [quad, text, score]
                quad = item[0]; text = item[1] if isinstance(item[1], (str, bytes)) else None; score = _to_float(item[2])
            elif len(item) == 2 and isinstance(item[0], (str, bytes)) and isinstance(item[1], (list, tuple)) and len(item[1]) == 2:  # [text, (score, quad)]
                text = item[0].decode("utf-8") if isinstance(item[0], bytes) else str(item[0])
                score = _to_float(item[1][0]); quad = item[1][1]
            else:
                cand_box = None
                for el in item:
                    if (_looks_like_quad_points(el) or _looks_like_flat8(el) or _looks_like_xyxy(el) or
                        (isinstance(el, dict) and any(k in el for k in ("box","points","bbox")))):
                        cand_box = el; break
                quad = cand_box
                if quad is not None:
                    remaining = [el for el in item if el is not quad]
                    t, s = _pick_text_score_from_iter(remaining)
                    text = t or text
                    score = s if s is not None else score
                else:
                    continue
        else:
            continue

        quad_pts = _as_quad_points(quad)
        if quad_pts is None:
            continue
        xyxy = _quad_to_xyxy(quad_pts)
        text = "" if text is None else (text.decode("utf-8") if isinstance(text, bytes) else str(text))
        out.append({"text": text, "bbox": [float(v) for v in xyxy], "confidence": score})

    return out


def decode_png_bytes_to_rgb(png_bytes: bytes) -> np.ndarray:
    """
    Decode PNG bytes → RGB np.uint8 (H,W,3) using OpenCV.
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    return rgb