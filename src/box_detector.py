"""
box_detector.py
---------------
Detect visual rectangular boxes drawn on the page.

Key points:
  - Use RETR_TREE to get all contours (not just outermost).
  - Keep 4-point rectangular-ish polygons.
  - Minimal closing to avoid welding neighboring boxes together.
  - Drop a big parent that just contains many children (keep children).
  - NMS to remove near-duplicates.

Returns list of [x0,y0,x1,y1] rectangles in pixel coords.
"""

from typing import List, Tuple
import numpy as np
import cv2


def _rect_from_contour(cnt) -> Tuple[bool, List[float], float, float]:
    if cv2.contourArea(cnt) < 300:
        return False, [], 0.0, 0.0

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return False, [], 0.0, 0.0

    x, y, w, h = cv2.boundingRect(approx)
    if w < 25 or h < 18:
        return False, [], 0.0, 0.0

    area = float(cv2.contourArea(approx))
    rect_area = float(w * h)
    if rect_area <= 0:
        return False, [], 0.0, 0.0

    rectangularity = area / rect_area
    if rectangularity < 0.65:
        return False, [], area, rect_area

    return True, [float(x), float(y), float(x + w), float(y + h)], area, rect_area


def _iou(b1, b2) -> float:
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


def _contains(outer, inner, pad: float = 2.0) -> bool:
    return (inner[0] >= outer[0] - pad and inner[1] >= outer[1] - pad and
            inner[2] <= outer[2] + pad and inner[3] <= outer[3] + pad)


def _nms_rects(rects: List[List[float]], iou_thresh: float = 0.7) -> List[List[float]]:
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r[2]-r[0]) * (r[3]-r[1]), reverse=True)
    keep = []
    for r in rects:
        if all(_iou(r, k) < iou_thresh for k in keep):
            keep.append(r)
    return keep


def detect_visual_boxes_from_png(png_bytes: bytes) -> List[List[float]]:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []

    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,
                               15, 8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        ok, r, area, rect_area = _rect_from_contour(cnt)
        if ok:
            rects.append(r)

    rects = sorted(rects, key=lambda r: (r[2]-r[0]) * (r[3]-r[1]))
    to_drop = set()
    for i in range(len(rects)):
        if i in to_drop:
            continue
        contain_count = 0
        for j in range(len(rects)):
            if i == j or j in to_drop:
                continue
            if _contains(rects[i], rects[j], pad=2.0):
                contain_count += 1
            if contain_count >= 3:
                to_drop.add(i)
                break
    rects = [r for k, r in enumerate(rects) if k not in to_drop]

    rects = _nms_rects(rects, iou_thresh=0.7)
    return rects