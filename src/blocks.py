"""
blocks.py
---------
Merge word-level boxes into smaller, readable flow blocks (for content outside visual boxes).
Moderate thresholds so blocks do not become overly large.
"""

from typing import List, Dict, Any

def _y_mid(b): return (b[1] + b[3]) / 2.0
def _h(b): return max(0.0, b[3] - b[1])
def _ov1(a0,a1,b0,b1): return max(0.0, min(a1,b1) - max(a0,b0))
def _merge_box(b1,b2): return [min(b1[0],b2[0]), min(b1[1],b2[1]), max(b1[2],b2[2]), max(b1[3],b2[3])]

def words_to_flow_blocks(
    words: List[Dict[str, Any]],
    gap_x: float = 25.0,
    gap_y: float = 18.0
) -> List[Dict[str, Any]]:
    if not words:
        return []

    ws = sorted(words, key=lambda w: (_y_mid(w["bbox"]), w["bbox"][0]))
    heights = [_h(w["bbox"]) for w in ws if w.get("bbox")]
    line_tol = max(6.0, (sum(heights)/len(heights) if heights else 12.0) * 0.6)

    lines=[]; cur=[]; last=None
    for w in ws:
        y=_y_mid(w["bbox"])
        if last is None or abs(y-last)<=line_tol:
            cur.append(w)
        else:
            lines.append(sorted(cur, key=lambda t: t["bbox"][0])); cur=[w]
        last=y
    if cur: lines.append(sorted(cur, key=lambda t: t["bbox"][0]))

    segs=[]
    for ln in lines:
        if not ln: continue
        s=[ln[0]]
        for w in ln[1:]:
            p=s[-1]
            hgap=w["bbox"][0]-p["bbox"][2]
            vov=_ov1(p["bbox"][1],p["bbox"][3],w["bbox"][1],w["bbox"][3])
            vmin=min(_h(p["bbox"]),_h(w["bbox"]))
            vrat=(vov/vmin) if vmin>0 else 0.0
            if hgap<=gap_x and vrat>=0.5:
                p["text"]=(p["text"]+" "+w["text"]).strip()
                p["bbox"]=_merge_box(p["bbox"],w["bbox"])
            else:
                s.append(w)
        segs.append(s)

    blocks=[]
    for s in segs:
        for w in s:
            blocks.append({"text": w["text"], "bbox": w["bbox"]})
    blocks.sort(key=lambda b: (_y_mid(b["bbox"]), b["bbox"][0]))

    merged=[]
    for b in blocks:
        if not merged: merged.append(b); continue
        p=merged[-1]
        vgap=b["bbox"][1]-p["bbox"][3]
        xov=_ov1(p["bbox"][0],p["bbox"][2],b["bbox"][0],b["bbox"][2])
        minw=min(p["bbox"][2]-p["bbox"][0], b["bbox"][2]-b["bbox"][0])
        xrat=(xov/minw) if minw>0 else 0.0
        if vgap<=gap_y and xrat>=0.35:
            p["text"]=(p["text"]+" "+b["text"]).strip()
            p["bbox"]=_merge_box(p["bbox"],b["bbox"])
        else:
            merged.append(b)

    out=[]
    for b in merged:
        txt=" ".join(b["text"].split()).strip()
        if txt:
            out.append({"text": txt, "bbox": b["bbox"]})
    return out