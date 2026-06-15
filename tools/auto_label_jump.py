#!/usr/bin/env python3
"""Pseudo-label real Jump screenshots: piece (reliable) + best-effort platforms.

Writes Labelme JSON next to each image (so you can open labelme and just
fix/add boxes instead of labeling from scratch) plus an annotated preview.

The piece is found by the same color heuristic adb_jump_ppo uses (very
reliable). Platforms are segmented by foreground extraction: a pixel is
platform-ish if it is clearly *brighter* than the per-row background or clearly
*saturated*. Cast shadows are darker and desaturated, so this rule drops them —
that was the failure mode of naive background subtraction.

Run --eval to score boxes against existing hand-labeled JSON (IoU/recall) so you
know how much manual review is left.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb_jump_ppo import find_piece  # noqa: E402


def foreground_mask(arr: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    f = arr.astype(np.float32)
    bright = f.max(axis=2)
    sat = f.max(axis=2) - f.min(axis=2)
    # robust per-row background brightness (the smooth gradient)
    bg_bright = np.median(bright, axis=1, keepdims=True)
    mask = ((bright > bg_bright + 16) | (sat > 42))
    mask[: int(h * 0.12), :] = False   # top UI / score
    mask[int(h * 0.90) :, :] = False   # bottom share button
    return mask


def components(mask: np.ndarray, min_pixels: int) -> list[np.ndarray]:
    h, w = mask.shape
    vis = np.zeros_like(mask, bool)
    ys, xs = np.where(mask)
    comps = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if vis[y, x]:
            continue
        q = deque([(y, x)])
        vis[y, x] = True
        pts = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not vis[ny, nx]:
                    vis[ny, nx] = True
                    q.append((ny, nx))
        if len(pts) >= min_pixels:
            comps.append(np.array(pts))
    return comps


def detect_platforms(arr: np.ndarray, piece_bbox) -> list[tuple[int, int, int, int]]:
    h, w, _ = arr.shape
    mask = foreground_mask(arr)
    if piece_bbox is not None:
        x0, y0, x1, y1 = piece_bbox
        mask[max(0, y0 - 6) : y1 + 6, max(0, x0 - 6) : x1 + 6] = False
    boxes = []
    for c in components(mask, min_pixels=2500):
        cy0, cy1 = int(c[:, 0].min()), int(c[:, 0].max())
        x0, x1 = int(c[:, 1].min()), int(c[:, 1].max())
        bw, bh = x1 - x0, cy1 - cy0
        if bw < 80 or bh < 60:
            continue
        if bw > w * 0.96 and bh > h * 0.6:   # background leak
            continue
        fill = len(c) / max(1, bw * bh)
        if fill < 0.18:                       # too sparse to be a solid block
            continue
        # The blob spans the whole cuboid (top diamond + extruded sides), but
        # the landing label wants only the top diamond. The diamond's widest
        # row is its left/right vertices (mid-line); the side faces drop
        # straight down from there, so above mid-line is pure top face.
        ys = c[:, 0]
        widths = np.bincount(ys - cy0, minlength=bh + 1)
        y_mid = cy0 + int(np.argmax(widths))
        top_h = y_mid - cy0
        y_bottom = min(cy1, y_mid + top_h)    # mirror the top half for the diamond
        if y_bottom - cy0 < 30:
            continue
        boxes.append((x0, cy0, x1, int(y_bottom)))
    return boxes


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def labelme_json(img_path: Path, w: int, h: int, piece, platforms, embed: bool) -> dict:
    shapes = []
    if piece is not None:
        x0, y0, x1, y1 = piece.bbox
        shapes.append({"label": "piece", "points": [[float(x0), float(y0)], [float(x1), float(y1)]],
                       "group_id": None, "shape_type": "rectangle", "flags": {}})
    for (x0, y0, x1, y1) in platforms:
        shapes.append({"label": "landing", "points": [[float(x0), float(y0)], [float(x1), float(y1)]],
                       "group_id": None, "shape_type": "rectangle", "flags": {}})
    image_data = None
    if embed:
        image_data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return {"version": "5.5.0", "flags": {}, "shapes": shapes,
            "imagePath": img_path.name, "imageData": image_data,
            "imageHeight": h, "imageWidth": w}


def gather_targets(src: Path, only_unlabeled: bool) -> list[Path]:
    pngs = sorted(src.glob("*.png"))
    if only_unlabeled:
        return [p for p in pngs if not (src / f"{p.stem}.json").exists()]
    return pngs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="datasets/jump_labelme/raw")
    p.add_argument("--preview-dir", default="datasets/jump_labelme/preview")
    p.add_argument("--all", action="store_true", help="process all images, not just unlabeled ones")
    p.add_argument("--no-embed", action="store_true", help="don't embed imageData in the JSON")
    p.add_argument("--eval", action="store_true",
                   help="score detector against existing JSON instead of writing labels")
    args = p.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src

    if args.eval:
        evaluate(src)
        return

    preview_dir = Path(args.preview_dir)
    if not preview_dir.is_absolute():
        preview_dir = ROOT / preview_dir
    preview_dir.mkdir(parents=True, exist_ok=True)

    targets = gather_targets(src, only_unlabeled=not args.all)
    print(f"labeling {len(targets)} images from {src}")
    for img_path in targets:
        arr = np.array(Image.open(img_path).convert("RGB"))
        h, w, _ = arr.shape
        piece = find_piece(arr)
        platforms = detect_platforms(arr, piece.bbox if piece else None)
        data = labelme_json(img_path, w, h, piece, platforms, embed=not args.no_embed)
        (src / f"{img_path.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        im = Image.open(img_path).convert("RGB")
        d = ImageDraw.Draw(im)
        for (x0, y0, x1, y1) in platforms:
            d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=6)
        if piece is not None:
            d.rectangle(list(piece.bbox), outline=(0, 200, 255), width=6)
        im.save(preview_dir / f"{img_path.stem}.png")
        print(f"  {img_path.name}: piece={'y' if piece else 'N'} platforms={len(platforms)}")
    print(f"previews -> {preview_dir}\nReview/fix in labelme, then run tools/labelme_to_yolo.py")


def evaluate(src: Path) -> None:
    jsons = sorted(src.glob("*.json"))
    piece_hit = piece_tot = 0
    land_tp = land_fp = land_fn = 0
    for jp in jsons:
        d = json.loads(jp.read_text(encoding="utf-8"))
        gt_piece = [s for s in d["shapes"] if s["label"] == "piece"]
        gt_land = [tuple(map(int, (s["points"][0][0], s["points"][0][1],
                                   s["points"][1][0], s["points"][1][1])))
                   for s in d["shapes"] if s["label"] == "landing"]
        gt_land = [(min(a, c), min(b, e), max(a, c), max(b, e)) for a, b, c, e in gt_land]
        img = src / (d.get("imagePath") or f"{jp.stem}.png")
        arr = np.array(Image.open(img).convert("RGB"))
        piece = find_piece(arr)
        if gt_piece:
            piece_tot += 1
            gp = gt_piece[0]["points"]
            gpb = (min(gp[0][0], gp[1][0]), min(gp[0][1], gp[1][1]),
                   max(gp[0][0], gp[1][0]), max(gp[0][1], gp[1][1]))
            if piece and iou(piece.bbox, gpb) > 0.5:
                piece_hit += 1
        det = detect_platforms(arr, piece.bbox if piece else None)
        matched = set()
        for db in det:
            best, bi = 0.0, -1
            for i, gb in enumerate(gt_land):
                if i in matched:
                    continue
                v = iou(db, gb)
                if v > best:
                    best, bi = v, i
            if best > 0.5:
                land_tp += 1
                matched.add(bi)
            else:
                land_fp += 1
        land_fn += len(gt_land) - len(matched)
    print(f"piece: {piece_hit}/{piece_tot} found @IoU>0.5")
    prec = land_tp / max(1, land_tp + land_fp)
    rec = land_tp / max(1, land_tp + land_fn)
    print(f"landing @IoU>0.5: precision={prec:.2f} recall={rec:.2f} "
          f"(tp={land_tp} fp={land_fp} fn={land_fn})")


if __name__ == "__main__":
    main()
