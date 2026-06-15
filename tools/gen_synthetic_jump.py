#!/usr/bin/env python3
"""Render synthetic Jump (跳一跳) scenes with perfect, complete YOLO labels.

The real game is isometric: a gradient background, a handful of cuboid
platforms (square top diamond + two side faces, sometimes striped), soft cast
shadows, and a single dark pawn standing on one platform. We draw all of that
from known geometry, so every piece/landing box is exact and *every* platform
is labeled (the thing simple CV can't do on real screenshots).

Output is written straight as a YOLO dataset (images/ + labels/), since the
labels are ground truth. Train a model on this, then finetune on real images.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]

W, H = 1080, 2400
PIECE_ID, LANDING_ID = 0, 1

# Platform top colors sampled to match the real game's flat-shaded palette.
TOP_COLORS = [
    (236, 236, 238),  # white
    (108, 168, 110),  # green
    (210, 96, 92),    # red
    (96, 150, 206),   # blue
    (224, 196, 96),   # yellow
    (206, 130, 176),  # pink
    (150, 132, 200),  # purple
    (110, 196, 198),  # teal
    (236, 168, 104),  # orange
]


def lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def shade(color: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    """Multiply brightness by f (clamped)."""
    return tuple(max(0, min(255, int(round(c * f)))) for c in color)


def draw_gradient_bg(rng: random.Random) -> Image.Image:
    top = (rng.randint(196, 214), rng.randint(200, 216), rng.randint(208, 224))
    bot = lerp(top, (rng.randint(150, 180), rng.randint(156, 186), rng.randint(170, 198)), 1.0)
    img = Image.new("RGB", (W, H), top)
    px = img.load()
    for y in range(H):
        c = lerp(top, bot, y / (H - 1))
        for x in range(0, W, 1):
            px[x, y] = c
    return img


def platform_polys(cx: float, cy: float, rx: float, ry: float, height: float):
    """Return (top, left_face, right_face) polygons for a cuboid platform.

    cx, cy is the center of the top diamond; rx/ry its half-extents.
    """
    top_pt = (cx, cy - ry)
    right_pt = (cx + rx, cy)
    bot_pt = (cx, cy + ry)
    left_pt = (cx - rx, cy)
    top = [top_pt, right_pt, bot_pt, left_pt]
    left_face = [left_pt, bot_pt, (bot_pt[0], bot_pt[1] + height), (left_pt[0], left_pt[1] + height)]
    right_face = [bot_pt, right_pt, (right_pt[0], right_pt[1] + height), (bot_pt[0], bot_pt[1] + height)]
    return top, left_face, right_face


def draw_platform(draw: ImageDraw.ImageDraw, rng: random.Random,
                  cx: float, cy: float, rx: float, height: float) -> tuple[int, int, int, int]:
    ry = rx * rng.uniform(0.46, 0.56)
    color = rng.choice(TOP_COLORS)
    top, left_face, right_face = platform_polys(cx, cy, rx, ry, height)

    # optional horizontal stripe bands on the sides (like the striped blocks)
    bands = rng.choice([1, 1, 1, 2, 3])
    band_colors = [color]
    for _ in range(bands - 1):
        band_colors.append(rng.choice([(236, 236, 238), shade(color, 0.7)]))

    # split each side face into stripe bands top->bottom; right is darker than left
    def draw_face(face, base_f):
        for i in range(bands):
            t0, t1 = i / bands, (i + 1) / bands
            p0 = (face[0][0], face[0][1] + (face[3][1] - face[0][1]) * t0)
            p1 = (face[1][0], face[1][1] + (face[2][1] - face[1][1]) * t0)
            p2 = (face[1][0], face[1][1] + (face[2][1] - face[1][1]) * t1)
            p3 = (face[0][0], face[0][1] + (face[3][1] - face[0][1]) * t1)
            c = shade(band_colors[i % len(band_colors)], base_f)
            draw.polygon([p0, p1, p2, p3], fill=c)

    draw_face(right_face, 0.62)
    draw_face(left_face, 0.82)
    draw.polygon(top, fill=color)

    # thin top edge highlight
    draw.line([top[3], top[0], top[1]], fill=shade(color, 1.12), width=2)

    # landing box = the top diamond surface only (not the extruded sides)
    xs = [p[0] for p in top]
    ys = [p[1] for p in top]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def draw_pawn(draw: ImageDraw.ImageDraw, rng: random.Random,
              cx: float, top_y: float, scale: float) -> tuple[int, int, int, int]:
    """Pawn standing on the platform top at (cx, top_y), pointing up."""
    head_r = 26 * scale
    body_h = 150 * scale
    body_top_w = 30 * scale
    body_bot_w = 56 * scale
    base = top_y - 4
    body_top = base - body_h
    body = [
        (cx - body_top_w, body_top),
        (cx + body_top_w, body_top),
        (cx + body_bot_w, base),
        (cx - body_bot_w, base),
    ]
    dark = (rng.randint(38, 58), rng.randint(34, 52), rng.randint(70, 96))
    draw.polygon(body, fill=dark)
    head_cy = body_top - head_r * 0.5
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                 fill=shade(dark, 1.15))
    # small specular highlight
    draw.ellipse([cx - head_r * 0.4, head_cy - head_r * 0.6,
                  cx - head_r * 0.05, head_cy - head_r * 0.2], fill=(150, 150, 180))
    x0, x1 = cx - body_bot_w, cx + body_bot_w
    y0, y1 = head_cy - head_r, base
    return int(x0), int(y0), int(x1), int(y1)


def gen_scene(rng: random.Random) -> tuple[Image.Image, list[tuple[int, int, int, int, int]]]:
    img = draw_gradient_bg(rng)
    boxes: list[tuple[int, int, int, int, int]] = []

    n = rng.choice([1, 2, 2, 2, 3, 3, 4])
    plats = []
    for _ in range(n):
        for _try in range(40):
            rx = rng.uniform(120, 250)
            height = rng.uniform(55, 150)
            cx = rng.uniform(rx + 30, W - rx - 30)
            cy = rng.uniform(H * 0.34, H * 0.66)
            # keep platforms apart so their boxes don't become near-duplicates
            if all((abs(cx - ox) > (rx + orx) * 0.75 or abs(cy - oy) > 110)
                   for ox, oy, orx, _ in plats):
                plats.append((cx, cy, rx, height))
                break

    # draw far (smaller cy) first for rough depth ordering
    plats.sort(key=lambda p: p[1])

    # all cast shadows on one soft layer, composited under the platforms
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for (cx, cy, rx, height) in plats:
        ry = rx * 0.5
        sx, sy = cx - rx * 0.35, cy + height * 0.85
        sd.ellipse([sx - rx * 1.15, sy - ry * 0.9, sx + rx * 1.05, sy + ry * 1.2],
                   fill=(58, 62, 72, rng.randint(70, 110)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(img)

    plat_boxes = []
    for (cx, cy, rx, height) in plats:
        bx = draw_platform(draw, rng, cx, cy, rx, height)
        plat_boxes.append((cx, cy, rx, bx))
        boxes.append((LANDING_ID, *bx))

    # pawn on one platform (prefer the lowest/nearest)
    host = max(plat_boxes, key=lambda p: p[1])
    cx, cy, rx, _ = host
    scale = max(0.7, min(1.3, rx / 200.0))
    pbx = draw_pawn(draw, rng, cx + rng.uniform(-rx * 0.2, rx * 0.2), cy, scale)
    boxes.append((PIECE_ID, *pbx))

    return img, boxes


def to_yolo_line(box: tuple[int, int, int, int, int]) -> str:
    cls, x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, x0), min(W, x1)))
    y0, y1 = sorted((max(0, y0), min(H, y1)))
    cx = (x0 + x1) / 2 / W
    cy = (y0 + y1) / 2 / H
    bw = max(1, x1 - x0) / W
    bh = max(1, y1 - y0) / H
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="datasets/jump_synth")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", type=int, default=0, help="also write N annotated previews to <out>/preview")
    args = p.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    rng = random.Random(args.seed)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    n_val = int(round(args.n * args.val_ratio))
    for i in range(args.n):
        img, boxes = gen_scene(rng)
        split = "val" if i < n_val else "train"
        stem = f"synth_{i:05d}"
        img.save(out / "images" / split / f"{stem}.png")
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(to_yolo_line(b) for b in boxes) + "\n", encoding="utf-8")
        if args.preview and i < args.preview:
            prev = img.copy()
            d = ImageDraw.Draw(prev)
            for cls, x0, y0, x1, y1 in boxes:
                col = (0, 200, 255) if cls == PIECE_ID else (255, 0, 0)
                d.rectangle([x0, y0, x1, y1], outline=col, width=6)
            prev.save(out / "preview" / f"{stem}.png")

    (out / "dataset.yaml").write_text(
        "\n".join([f"path: {out.resolve()}", "train: images/train", "val: images/val",
                   "names:", "  0: piece", "  1: landing", ""]), encoding="utf-8")
    print(f"wrote {args.n} synthetic images to {out} (val={n_val})")
    if args.preview:
        print(f"previews: {out/'preview'}")


if __name__ == "__main__":
    main()
