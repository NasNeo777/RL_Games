#!/usr/bin/env python3
"""Render synthetic Jump (跳一跳) scenes with perfect, complete YOLO labels.

Copy-paste augmentation, tuned to look like the real game:
  * piece  -> a real pawn cut out of a screenshot (tools/sprites/pawn_*.png),
              pasted with scale/brightness jitter (no hand-drawn pawn).
  * landing-> a cuboid built by *stacking layered slabs* (the cake look):
              one top diamond + alternating-color side bands.
Background is a soft vertical gradient, platforms cast soft shadows.

Every box is exact and *every* platform is labeled (the thing simple CV can't
do on real screenshots). The landing box covers ONLY the top diamond surface,
never the extruded sides. Output is a ready YOLO dataset (images/ + labels/);
pretrain on this, then finetune on real images.

Needs curated pawn sprites in tools/sprites/pawn_*.png. To add more, run
tools/extract_pawn_sprites.py (writes candidates to review) and hand-copy the
good ones — find_piece is not reliable enough to auto-trust every matte.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
SPRITE_DIR = ROOT / "tools" / "sprites"

W, H = 1080, 2400
PIECE_ID, LANDING_ID = 0, 1

# Platform colors sampled to match the real game's flat-shaded palette.
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
    (120, 124, 134),  # gray
]
STRIPE = (238, 238, 240)  # the white band seen between slabs


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def shade(color, f):
    return tuple(max(0, min(255, int(round(c * f)))) for c in color)


def load_pawns() -> list[Image.Image]:
    sprites = sorted(SPRITE_DIR.glob("pawn_*.png"))
    if not sprites:
        raise SystemExit(
            f"no pawn sprites in {SPRITE_DIR}. Run tools/extract_pawn_sprites.py first.")
    return [Image.open(p).convert("RGBA") for p in sprites]


def draw_gradient_bg(rng: random.Random) -> Image.Image:
    top = np.array([rng.randint(196, 214), rng.randint(200, 216), rng.randint(208, 224)])
    bot = np.array([rng.randint(150, 180), rng.randint(156, 186), rng.randint(170, 198)])
    t = np.linspace(0.0, 1.0, H)[:, None]
    col = (top[None, :] + (bot - top)[None, :] * t).astype(np.uint8)  # (H,3)
    arr = np.broadcast_to(col[:, None, :], (H, W, 3)).copy()
    return Image.fromarray(arr, "RGB")


def diamond(cx, cy, rx, ry):
    return [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]


def build_layers(rng, base, total_h):
    """Slab layers top->bottom: main color, thin white stripe, main, ... (cake)."""
    layers, remaining, main = [], total_h, True
    while remaining > 4:
        if main:
            t = min(remaining, rng.uniform(total_h * 0.28, total_h * 0.5))
            layers.append((t, base))
        else:
            t = min(remaining, rng.uniform(12, 26))
            layers.append((t, STRIPE))
        remaining -= t
        main = not main
    return layers or [(total_h, base)]


def draw_cuboid(draw, rng, cx, cy, rx, total_h, base):
    """Square platform: top diamond + two stacked-slab side faces."""
    ry = rx * rng.uniform(0.48, 0.55)
    left_pt, right_pt, bot_pt = (cx - rx, cy), (cx + rx, cy), (cx, cy + ry)
    y = 0.0
    for thick, col in build_layers(rng, base, total_h):
        y0, y1 = y, y + thick
        draw.polygon([(left_pt[0], left_pt[1] + y0), (bot_pt[0], bot_pt[1] + y0),
                      (bot_pt[0], bot_pt[1] + y1), (left_pt[0], left_pt[1] + y1)],
                     fill=shade(col, 0.82))
        draw.polygon([(bot_pt[0], bot_pt[1] + y0), (right_pt[0], right_pt[1] + y0),
                      (right_pt[0], right_pt[1] + y1), (bot_pt[0], bot_pt[1] + y1)],
                     fill=shade(col, 0.60))
        y = y1
    top = diamond(cx, cy, rx, ry)
    draw.polygon(top, fill=base)
    draw.line([top[3], top[0], top[1]], fill=shade(base, 1.12), width=2)
    return int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)


def draw_round(draw, rng, cx, cy, rx, total_h, base, ry_ratio):
    """Cylinder (round) / disc (oval) platform: elliptical top + curved body.

    ry_ratio ~0.5 reads as a circle in isometric; smaller/larger reads oval.
    """
    ry = rx * ry_ratio
    # rounded bottom: bottom ellipse peeks below the straight body
    draw.ellipse([cx - rx, cy + total_h - ry, cx + rx, cy + total_h + ry],
                 fill=shade(base, 0.58))
    # body split into horizontal stripe bands
    y = 0.0
    for thick, col in build_layers(rng, base, total_h):
        y0, y1 = cy + y, cy + y + thick
        draw.rectangle([cx - rx, y0, cx + rx, y1], fill=shade(col, 0.72))
        # subtle left-light / right-dark shading on the cylinder body
        draw.rectangle([cx - rx, y0, cx - rx * 0.2, y1], fill=shade(col, 0.82))
        draw.rectangle([cx + rx * 0.3, y0, cx + rx, y1], fill=shade(col, 0.6))
        y += thick
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=base)
    draw.arc([cx - rx, cy - ry, cx + rx, cy + ry], 180, 360, fill=shade(base, 1.12), width=2)
    return int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)


def draw_table(draw, rng, cx, cy, rx, total_h, base):
    """Table: a thin top slab on four legs. Landing = the tabletop surface."""
    ry = rx * rng.uniform(0.48, 0.55)
    top_thick = rng.uniform(18, 30)
    leg_h = total_h - top_thick
    leg_w = max(8, rx * 0.12)
    # legs drop from inset corners of the top diamond (front three are visible)
    for vx, vy, f in [(cx - rx * 0.78, cy, 0.74), (cx, cy + ry * 0.82, 0.66),
                      (cx + rx * 0.78, cy, 0.6)]:
        draw.rectangle([vx - leg_w / 2, vy, vx + leg_w / 2, vy + leg_h],
                       fill=shade(base, f))
    # thin tabletop slab
    left_pt, right_pt, bot_pt = (cx - rx, cy), (cx + rx, cy), (cx, cy + ry)
    draw.polygon([left_pt, bot_pt, (bot_pt[0], bot_pt[1] + top_thick),
                  (left_pt[0], left_pt[1] + top_thick)], fill=shade(base, 0.82))
    draw.polygon([bot_pt, right_pt, (right_pt[0], right_pt[1] + top_thick),
                  (bot_pt[0], bot_pt[1] + top_thick)], fill=shade(base, 0.6))
    top = diamond(cx, cy, rx, ry)
    draw.polygon(top, fill=base)
    draw.line([top[3], top[0], top[1]], fill=shade(base, 1.12), width=2)
    return int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)


SHAPES = ["cuboid", "cuboid", "cuboid", "round", "round", "oval", "table"]


def draw_platform(draw, rng, cx, cy, rx, total_h) -> tuple[int, int, int, int]:
    """Dispatch to a random platform shape. Returns the top-surface bbox
    (the landing label) — never the extruded sides/legs."""
    base = rng.choice(TOP_COLORS)
    shape = rng.choice(SHAPES)
    if shape == "round":
        return draw_round(draw, rng, cx, cy, rx, total_h, base, rng.uniform(0.46, 0.54))
    if shape == "oval":
        return draw_round(draw, rng, cx, cy, rx, total_h, base, rng.uniform(0.30, 0.40))
    if shape == "table":
        return draw_table(draw, rng, cx, cy, rx, total_h, base)
    return draw_cuboid(draw, rng, cx, cy, rx, total_h, base)


def paste_pawn(scene: Image.Image, pawn: Image.Image, rng: random.Random,
               cx: float, top_y: float, rx: float) -> tuple[int, int, int, int]:
    # scale pawn so its width ~ 0.42 of platform half-width range
    target_w = max(46, min(120, rx * rng.uniform(0.62, 0.82)))
    scale = target_w / pawn.width
    pw, ph = int(pawn.width * scale), int(pawn.height * scale)
    sp = pawn.resize((pw, ph), Image.LANCZOS)
    if rng.random() < 0.5:                      # mirror the highlight side for variety
        sp = sp.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        sp = ImageEnhance.Brightness(sp).enhance(rng.uniform(0.85, 1.15))
    px = int(cx - pw / 2 + rng.uniform(-rx * 0.12, rx * 0.12))
    py = int(top_y - ph + ph * 0.06)   # feet just below the top-diamond center
    scene.alpha_composite(sp, (px, py))
    # tight bbox from alpha
    bbox = sp.getbbox()
    ax0, ay0, ax1, ay1 = bbox
    return px + ax0, py + ay0, px + ax1, py + ay1


def gen_scene(rng: random.Random, pawns: list[Image.Image]):
    img = draw_gradient_bg(rng).convert("RGBA")
    boxes: list[tuple[int, int, int, int, int]] = []

    n = rng.choice([1, 2, 2, 2, 3, 3, 4])
    plats = []
    for _ in range(n):
        for _try in range(40):
            rx = rng.uniform(120, 250)
            total_h = rng.uniform(70, 190)
            cx = rng.uniform(rx + 30, W - rx - 30)
            cy = rng.uniform(H * 0.32, H * 0.64)
            # spread platforms horizontally (diagonal layout like the real game),
            # never stacked vertically
            if all(abs(cx - ox) > (rx + orx) * 0.72 for ox, oy, orx, _ in plats):
                plats.append((cx, cy, rx, total_h))
                break
    plats.sort(key=lambda p: p[1])  # far (higher) first

    # soft shadows on one blurred layer under the platforms
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for (cx, cy, rx, total_h) in plats:
        ry = rx * 0.5
        sx, sy = cx - rx * 0.35, cy + total_h * 0.9
        sd.ellipse([sx - rx * 1.15, sy - ry * 0.9, sx + rx * 1.05, sy + ry * 1.25],
                   fill=(58, 62, 72, rng.randint(70, 105)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    img = Image.alpha_composite(img, shadow)
    draw = ImageDraw.Draw(img)

    plat_boxes = []
    for (cx, cy, rx, total_h) in plats:
        bx = draw_platform(draw, rng, cx, cy, rx, total_h)
        plat_boxes.append((cx, cy, rx, bx))
        boxes.append((LANDING_ID, *bx))

    host = max(plat_boxes, key=lambda p: p[1])
    cx, cy, rx, _ = host
    pbx = paste_pawn(img, rng.choice(pawns), rng, cx, cy, rx)
    boxes.append((PIECE_ID, *pbx))

    return img.convert("RGB"), boxes


def to_yolo_line(box):
    cls, x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, x0), min(W, x1)))
    y0, y1 = sorted((max(0, y0), min(H, y1)))
    return (f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
            f"{max(1, x1 - x0) / W:.6f} {max(1, y1 - y0) / H:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="datasets/jump_synth")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", type=int, default=0, help="also write N annotated previews")
    args = p.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    rng = random.Random(args.seed)
    pawns = load_pawns()

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    n_val = int(round(args.n * args.val_ratio))
    for i in range(args.n):
        img, boxes = gen_scene(rng, pawns)
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


if __name__ == "__main__":
    main()
