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
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

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


# --- decorations -----------------------------------------------------------
# Real platforms carry surface decorations (cute faces, LED clocks, express-box
# logos, knobs, ring lids...). We render each on a flat RGBA tile, then warp it
# onto a face/top quad in perspective so it sits on the surface realistically.

def _find_coeffs(dst, src):
    m = []
    for (dx, dy), (sx, sy) in zip(dst, src):
        m.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        m.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(m, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, b)


def paste_quad(scene, tile, quad):
    """Warp axis-aligned RGBA `tile` so its corners land on `quad`
    ([tl, tr, br, bl] in scene coords) and composite it onto the scene."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, y0 = int(math.floor(min(xs))), int(math.floor(min(ys)))
    x1, y1 = int(math.ceil(max(xs))), int(math.ceil(max(ys)))
    ow, oh = max(1, x1 - x0), max(1, y1 - y0)
    dst = [(p[0] - x0, p[1] - y0) for p in quad]
    src = [(0, 0), (tile.width, 0), (tile.width, tile.height), (0, tile.height)]
    try:
        coeffs = _find_coeffs(dst, src)
    except np.linalg.LinAlgError:
        return
    warped = tile.transform((ow, oh), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    scene.alpha_composite(warped, (x0, y0))


def _qpt(corners, s, t):
    tl, tr, br, bl = corners
    return (tl[0] + s * (tr[0] - tl[0]) + t * (bl[0] - tl[0]),
            tl[1] + s * (tr[1] - tl[1]) + t * (bl[1] - tl[1]))


def _subquad(corners, s0, s1, t0, t1):
    return [_qpt(corners, s0, t0), _qpt(corners, s1, t0),
            _qpt(corners, s1, t1), _qpt(corners, s0, t1)]


def tile_face(rng):
    s = 200
    t = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    ink = (28, 28, 34, 255)
    er = rng.randint(20, 30)
    ey = rng.randint(95, 120)
    for ex in (70, 130):
        d.ellipse([ex - er, ey - er, ex + er, ey + er], fill=ink)
        d.ellipse([ex - er * 0.45, ey - er * 0.6, ex + er * 0.1, ey - er * 0.1],
                  fill=(255, 255, 255, 230))
    if rng.random() < 0.7:                       # pink cheeks
        for ex in (52, 148):
            d.ellipse([ex - 16, ey + 18, ex + 16, ey + 36], fill=(244, 150, 160, 150))
    if rng.random() < 0.5:                       # tiny mouth
        d.arc([88, ey + 6, 112, ey + 30], 20, 160, fill=ink, width=4)
    return t


def tile_clock(rng):
    t = Image.new("RGBA", (240, 150), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    d.rounded_rectangle([10, 10, 230, 140], radius=18, fill=(34, 30, 28, 255))
    txt = rng.choice(["14:16", "08:42", "21:05", "12:00", "06:30"])
    try:
        d.text((40, 45), txt, fill=(247, 150, 60, 255))
    except Exception:
        pass
    for i, x in enumerate(range(45, 210, 34)):   # faux LED segments
        d.rectangle([x, 55, x + 18, 95], outline=(247, 150, 60, 220), width=5)
    return t


def tile_dot(rng):
    s = 200
    t = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    d.ellipse([60, 60, 150, 150], fill=(245, 245, 245, 235))
    d.ellipse([95, 70, 120, 95], fill=(180, 180, 184, 235))
    return t


def tile_square(rng):
    s = 160
    t = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(t).rounded_rectangle([30, 30, 130, 130], radius=14,
                                        fill=(238, 238, 240, 235))
    return t


def tile_knob(rng):
    s = 160
    t = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(t).ellipse([45, 45, 115, 115], fill=(238, 238, 240, 235))
    return t


def tile_rings(rng, base):
    s = 220
    t = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    c = s // 2
    for i, r in enumerate(range(100, 30, -22)):
        col = shade(base, 0.82 if i % 2 else 1.08) + (255,)
        d.ellipse([c - r, c - r, c + r, c + r], fill=col)
    return t


def decorate_cuboid(im, rng, left_face, right_face, top, base):
    r = rng.random()
    if r < 0.32:
        paste_quad(im, tile_face(rng), _subquad(right_face, 0.24, 0.86, 0.20, 0.78))
    elif r < 0.48:
        paste_quad(im, tile_clock(rng), _subquad(right_face, 0.30, 0.9, 0.22, 0.62))
    elif r < 0.62:
        paste_quad(im, tile_dot(rng), _subquad(right_face, 0.45, 0.85, 0.30, 0.78))
    if rng.random() < 0.4:
        kind = tile_square(rng) if rng.random() < 0.5 else tile_knob(rng)
        paste_quad(im, kind, _subquad(left_face, 0.18, 0.6, 0.28, 0.78))


def draw_cuboid(im, draw, rng, cx, cy, rx, total_h, base):
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

    # face quads ([tl, tr, br, bl]); height uses the FIRST main slab so the
    # decoration sits on the coloured band, not across stripe seams.
    bh = total_h
    left_face = [left_pt, bot_pt, (bot_pt[0], bot_pt[1] + bh), (left_pt[0], left_pt[1] + bh)]
    right_face = [bot_pt, right_pt, (right_pt[0], right_pt[1] + bh), (bot_pt[0], bot_pt[1] + bh)]
    decorate_cuboid(im, rng, left_face, right_face, top, base)
    return int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)


def draw_round(im, draw, rng, cx, cy, rx, total_h, base, ry_ratio):
    """Cylinder (round) / disc (oval) platform: elliptical top + curved body.

    The body is a *smooth* horizontal light->dark gradient (real cylinder
    shading), not hard vertical bands; stripes are thin rings, not flat slabs.
    ry_ratio ~0.5 reads as a circle in isometric; smaller reads as a flat disc.
    """
    rx_i = max(2, int(round(rx)))
    th = max(2, int(round(total_h)))
    ry = rx * ry_ratio
    ry_i = max(1, int(round(ry)))
    w = 2 * rx_i
    H = th + ry_i                      # extra room for the rounded bottom rim

    # smooth cylindrical shading across the width: brightest left-of-centre
    # (light from upper-left), falling off to a dark right edge.
    u = np.linspace(-1.0, 1.0, w)
    theta = u * (math.pi / 2)
    fac = 0.42 + 0.60 * np.clip(np.cos(theta + math.pi / 5), 0.0, 1.0)  # (w,)
    bow = (ry_i * np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))).astype(int)  # (w,) front dip

    # base body colour, then thin stripe rings that *wrap* the cylinder (a ring
    # bows downward at the front like the top ellipse).
    body_c = np.tile(np.array(base, dtype=np.float32), (H, w, 1))
    st = max(5, int(th * 0.07))
    for _ in range(rng.choice([0, 1, 1, 2])):
        sy = rng.randint(int(th * 0.28), max(int(th * 0.28) + 1, int(th * 0.74)))
        for j in range(w):
            y0 = sy + int(bow[j])
            body_c[y0:y0 + st, j] = STRIPE

    rgb = (body_c * fac[None, :, None]).clip(0, 255).astype(np.uint8)
    # alpha: each column is opaque only down to the curved bottom rim, so the
    # base reads as a real rounded bottom instead of a flat dark cap.
    yy = np.arange(H)[:, None]
    alpha = ((yy < (th + bow[None, :]))).astype(np.uint8) * 255          # (H,w)
    rgba = np.dstack([rgb, alpha])
    im.alpha_composite(Image.fromarray(rgba, "RGBA"), (int(round(cx)) - rx_i, int(round(cy))))

    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=base)
    draw.arc([cx - rx, cy - ry, cx + rx, cy + ry], 180, 360, fill=shade(base, 1.12), width=2)

    # concentric ring lid on the top ellipse (like a medicine-box / manhole top)
    if rng.random() < 0.35:
        ring_base = rng.choice(TOP_COLORS)
        top_quad = [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]
        paste_quad(im, tile_rings(rng, ring_base), top_quad)
    return int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)


def draw_table(im, draw, rng, cx, cy, rx, total_h, base):
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


def draw_platform(im, draw, rng, cx, cy, rx, total_h, shape) -> tuple[int, int, int, int]:
    """Dispatch to the given platform shape. Returns the top-surface bbox
    (the landing label) — never the extruded sides/legs."""
    base = rng.choice(TOP_COLORS)
    if shape == "round":
        return draw_round(im, draw, rng, cx, cy, rx, total_h, base, rng.uniform(0.46, 0.54))
    if shape == "oval":
        return draw_round(im, draw, rng, cx, cy, rx, total_h, base, rng.uniform(0.30, 0.40))
    if shape == "table":
        return draw_table(im, draw, rng, cx, cy, rx, total_h, base)
    return draw_cuboid(im, draw, rng, cx, cy, rx, total_h, base)


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
            if all(abs(cx - ox) > (rx + orx) * 0.72 for ox, oy, orx, _, _ in plats):
                plats.append((cx, cy, rx, total_h, rng.choice(SHAPES)))
                break
    plats.sort(key=lambda p: p[1])  # far (higher) first

    # shape-matched cast shadows (diamond for boxes, ellipse for round), flat
    # and crisp -- no Gaussian feather (the "毛边" the real game doesn't have).
    # Light comes from the upper-right, so the footprint is offset down-left.
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for (cx, cy, rx, total_h, shape) in plats:
        ry = rx * 0.5
        scx, scy = cx - rx * 0.5, cy + total_h + ry * 0.35
        col = (52, 56, 66, 90)
        if shape in ("round", "oval"):
            sry = rx * (0.33 if shape == "oval" else 0.5)
            sd.ellipse([scx - rx, scy - sry, scx + rx, scy + sry], fill=col)
        else:
            sd.polygon(diamond(scx, scy, rx, ry), fill=col)
    img = Image.alpha_composite(img, shadow)
    draw = ImageDraw.Draw(img)

    plat_boxes = []
    for (cx, cy, rx, total_h, shape) in plats:
        bx = draw_platform(img, draw, rng, cx, cy, rx, total_h, shape)
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
