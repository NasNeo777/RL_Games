#!/usr/bin/env python3
"""Cut the pawn out of real screenshots into RGBA sprites for synthetic scenes.

The pawn is much darker than whatever platform it stands on, so a brightness
threshold gives a clean alpha matte. We run the same find_piece heuristic
adb_jump_ppo uses to locate it, crop with padding, and matte by darkness.
Sprites land in tools/sprites/pawn_*.png for gen_synthetic_jump.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb_jump_ppo import find_piece  # noqa: E402


def matte(arr: np.ndarray, bbox, pad: int = 14) -> Image.Image:
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(arr.shape[1], x1 + pad), min(arr.shape[0], y1 + pad)
    crop = arr[y0:y1, x0:x1].astype(np.int16)
    bright = crop.max(2)
    # dark -> opaque; soft ramp around the pawn edge
    alpha = np.clip((150 - bright) * 4, 0, 255).astype(np.uint8)
    rgba = np.dstack([crop.clip(0, 255).astype(np.uint8), alpha])
    return Image.fromarray(rgba, "RGBA")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="datasets/jump_labelme/raw")
    # find_piece is NOT reliable on every screenshot (it can lock onto a shadow
    # or a dark platform edge), so candidates go to a review folder. Hand-copy
    # the good ones to tools/sprites/pawn_NN.png — never auto-overwrite curated
    # sprites.
    p.add_argument("--out", default="tools/sprites/candidates")
    p.add_argument("--max", type=int, default=20)
    args = p.parse_args()

    src = ROOT / args.src if not Path(args.src).is_absolute() else Path(args.src)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    seen_sizes: set[tuple[int, int]] = set()
    for img_path in sorted(src.glob("*.png")):
        if saved >= args.max:
            break
        arr = np.array(Image.open(img_path).convert("RGB"))
        piece = find_piece(arr)
        if piece is None:
            continue
        w = piece.bbox[2] - piece.bbox[0]
        h = piece.bbox[3] - piece.bbox[1]
        if w < 40 or h < 90 or h > 320:        # sanity: pawn proportions
            continue
        key = (w // 10, h // 10)
        if key in seen_sizes:                  # skip near-duplicate captures
            continue
        seen_sizes.add(key)
        sprite = matte(arr, piece.bbox)
        sprite.save(out / f"cand_{img_path.stem}.png")
        # preview the matte on a mid-gray bg so a bad crop is obvious at a glance
        canvas = Image.new("RGBA", sprite.size, (150, 156, 170, 255))
        canvas.alpha_composite(sprite)
        canvas.convert("RGB").save(out / f"cand_{img_path.stem}_onbg.png")
        saved += 1
        print(f"  {img_path.name} -> cand_{img_path.stem}.png  size={sprite.size}")
    print(f"\nsaved {saved} candidate mattes to {out}")
    print("Review *_onbg.png, then copy the GOOD ones to tools/sprites/pawn_NN.png")


if __name__ == "__main__":
    main()
