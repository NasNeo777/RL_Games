#!/usr/bin/env python3
"""Capture raw Jump screenshots from adb and export YOLO labels."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adb_jump_ppo import capture_screen, heuristic_detect, save_debug_image


def yolo_line(cls_id: int, bbox: tuple[int, int, int, int], width: int, height: int) -> str:
    x0, y0, x1, y1 = bbox
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    bw = max(1.0, x1 - x0) / width
    bh = max(1.0, y1 - y0) / height
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def ensure_dataset_yaml(root: Path) -> Path:
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: piece",
                "  1: landing",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", help="adb serial; default uses the only attached device")
    p.add_argument("--count", type=int, default=80, help="screenshots to capture")
    p.add_argument("--interval", type=float, default=0.8, help="seconds between captures")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--out-dir", default="datasets/jump_yolo")
    p.add_argument("--prefix", default="jump")
    p.add_argument("--preview", action="store_true", help="also save annotated previews")
    args = p.parse_args()

    root = Path(args.out_dir)
    if not root.is_absolute():
        root = ROOT / root
    img_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split
    preview_dir = root / "preview" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = ensure_dataset_yaml(root)

    saved = 0
    skipped = 0
    for idx in range(args.count):
        img = capture_screen(args.serial)
        arr = np.array(img)
        piece, target = heuristic_detect(arr)
        stem = f"{args.prefix}_{args.split}_{idx:04d}"
        img_path = img_dir / f"{stem}.png"
        label_path = label_dir / f"{stem}.txt"
        img.save(img_path)

        if piece is None or target is None:
            label_path.write_text("", encoding="utf-8")
            skipped += 1
            print(f"[{idx + 1}/{args.count}] saved empty label -> {img_path.name}")
        else:
            w, h = img.size
            lines = [
                yolo_line(0, piece.bbox, w, h),
                yolo_line(1, target.bbox, w, h),
            ]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            saved += 1
            print(f"[{idx + 1}/{args.count}] labeled -> {img_path.name}")
            if args.preview:
                save_debug_image(img, preview_dir / f"{stem}.png", piece, target, "heuristic bootstrap")

        if idx + 1 < args.count:
            time.sleep(args.interval)

    print(f"dataset yaml: {yaml_path}")
    print(f"labeled images: {saved}, empty labels: {skipped}")


if __name__ == "__main__":
    main()
