#!/usr/bin/env python3
"""Convert labeled Labelme rectangles into a YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASS_TO_ID = {
    "piece": 0,
    "landing": 1,
}


def rect_to_yolo(points: list[list[float]], width: int, height: int) -> tuple[float, float, float, float]:
    (x0, y0), (x1, y1) = points
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    cx = ((left + right) / 2.0) / width
    cy = ((top + bottom) / 2.0) / height
    bw = max(1.0, right - left) / width
    bh = max(1.0, bottom - top) / height
    return cx, cy, bw, bh


def write_dataset_yaml(out_dir: Path) -> None:
    (out_dir / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="datasets/jump_labelme/raw")
    p.add_argument("--out", default="datasets/jump_yolo_manual")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src_dir = Path(args.src)
    if not src_dir.is_absolute():
        src_dir = ROOT / src_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    json_files = sorted(src_dir.glob("*.json"))
    records: list[tuple[Path, list[str]]] = []
    skipped: list[str] = []
    for json_path in json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_name = data.get("imagePath") or f"{json_path.stem}.png"
        image_path = src_dir / image_name
        if not image_path.exists():
            skipped.append(f"{json_path.name}: missing image {image_name}")
            continue

        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            skipped.append(f"{json_path.name}: missing image size")
            continue

        lines: list[str] = []
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip().lower()
            if label not in CLASS_TO_ID:
                continue
            if shape.get("shape_type") != "rectangle":
                continue
            points = shape.get("points") or []
            if len(points) != 2:
                continue
            cx, cy, bw, bh = rect_to_yolo(points, width, height)
            lines.append(f"{CLASS_TO_ID[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            skipped.append(f"{json_path.name}: no usable rectangles")
            continue
        records.append((image_path, lines))

    random.Random(args.seed).shuffle(records)
    val_count = max(1, int(round(len(records) * args.val_ratio))) if len(records) > 1 else 0
    val_set = set(path for path, _ in records[:val_count])

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_path, lines in records:
        split = "val" if image_path in val_set else "train"
        shutil.copy2(image_path, out_dir / "images" / split / image_path.name)
        (out_dir / "labels" / split / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    write_dataset_yaml(out_dir)
    print(f"converted labeled images: {len(records)}")
    print(f"train: {len(records) - val_count}, val: {val_count}")
    if skipped:
        print("skipped:")
        for item in skipped[:20]:
            print(f"  - {item}")
        if len(skipped) > 20:
            print(f"  - ... {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
