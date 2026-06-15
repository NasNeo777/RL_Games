#!/usr/bin/env python3
"""Train a YOLO detector for Jump piece/landing detection."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="datasets/jump_yolo/dataset.yaml")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="jump_yolo")
    args = p.parse_args()

    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise SystemExit(f"dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
