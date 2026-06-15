#!/usr/bin/env python3
"""Detect the piece/target, then jump over adb with the empirical linear law.

Press time (ms) = pixel distance from the piece's foot to the target centre
times a constant (`--coef`, default 1.35) -- the well-known WeChat-Jump rule.
"""

from __future__ import annotations

import argparse
import io
import math
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Piece:
    x: float
    y: float
    bbox: tuple[int, int, int, int]


@dataclass
class Target:
    x: float
    y: float
    half_width_px: float
    bbox: tuple[int, int, int, int]
    seed: tuple[int, int]


YOLO_PIECE_NAMES = {"piece", "pawn", "player", "jumper"}
YOLO_TARGET_NAMES = {"landing", "target", "platform", "goal"}


class YoloJumpDetector:
    def __init__(self, model_path: str, conf: float = 0.25, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.conf = conf
        self.device = device
        names = getattr(self.model, "names", {}) or {}
        self.names = {int(k): str(v).lower() for k, v in names.items()}

    def detect(self, arr: np.ndarray) -> tuple[Piece | None, Target | None]:
        results = self.model.predict(source=arr, conf=self.conf, device=self.device, verbose=False)

        if not results:
            return None, None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None, None

        piece_box: tuple[float, tuple[int, int, int, int]] | None = None
        landings: list[tuple[int, int, int, int]] = []

        for box in boxes:
            cls_id = int(float(box.cls[0]))
            conf = float(box.conf[0])
            x0, y0, x1, y1 = [int(round(v)) for v in box.xyxy[0].tolist()]
            name = self.names.get(cls_id, "")
            if name in YOLO_PIECE_NAMES or (not name and cls_id == 0):
                if piece_box is None or conf > piece_box[0]:  # one piece: keep best
                    piece_box = (conf, (x0, y0, x1, y1))
            elif name in YOLO_TARGET_NAMES or (not name and cls_id == 1):
                landings.append((x0, y0, x1, y1))

        piece = piece_from_bbox(piece_box[1]) if piece_box else None

        target_bbox = self._pick_target(landings, piece)
        target = target_from_bbox(target_bbox) if target_bbox else None
        return piece, target

    @staticmethod
    def _pick_target(
        landings: list[tuple[int, int, int, int]],
        piece: Piece | None,
    ) -> tuple[int, int, int, int] | None:
        """With several platforms detected, jump to the *topmost* one (highest
        on screen). Skip the platform the piece is currently standing on (its
        foot point lies inside that box) since that is not a jump target."""
        if not landings:
            return None
        cands = landings
        if piece is not None:
            others = [
                b for b in landings
                if not (b[0] - 12 <= piece.x <= b[2] + 12 and b[1] - 12 <= piece.y <= b[3] + 40)
            ]
            if others:
                cands = others
        # topmost = smallest vertical center
        return min(cands, key=lambda b: (b[1] + b[3]) / 2)


def piece_from_bbox(bbox: tuple[int, int, int, int]) -> Piece:
    x0, y0, x1, y1 = bbox
    x = (x0 + x1) / 2.0
    y = float(y1 - max(8, int((y1 - y0) * 0.06)))
    return Piece(x=x, y=y, bbox=bbox)


def target_from_bbox(bbox: tuple[int, int, int, int]) -> Target:
    x0, y0, x1, y1 = bbox
    x = (x0 + x1) / 2.0
    y = (y0 + y1) / 2.0
    half_width_px = max(12.0, (x1 - x0) / 2.0)
    seed = (int(round(x)), int(round(y)))
    return Target(x=x, y=y, half_width_px=half_width_px, bbox=bbox, seed=seed)


def adb_bytes(serial: str | None, *args: str) -> bytes:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += list(args)
    return subprocess.check_output(cmd)


def adb_run(serial: str | None, *args: str) -> None:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += list(args)
    subprocess.run(cmd, check=True)


def capture_screen(serial: str | None) -> Image.Image:
    data = adb_bytes(serial, "exec-out", "screencap", "-p")
    return Image.open(io.BytesIO(data)).convert("RGB")


def heuristic_detect(arr: np.ndarray) -> tuple[Piece | None, Target | None]:
    piece = find_piece(arr)
    target = find_target(arr, piece) if piece is not None else None
    return piece, target


def capture_and_detect(
    serial: str | None,
    detector: YoloJumpDetector | None = None,
) -> tuple[Image.Image, np.ndarray, Piece | None, Target | None]:
    img = capture_screen(serial)
    arr = np.array(img)
    if detector is not None:
        piece, target = detector.detect(arr)
        if piece is None or target is None:
            piece, target = heuristic_detect(arr)
    else:
        piece, target = heuristic_detect(arr)
    return img, arr, piece, target


def connected_components(mask: np.ndarray, min_pixels: int = 20) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    comps: list[np.ndarray] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        q = deque([(y, x)])
        visited[y, x] = True
        pts: list[tuple[int, int]] = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        if len(pts) >= min_pixels:
            comps.append(np.array(pts, dtype=np.int32))
    return comps


def merge_piece_components(comps: list[np.ndarray]) -> np.ndarray | None:
    if not comps:
        return None
    comps = sorted(comps, key=len, reverse=True)
    merged = [comps[0]]
    x0 = comps[0][:, 1].min()
    x1 = comps[0][:, 1].max()
    y0 = comps[0][:, 0].min()
    y1 = comps[0][:, 0].max()
    for arr in comps[1:]:
        ax0 = arr[:, 1].min()
        ax1 = arr[:, 1].max()
        ay0 = arr[:, 0].min()
        ay1 = arr[:, 0].max()
        overlap = min(x1, ax1) - max(x0, ax0)
        gap = max(0, max(y0, ay0) - min(y1, ay1))
        if overlap >= -8 and gap <= 80:
            merged.append(arr)
    return np.concatenate(merged, axis=0)


def find_piece(arr: np.ndarray) -> Piece | None:
    h, w, _ = arr.shape
    r, g, b = [arr[:, :, i] for i in range(3)]
    avg = (r.astype(np.int16) + g.astype(np.int16) + b.astype(np.int16)) / 3.0
    mask = (
        (r >= 35)
        & (r <= 95)
        & (g >= 30)
        & (g <= 85)
        & (b >= 60)
        & (b <= 125)
        & ((b.astype(np.int16) - r.astype(np.int16)) >= 12)
        & (avg < 105)
    )
    mask[: int(h * 0.30), :] = False
    mask[int(h * 0.78) :, :] = False
    comps = connected_components(mask, min_pixels=80)
    comps = [
        c
        for c in comps
        if int(c[:, 0].mean()) > h * 0.38 and int(c[:, 0].mean()) < h * 0.68
    ]
    merged = merge_piece_components(comps)
    if merged is None or len(merged) < 500:
        return None
    x0 = int(merged[:, 1].min())
    x1 = int(merged[:, 1].max())
    y0 = int(merged[:, 0].min())
    y1 = int(merged[:, 0].max())
    x = float(merged[:, 1].mean())
    y = float(y1 - max(8, int((y1 - y0) * 0.06)))
    return Piece(x=x, y=y, bbox=(x0, y0, x1, y1))


def row_runs(mask_row: np.ndarray, min_len: int = 8, max_len: int = 320) -> list[tuple[int, int]]:
    xs = np.where(mask_row)[0]
    if len(xs) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = xs[0]
    prev = xs[0]
    for x in xs[1:].tolist() + [None]:
        if x is None or x != prev + 1:
            if min_len <= prev - start + 1 <= max_len:
                runs.append((int(start), int(prev)))
            if x is not None:
                start = x
        prev = x if x is not None else prev
    return runs


def grow_mask_region(
    mask: np.ndarray,
    seed_x: int,
    seed_y: int,
    x0: int,
    y0: int,
) -> np.ndarray:
    h, w = mask.shape
    sx = max(0, min(w - 1, seed_x - x0))
    sy = max(0, min(h - 1, seed_y - y0))
    if not mask[sy, sx]:
        return np.empty((0, 2), dtype=np.int32)
    q = deque([(sy, sx)])
    visited = np.zeros((h, w), dtype=bool)
    visited[sy, sx] = True
    region: list[tuple[int, int]] = []
    while q:
        cy, cx = q.popleft()
        region.append((cy, cx))
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    if not region:
        return np.empty((0, 2), dtype=np.int32)
    pts = np.array(region, dtype=np.int32)
    pts[:, 0] += y0
    pts[:, 1] += x0
    return pts


def landing_band_from_region(
    region: np.ndarray,
    seed_x: int,
) -> tuple[float, float, float, tuple[int, int, int, int]] | None:
    if len(region) < 50:
        return None

    y0 = int(region[:, 0].min())
    y1 = int(region[:, 0].max())
    height = y1 - y0 + 1
    if height < 10:
        return None
    cap_limit = y0 + int(height * 0.64)
    cap = region[region[:, 0] <= cap_limit]
    if len(cap) < 40:
        cap = region

    cap_y = float(np.quantile(cap[:, 0], 0.72))
    row_window = max(5, int(round(height * 0.04)))
    band = cap[np.abs(cap[:, 0] - cap_y) <= row_window]
    if len(band) < 20:
        band = cap

    cx = float(np.median(band[:, 1]))
    cy = float(np.quantile(band[:, 0], 0.58))
    left = float(np.quantile(band[:, 1], 0.08))
    right = float(np.quantile(band[:, 1], 0.92))
    half_width = max(12.0, (right - left) / 2.0)
    bbox = (
        int(np.floor(left)),
        int(np.floor(np.quantile(cap[:, 0], 0.18))),
        int(np.ceil(right)),
        int(np.ceil(np.quantile(cap[:, 0], 0.82))),
    )
    return cx, cy, half_width, bbox


def find_target(arr: np.ndarray, piece: Piece) -> Target | None:
    h, w, _ = arr.shape
    arrf = arr.astype(np.float32)
    # The real game background has a gentle vertical gradient. Use a per-row
    # bright-percentile estimate instead of one fixed color, otherwise darker
    # targets on lower rows blend into the background model.
    bg_rows = np.percentile(arrf, 90, axis=1)
    bg_dist = np.sqrt(((arrf - bg_rows[:, None, :]) ** 2).sum(axis=2))
    mask = bg_dist > 18

    px0, py0, px1, py1 = piece.bbox
    ex0 = max(0, px0 - 140)
    ex1 = min(w, px1 + 140)
    ey0 = max(0, py0 - 70)
    ey1 = min(h, py1 + 90)
    mask[ey0:ey1, ex0:ex1] = False

    run_seed: tuple[int, int, int] | None = None
    run_seed_score = float("-inf")
    for y in range(int(h * 0.30), int(h * 0.62)):
        runs = row_runs(mask[y], min_len=14, max_len=int(w * 0.32))
        runs = [
            (sx, ex)
            for sx, ex in runs
            if sx > 10 and ex < w - 10 and abs((sx + ex) / 2 - piece.x) > 75
        ]
        for sx, ex in runs:
            cx = (sx + ex) / 2.0
            width = ex - sx + 1
            score = width * 1.7 + y * 0.22 - abs(cx - piece.x) * 0.12
            if score > run_seed_score:
                run_seed_score = score
                run_seed = (y, sx, ex)
    if run_seed is None:
        return None

    row_y, sx, ex = run_seed
    rx0 = max(0, sx - 140)
    rx1 = min(w, ex + 140)
    ry0 = max(0, row_y)
    ry1 = min(h, row_y + 220)
    roi = arr[ry0:ry1, rx0:rx1]
    roi_dist = bg_dist[ry0:ry1, rx0:rx1]
    non_bg = roi_dist > 18
    if not np.any(non_bg):
        return None

    ys, xs = np.where(non_bg)
    min_y = int(ys.min())
    band = (ys >= min_y) & (ys <= min_y + 12)
    band_xs = xs[band]
    if len(band_xs) == 0:
        return None
    seed_x = int(np.median(band_xs)) + rx0
    candidates = [(min_y + dy, seed_x) for dy in range(0, min(72, roi.shape[0] - min_y), 4)]
    best_seed = None
    best_score = -1.0
    best_region = None
    best_landing = None
    for rel_y, sx_abs in candidates:
        y_abs = ry0 + rel_y
        if y_abs >= h:
            continue
        x_abs = max(rx0, min(rx1 - 1, sx_abs))
        px = arr[y_abs, x_abs].astype(np.int16)
        score = float(np.linalg.norm(px - bg_rows[y_abs]))
        if score < 10:
            continue
        region = grow_mask_region(non_bg, x_abs, y_abs, rx0, ry0)
        if len(region) < 40:
            continue
        landing = landing_band_from_region(region, x_abs)
        if landing is not None:
            _, ly, half_width, _ = landing
            region_y0 = float(region[:, 0].min())
            region_y1 = float(region[:, 0].max())
            rel_landing = (ly - region_y0) / max(1.0, region_y1 - region_y0)
            candidate_score = half_width * 2.0 + 18.0 * max(0.0, 1.0 - rel_landing / 0.45) + 0.05 * score
        else:
            candidate_score = len(region) * 0.01 + 0.05 * score
        if candidate_score > best_score:
            best_score = candidate_score
            best_seed = (x_abs, y_abs)
            best_region = region
            best_landing = landing
    if best_seed is None or best_region is None:
        return None

    region = best_region
    landing = best_landing
    if landing is not None:
        x, y, half_width_px, bbox = landing
        x0, y0, x1, y1 = bbox
    else:
        x0 = int(region[:, 1].min())
        x1 = int(region[:, 1].max())
        y0 = int(region[:, 0].min())
        y1 = int(region[:, 0].max())
        x = float(region[:, 1].mean())
        y = float(region[:, 0].mean())
        half_width_px = (x1 - x0) / 2.0
    return Target(
        x=x,
        y=y,
        half_width_px=half_width_px,
        bbox=(x0, y0, x1, y1),
        seed=best_seed,
    )


def long_press(serial: str | None, x: int, y: int, ms: int) -> None:
    adb_run(serial, "shell", "input", "swipe", str(x), str(y), str(x + 1), str(y + 1), str(ms))


def tap(serial: str | None, x: int, y: int) -> None:
    adb_run(serial, "shell", "input", "tap", str(x), str(y))


def save_debug_image(
    img: Image.Image,
    out_path: Path,
    piece: Piece | None,
    target: Target | None,
    text: str,
) -> None:
    dbg = img.copy()
    draw = ImageDraw.Draw(dbg)
    if piece:
        draw.rectangle(piece.bbox, outline="red", width=4)
        draw.ellipse((piece.x - 8, piece.y - 8, piece.x + 8, piece.y + 8), outline="red", width=4)
    if target:
        draw.rectangle(target.bbox, outline="lime", width=4)
        draw.ellipse((target.x - 8, target.y - 8, target.x + 8, target.y + 8), outline="lime", width=4)
        draw.ellipse((target.seed[0] - 6, target.seed[1] - 6, target.seed[0] + 6, target.seed[1] + 6), outline="yellow", width=3)
    draw.text((24, 24), text, fill="black")
    dbg.save(out_path)


def save_raw_image(
    img: Image.Image,
    out_dir: Path | None,
    frame_idx: int,
) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"prejump_{frame_idx:05d}.png")


def wait_until_ready(
    serial: str | None,
    min_air_time: float,
    poll_delay: float,
    ready_streak: int,
    timeout: float,
    detector: YoloJumpDetector | None = None,
) -> bool:
    time.sleep(min_air_time)
    deadline = time.time() + timeout
    prev_piece = None
    prev_target = None
    streak = 0
    while time.time() < deadline:
        _, _, piece, target = capture_and_detect(serial, detector=detector)
        if piece is None or target is None:
            prev_piece = None
            prev_target = None
            streak = 0
            time.sleep(poll_delay)
            continue

        if prev_piece is None or prev_target is None:
            prev_piece = piece
            prev_target = target
            streak = 1
            time.sleep(poll_delay)
            continue

        stable_piece = abs(piece.x - prev_piece.x) <= 10 and abs(piece.y - prev_piece.y) <= 12
        stable_target = abs(target.x - prev_target.x) <= 12 and abs(target.y - prev_target.y) <= 12
        if stable_piece and stable_target:
            streak += 1
            if streak >= ready_streak:
                return True
        else:
            streak = 1
        prev_piece = piece
        prev_target = target
        time.sleep(poll_delay)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", help="adb serial; default uses the only attached device")
    p.add_argument("--detector", choices=["auto", "heuristic", "yolo"], default="yolo")
    p.add_argument("--yolo-model", default="runs/detect/runs/jump_yolo_synth/weights/best.pt")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-device", default="cpu")
    p.add_argument("--coef", type=float, default=1.35, help="press ms per pixel of gap distance")
    p.add_argument("--max-jumps", type=int, default=0, help="0 means unlimited")
    p.add_argument("--min-air-time", type=float, default=0.34, help="minimum seconds to wait before checking whether the piece has landed")
    p.add_argument("--poll-delay", type=float, default=0.05, help="seconds between ready checks after a jump")
    p.add_argument("--ready-timeout", type=float, default=1.0, help="seconds to wait for the next stable ready state")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug-dir", default="debug_jump_ppo")
    p.add_argument("--raw-dir", help="save unannotated screenshots for later labeling")
    args = p.parse_args()

    detector_name = "heuristic"
    detector: YoloJumpDetector | None = None
    if args.detector != "heuristic":
        try:
            detector = YoloJumpDetector(args.yolo_model, conf=args.yolo_conf, device=args.yolo_device)
            detector_name = "yolo"
        except Exception as exc:
            if args.detector == "yolo":
                raise SystemExit(f"failed to load YOLO detector: {exc}") from exc
            print(f"YOLO unavailable, fallback to heuristic detector: {exc}")
    print(f"jump bot: press_ms = {args.coef} x gap_px  (det={detector_name})")

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else None

    coef = args.coef
    jumps = 0
    frame_idx = 0
    idle_retries = 0
    last_jump_time: float | None = None
    while args.max_jumps <= 0 or jumps < args.max_jumps:
        img, arr, piece, target = capture_and_detect(args.serial, detector=detector)
        h, w, _ = arr.shape
        if piece is None:
            if last_jump_time is not None and time.time() - last_jump_time < args.ready_timeout:
                time.sleep(args.poll_delay)
                continue
            idle_retries += 1
            print(f"[{jumps}] piece not found, tap to (re)start")
            save_debug_image(img, debug_dir / f"{jumps:04d}_no_piece.png", None, None, "piece not found")
            if not args.dry_run:
                tap(args.serial, w // 2, int(h * 0.78))
                time.sleep(1.5 if idle_retries > 1 else 1.0)
            continue

        if target is not None:
            last_jump_time = None

        if target is None:
            if last_jump_time is not None and time.time() - last_jump_time < args.ready_timeout:
                time.sleep(args.poll_delay)
                continue
            print(f"[{jumps}] target not found, waiting")
            save_debug_image(
                img,
                debug_dir / f"{jumps:04d}_no_target.png",
                piece,
                None,
                "target not found",
            )
            time.sleep(0.8)
            continue

        idle_retries = 0
        # Empirical WeChat-Jump law: press time is linear in the pixel distance
        # from the piece foot to the target centre.  press_ms = coef * gap_px.
        gap_px = math.hypot(target.x - piece.x, target.y - piece.y)
        press_ms = int(round(coef * gap_px))

        text = (
            f"jump={jumps + 1} gap_px={gap_px:.1f} half_px={target.half_width_px:.1f} "
            f"coef={coef:.3f} det={detector_name} press={press_ms}ms"
        )
        print(text)
        save_raw_image(img, raw_dir, frame_idx)
        frame_idx += 1
        save_debug_image(img, debug_dir / f"{jumps:04d}.png", piece, target, text)

        if args.dry_run:
            break

        long_press(args.serial, w // 2, int(h * 0.75), press_ms)
        jumps += 1
        last_jump_time = time.time()
        time.sleep(args.min_air_time)


if __name__ == "__main__":
    main()
