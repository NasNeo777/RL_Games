#!/usr/bin/env python3
"""Detect the piece/target first, then let the trained PPO checkpoint play Jump over adb."""

from __future__ import annotations

import argparse
import io
import math
import subprocess
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rl_lab.base_agent_loader import load_agent_for_demo
from rl_lab.envs.jump import (
    D_MAX,
    D_MIN,
    GAP_MAX,
    GAP_MIN,
    HALF_MAX,
    HALF_MIN,
    JumpPixelsEnv,
)


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


@dataclass
class JumpRecord:
    piece: Piece
    target: Target
    gap_px: float
    press_ms: int


@dataclass
class Kalman1D:
    x: float
    p: float = 1.0
    q: float = 1.5
    r: float = 16.0

    def update(self, z: float) -> float:
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


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


def estimate_world_state(gap_px: float, half_width_px: float, screen_w: int) -> tuple[float, float]:
    gap_world = float(
        np.clip(
            np.interp(gap_px, [screen_w * 0.12, screen_w * 0.48], [GAP_MIN, GAP_MAX]),
            GAP_MIN,
            GAP_MAX,
        )
    )
    half_world = float(
        np.clip(
            np.interp(half_width_px, [screen_w * 0.035, screen_w * 0.16], [HALF_MIN, HALF_MAX]),
            HALF_MIN,
            HALF_MAX,
        )
    )
    return gap_world, half_world


def synthetic_obs(gap_px: float, half_width_px: float, screen_w: int) -> tuple[np.ndarray, float, float]:
    gap_world, half_world = estimate_world_state(gap_px, half_width_px, screen_w)
    env = JumpPixelsEnv()
    env.a = env.b = 0.0
    env.cur_half = 0.4
    env.next_a = gap_world
    env.next_b = 0.0
    env.next_half = half_world
    return env._obs(), gap_world, half_world


def vector_obs(gap_px: float, half_width_px: float, screen_w: int) -> tuple[np.ndarray, float, float]:
    gap_world, half_world = estimate_world_state(gap_px, half_width_px, screen_w)
    gap_mid, gap_amp = (GAP_MIN + GAP_MAX) / 2, (GAP_MAX - GAP_MIN) / 2
    half_mid, half_amp = (HALF_MIN + HALF_MAX) / 2, (HALF_MAX - HALF_MIN) / 2
    obs = np.array([
        (gap_world - gap_mid) / gap_amp,
        (half_world - half_mid) / half_amp,
    ], dtype=np.float32)
    return obs, gap_world, half_world


def action_to_distance(action: int) -> float:
    return D_MIN + action / 40.0 * (D_MAX - D_MIN)


def update_coef_from_landing(coef: float, prev: JumpRecord, current_piece: Piece) -> tuple[float, str | None]:
    start = np.array([prev.piece.x, prev.piece.y], dtype=np.float32)
    target = np.array([prev.target.x, prev.target.y], dtype=np.float32)
    actual = np.array([current_piece.x, current_piece.y], dtype=np.float32)
    jump_vec = target - start
    jump_len = float(np.linalg.norm(jump_vec))
    if jump_len < 1e-6:
        return coef, None

    unit = jump_vec / jump_len
    delta = actual - start
    actual_proj = float(np.dot(delta, unit))
    actual_proj = float(np.clip(actual_proj, jump_len * 0.6, jump_len * 1.4))
    landing_err = float(np.linalg.norm(actual - target))
    perp_err = float(np.linalg.norm(delta - unit * actual_proj))
    # Only learn from landings that stayed on roughly the same jump ray.
    if perp_err > max(85.0, prev.target.half_width_px * 1.4):
        return coef, None
    if actual_proj < jump_len * 0.78 or actual_proj > jump_len * 1.22:
        return coef, None

    ratio = jump_len / max(actual_proj, 1e-6)
    ratio = float(np.clip(ratio, 0.94, 1.06))
    new_coef = float(np.clip(coef * (ratio ** 0.35), 1.18, 1.56))
    msg = (
        f"calib landing_err={landing_err:.1f}px perp_err={perp_err:.1f}px actual_proj={actual_proj:.1f}px "
        f"target_gap={jump_len:.1f}px coef {coef:.4f}->{new_coef:.4f}"
    )
    return new_coef, msg


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


def stable_detect(
    serial: str | None,
    captures: int,
    sample_delay: float,
    detector: YoloJumpDetector | None = None,
) -> tuple[Image.Image, np.ndarray, Piece | None, Target | None, int]:
    last_img = None
    last_arr = None
    last_piece = None
    last_target = None
    valid = 0
    fx = fy = tx = ty = th = None

    for i in range(max(1, captures)):
        img, arr, piece, target = capture_and_detect(serial, detector=detector)
        if piece is not None:
            last_img, last_arr, last_piece = img, arr, piece
        if piece is not None and target is not None:
            last_img, last_arr, last_piece, last_target = img, arr, piece, target
            if fx is None:
                fx = Kalman1D(piece.x, p=4.0, q=1.2, r=9.0)
                fy = Kalman1D(piece.y, p=4.0, q=1.2, r=9.0)
                tx = Kalman1D(target.x, p=9.0, q=2.0, r=16.0)
                ty = Kalman1D(target.y, p=9.0, q=2.0, r=16.0)
                th = Kalman1D(target.half_width_px, p=9.0, q=2.0, r=20.0)
            else:
                fx.update(piece.x)
                fy.update(piece.y)
                tx.update(target.x)
                ty.update(target.y)
                th.update(target.half_width_px)
            valid += 1
        if i + 1 < captures:
            time.sleep(sample_delay)

    if last_img is None or last_arr is None:
        img, arr, piece, target = capture_and_detect(serial, detector=detector)
        return img, arr, piece, target, 1 if (piece and target) else 0

    if valid >= 2 and last_piece is not None and last_target is not None:
        last_piece = replace(last_piece, x=float(fx.x), y=float(fy.x))
        last_target = replace(
            last_target,
            x=float(tx.x),
            y=float(ty.x),
            half_width_px=float(th.x),
        )
    return last_img, last_arr, last_piece, last_target, valid


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
    p.add_argument("--ckpt", default="runs/jump_ppo/best.pt")
    p.add_argument("--detector", choices=["auto", "heuristic", "yolo"], default="yolo")
    p.add_argument("--yolo-model", default="runs/detect/runs/jump_yolo_manual/weights/best.pt")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-device", default="cpu")
    p.add_argument("--coef", type=float, default=1.36, help="ms per pixel baseline")
    p.add_argument("--interval", type=float, default=0.2, help="fallback extra wait after an abnormal transition")
    p.add_argument("--max-jumps", type=int, default=0, help="0 means unlimited")
    p.add_argument("--captures", type=int, default=3, help="screenshots to fuse with Kalman before each jump")
    p.add_argument("--sample-delay", type=float, default=0.06, help="seconds between fused screenshots")
    p.add_argument("--min-air-time", type=float, default=0.34, help="minimum seconds to wait before checking whether the piece has landed")
    p.add_argument("--poll-delay", type=float, default=0.05, help="seconds between ready checks after a jump")
    p.add_argument("--ready-streak", type=int, default=1, help="consecutive stable detections required before the next jump")
    p.add_argument("--ready-timeout", type=float, default=1.0, help="seconds to wait for the next stable ready state")
    p.add_argument("--no-adapt", action="store_true", help="disable online coefficient calibration")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug-dir", default="debug_jump_ppo")
    p.add_argument("--raw-dir", help="save unannotated screenshots for later labeling")
    args = p.parse_args()

    _, agent, ckpt = load_agent_for_demo(args.ckpt)
    env_name = ckpt.get("env")
    if env_name == "jump_pixels":
        obs_mode = "image"
    else:
        obs_mode = "vector"

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
    print(f"loaded {args.ckpt} ({ckpt['env']} / {ckpt['algo']} / {obs_mode} / det={detector_name})")

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else None

    coef = args.coef
    jumps = 0
    frame_idx = 0
    idle_retries = 0
    prev_jump: JumpRecord | None = None
    last_jump_time: float | None = None
    while args.max_jumps <= 0 or jumps < args.max_jumps:
        img, arr, piece, target, valid_obs = stable_detect(
            args.serial,
            captures=args.captures,
            sample_delay=args.sample_delay,
            detector=detector,
        )
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

        if prev_jump is not None and not args.no_adapt:
            coef, calib_msg = update_coef_from_landing(coef, prev_jump, piece)
            if calib_msg:
                print(calib_msg)
            prev_jump = None
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
                f"target not found obs={valid_obs}/{max(1, args.captures)}",
            )
            time.sleep(0.8)
            continue

        idle_retries = 0
        gap_px = math.hypot(target.x - piece.x, target.y - piece.y)
        if obs_mode == "vector":
            obs, gap_world, half_world = vector_obs(gap_px, target.half_width_px, w)
        else:
            obs, gap_world, half_world = synthetic_obs(gap_px, target.half_width_px, w)
        action = int(agent.act(obs, deterministic=True))
        pred_dist = action_to_distance(action)
        scale = pred_dist / max(gap_world, 1e-6)
        press_ms = int(round(np.clip(coef * gap_px * scale, 220, 1250)))

        text = (
            f"jump={jumps + 1} gap_px={gap_px:.1f} half_px={target.half_width_px:.1f} "
            f"gap_w={gap_world:.2f} half_w={half_world:.2f} action={action} "
            f"pred={pred_dist:.2f} scale={scale:.3f} coef={coef:.4f} mode={obs_mode} det={detector_name} "
            f"obs={valid_obs}/{max(1, args.captures)} press={press_ms}ms"
        )
        print(text)
        save_raw_image(img, raw_dir, frame_idx)
        frame_idx += 1
        save_debug_image(img, debug_dir / f"{jumps:04d}.png", piece, target, text)

        if args.dry_run:
            break

        long_press(args.serial, w // 2, int(h * 0.75), press_ms)
        prev_jump = JumpRecord(piece=piece, target=target, gap_px=gap_px, press_ms=press_ms)
        jumps += 1
        last_jump_time = time.time()
        time.sleep(args.min_air_time)


if __name__ == "__main__":
    main()
