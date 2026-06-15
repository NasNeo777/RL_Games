#!/usr/bin/env python3
"""Use the trained jump_pixels_ppo checkpoint to play Jump on an adb device."""

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


def grow_region(
    arr: np.ndarray,
    seed_x: int,
    seed_y: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color_tol: float = 34.0,
) -> np.ndarray:
    roi = arr[y0:y1, x0:x1]
    h, w, _ = roi.shape
    sx = max(0, min(w - 1, seed_x - x0))
    sy = max(0, min(h - 1, seed_y - y0))
    seed = roi[sy, sx].astype(np.int16)
    q = deque([(sy, sx)])
    visited = np.zeros((h, w), dtype=bool)
    visited[sy, sx] = True
    region: list[tuple[int, int]] = []
    while q:
        cy, cx = q.popleft()
        px = roi[cy, cx].astype(np.int16)
        if np.linalg.norm(px - seed) > color_tol:
            continue
        region.append((cy, cx))
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    if not region:
        return np.empty((0, 2), dtype=np.int32)
    pts = np.array(region, dtype=np.int32)
    pts[:, 0] += y0
    pts[:, 1] += x0
    return pts


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
    for y in range(int(h * 0.30), int(h * 0.62)):
        runs = row_runs(mask[y], min_len=14, max_len=int(w * 0.32))
        runs = [
            (sx, ex)
            for sx, ex in runs
            if sx > 10 and ex < w - 10 and abs((sx + ex) / 2 - piece.x) > 75
        ]
        if runs:
            sx, ex = runs[0]
            run_seed = (y, sx, ex)
            break
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
    candidates = [(min_y + dy, seed_x) for dy in range(0, 16)]
    best_seed = None
    best_score = -1.0
    for rel_y, sx_abs in candidates:
        y_abs = ry0 + rel_y
        if y_abs >= h:
            continue
        x_abs = max(rx0, min(rx1 - 1, sx_abs))
        px = arr[y_abs, x_abs].astype(np.int16)
        score = float(np.linalg.norm(px - bg_rows[y_abs]))
        if score > best_score:
            best_score = score
            best_seed = (x_abs, y_abs)
    if best_seed is None:
        return None

    region = grow_region(arr, best_seed[0], best_seed[1], rx0, ry0, rx1, ry1)
    if len(region) < 50:
        return None
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


def synthetic_obs(gap_px: float, half_width_px: float, screen_w: int) -> tuple[np.ndarray, float, float]:
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
    env = JumpPixelsEnv()
    env.a = env.b = 0.0
    env.cur_half = 0.4
    env.next_a = gap_world
    env.next_b = 0.0
    env.next_half = half_world
    return env._obs(), gap_world, half_world


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", help="adb serial; default uses the only attached device")
    p.add_argument("--ckpt", default="runs/jump_pixels_ppo/best.pt")
    p.add_argument("--coef", type=float, default=1.36, help="ms per pixel baseline")
    p.add_argument("--interval", type=float, default=2.05, help="seconds to wait after each jump")
    p.add_argument("--max-jumps", type=int, default=0, help="0 means unlimited")
    p.add_argument("--no-adapt", action="store_true", help="disable online coefficient calibration")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug-dir", default="debug_jump_ppo")
    args = p.parse_args()

    _, agent, ckpt = load_agent_for_demo(args.ckpt)
    print(f"loaded {args.ckpt} ({ckpt['env']} / {ckpt['algo']})")

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    coef = args.coef
    jumps = 0
    idle_retries = 0
    prev_jump: JumpRecord | None = None
    while args.max_jumps <= 0 or jumps < args.max_jumps:
        img = capture_screen(args.serial)
        arr = np.array(img)
        h, w, _ = arr.shape
        piece = find_piece(arr)
        if piece is None:
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

        target = find_target(arr, piece)
        if target is None:
            print(f"[{jumps}] target not found, waiting")
            save_debug_image(img, debug_dir / f"{jumps:04d}_no_target.png", piece, None, "target not found")
            time.sleep(0.8)
            continue

        idle_retries = 0
        gap_px = math.hypot(target.x - piece.x, target.y - piece.y)
        obs, gap_world, half_world = synthetic_obs(gap_px, target.half_width_px, w)
        action = int(agent.act(obs, deterministic=True))
        pred_dist = action_to_distance(action)
        scale = pred_dist / max(gap_world, 1e-6)
        press_ms = int(round(np.clip(coef * gap_px * scale, 220, 1250)))

        text = (
            f"jump={jumps + 1} gap_px={gap_px:.1f} half_px={target.half_width_px:.1f} "
            f"gap_w={gap_world:.2f} half_w={half_world:.2f} action={action} "
            f"pred={pred_dist:.2f} scale={scale:.3f} coef={coef:.4f} press={press_ms}ms"
        )
        print(text)
        save_debug_image(img, debug_dir / f"{jumps:04d}.png", piece, target, text)

        if args.dry_run:
            break

        long_press(args.serial, w // 2, int(h * 0.75), press_ms)
        prev_jump = JumpRecord(piece=piece, target=target, gap_px=gap_px, press_ms=press_ms)
        jumps += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
