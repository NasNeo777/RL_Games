#!/usr/bin/env python3
"""真机跳一跳:纯线性基线 + 在线自动微调 coef。

这版不再用 DQN 学离散动作,而是直接保留微信跳一跳常用的线性规律:

    press_ms = coef * gap_px

然后在真机上把 `coef` 当成待调超参,按「整局表现」在线挑选更好的候选值。
旧的 DQN 版本已保留为 `adb_jump_dqn_legacy.py`,这里不会覆盖它。

用法:
    # 一次性标定分数模板
    .venv/bin/python adb_jump_autocoef.py --serial <serial> --calibrate-score

    # 真机自动调 coef
    .venv/bin/python adb_jump_autocoef.py --serial <serial> \\
        --yolo-model runs/detect/runs/jump_yolo_synth/weights/best.pt \\
        --episodes 300
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from adb_jump_ppo import (
    YoloJumpDetector,
    capture_and_detect,
    capture_screen,
    long_press,
    tap,
    wait_until_ready,
)

ROOT = Path(__file__).resolve().parent
DIGIT_DIR = ROOT / "tools" / "score_digits"


class ScoreReader:
    """读取屏幕左上角分数,模板缺失时自动退化。"""

    def __init__(self, region_frac=(0.095, 0.165, 0.455, 0.240), thresh=110):
        self.region_frac = region_frac
        self.thresh = thresh
        self.templates = self._load_templates()

    @staticmethod
    def _norm(gray: np.ndarray):
        import cv2

        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if b.mean() > 127:
            b = 255 - b
        ys, xs = np.where(b > 0)
        if len(xs) == 0:
            return None
        b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return cv2.resize(b, (24, 36))

    def _load_templates(self):
        import cv2

        tpl = {}
        if DIGIT_DIR.is_dir():
            for d in range(10):
                p = DIGIT_DIR / f"{d}.png"
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    norm = self._norm(img) if img is not None else None
                    if norm is not None:
                        tpl[d] = norm
        return tpl

    @property
    def ready(self) -> bool:
        return len(self.templates) == 10

    def _crop(self, arr: np.ndarray) -> np.ndarray:
        h, w, _ = arr.shape
        x0, y0, x1, y1 = self.region_frac
        return arr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]

    def _digit_boxes(self, mask: np.ndarray):
        cols = mask.any(axis=0)
        boxes, run = [], None
        for x, on in enumerate(cols.tolist() + [False]):
            if on and run is None:
                run = x
            elif not on and run is not None:
                if x - run >= 4:
                    boxes.append((run, x))
                run = None
        return boxes

    def read(self, arr: np.ndarray) -> int | None:
        if not self.ready:
            return None
        import cv2

        crop = self._crop(arr)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        mask = (gray < self.thresh).astype(np.uint8)
        boxes = self._digit_boxes(mask)
        if not boxes:
            return None
        digits = []
        for x0, x1 in boxes:
            cell = self._norm(gray[:, x0:x1])
            if cell is None:
                return None
            best_d, best_s = None, -1.0
            for d, tpl in self.templates.items():
                s = float(cv2.matchTemplate(cell, tpl, cv2.TM_CCOEFF_NORMED).max())
                if s > best_s:
                    best_s, best_d = s, d
            if best_s < 0.3:
                return None
            digits.append(str(best_d))
        try:
            return int("".join(digits))
        except ValueError:
            return None

    def calibrate(self, arr: np.ndarray, out_dir: Path):
        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)
        crop = self._crop(arr)
        cv2.imwrite(str(out_dir / "_region.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        mask = (gray < self.thresh).astype(np.uint8)
        for i, (x0, x1) in enumerate(self._digit_boxes(mask)):
            cv2.imwrite(str(out_dir / f"_digit_{i}.png"), mask[:, x0:x1] * 255)
        print(f"已写出 {out_dir}/_region.png 和切分数字 _digit_*.png")
        print("把每个数字裁好存成 0.png..9.png(同字号),即可启用屏幕分数奖励。")


class CoefBandit:
    """在一小段 coef 网格里做 UCB 选臂,必要时向边界外平移搜索。"""

    def __init__(self, center: float, step: float, radius: int,
                 coef_min: float, coef_max: float, explore_bonus: float,
                 counts=None, values=None, total_updates: int = 0):
        self.center = float(center)
        self.step = float(step)
        self.radius = int(radius)
        self.coef_min = float(coef_min)
        self.coef_max = float(coef_max)
        self.explore_bonus = float(explore_bonus)
        self.total_updates = int(total_updates)
        self.candidates = []
        self.counts = []
        self.values = []
        self._rebuild_grid(counts=counts, values=values)

    def _build_candidates(self, center: float):
        vals = []
        for k in range(-self.radius, self.radius + 1):
            coef = round(center + k * self.step, 4)
            if self.coef_min <= coef <= self.coef_max:
                vals.append(coef)
        if not vals:
            vals = [round(min(max(center, self.coef_min), self.coef_max), 4)]
        return vals

    def _rebuild_grid(self, counts=None, values=None):
        old = {}
        if self.candidates and self.counts and self.values:
            old = {
                round(c, 4): (self.counts[i], self.values[i])
                for i, c in enumerate(self.candidates)
            }
        if counts is not None and values is not None and not old:
            old = {
                round(c, 4): (int(counts[i]), float(values[i]))
                for i, c in enumerate(self._build_candidates(self.center))
                if i < len(counts) and i < len(values)
            }
        self.candidates = self._build_candidates(self.center)
        self.counts = []
        self.values = []
        for coef in self.candidates:
            count, value = old.get(round(coef, 4), (0, 0.0))
            self.counts.append(int(count))
            self.values.append(float(value))

    def choose(self) -> int:
        pending = [i for i, c in enumerate(self.counts) if c == 0]
        if pending:
            pending.sort(key=lambda i: (abs(self.candidates[i] - self.center), self.candidates[i]))
            return pending[0]
        log_total = math.log(self.total_updates + 1.0)
        scores = []
        for i in range(len(self.candidates)):
            bonus = self.explore_bonus * math.sqrt(log_total / self.counts[i])
            scores.append(self.values[i] + bonus)
        return int(np.argmax(scores))

    def update(self, idx: int, reward: float):
        self.total_updates += 1
        self.counts[idx] += 1
        self.values[idx] += (reward - self.values[idx]) / self.counts[idx]
        self._maybe_shift_grid()

    def _maybe_shift_grid(self):
        seen = [i for i, c in enumerate(self.counts) if c > 0]
        if not seen:
            return
        best_idx = max(seen, key=lambda i: self.values[i])
        if 0 < best_idx < len(self.candidates) - 1:
            return
        new_center = self.candidates[best_idx]
        if abs(new_center - self.center) < self.step * 0.5:
            return
        self.center = new_center
        self._rebuild_grid()

    def best(self):
        seen = [i for i, c in enumerate(self.counts) if c > 0]
        if not seen:
            return self.center, 0.0
        best_idx = max(seen, key=lambda i: self.values[i])
        return self.candidates[best_idx], self.values[best_idx]

    def state_dict(self):
        return {
            "center": self.center,
            "step": self.step,
            "radius": self.radius,
            "coef_min": self.coef_min,
            "coef_max": self.coef_max,
            "explore_bonus": self.explore_bonus,
            "candidates": self.candidates,
            "counts": self.counts,
            "values": self.values,
            "total_updates": self.total_updates,
        }

    @classmethod
    def from_state_dict(cls, sd):
        bandit = cls(
            center=sd["center"],
            step=sd["step"],
            radius=sd["radius"],
            coef_min=sd["coef_min"],
            coef_max=sd["coef_max"],
            explore_bonus=sd["explore_bonus"],
            total_updates=sd.get("total_updates", 0),
        )
        bandit.candidates = [float(x) for x in sd["candidates"]]
        bandit.counts = [int(x) for x in sd["counts"]]
        bandit.values = [float(x) for x in sd["values"]]
        return bandit


def detect_state(serial, detector, retries=3, delay=0.12):
    last = None
    for _ in range(retries):
        img, arr, piece, target = capture_and_detect(serial, detector=detector)
        last = (img, arr, piece, target)
        if piece is not None and target is not None:
            return last
        time.sleep(delay)
    return last


def ensure_playable(serial, detector, reader, max_taps=6):
    for attempt in range(max_taps):
        img, arr, piece, target = detect_state(serial, detector)
        h, w, _ = arr.shape
        if piece is not None and target is not None:
            score = reader.read(arr) if reader.ready else None
            return {
                "img": img,
                "arr": arr,
                "piece": piece,
                "target": target,
                "score": score if score is not None else 0,
                "w": w,
                "h": h,
            }
        tap(serial, w // 2, int(h * 0.78))
        time.sleep(1.4 if attempt else 1.0)
    return None


def read_score(reader, arr, default):
    score = reader.read(arr) if reader.ready else None
    return score if score is not None else default


def clamp_press_ms(gap_px: float, coef: float, press_min: int, press_max: int) -> int:
    return int(np.clip(round(coef * gap_px), press_min, press_max))


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_episode(args, detector, reader, coef: float):
    state = ensure_playable(args.serial, detector, reader)
    if state is None:
        raise SystemExit("进不去游戏(检测不到棋子/台子),检查 adb / 游戏是否在前台。")

    piece = state["piece"]
    target = state["target"]
    w = state["w"]
    h = state["h"]
    prev_score = state["score"]
    start_score = prev_score
    episode_return = 0.0
    steps = 0
    noisy_abort = False
    started_at = time.time()

    while steps < args.max_steps:
        gap_px = math.hypot(target.x - piece.x, target.y - piece.y)
        press_ms = clamp_press_ms(gap_px, coef, args.press_min, args.press_max)
        long_press(args.serial, w // 2, int(h * 0.75), press_ms)
        ready = wait_until_ready(
            args.serial,
            min_air_time=args.min_air_time,
            poll_delay=args.poll_delay,
            ready_streak=args.ready_streak,
            timeout=args.ready_timeout,
            detector=detector,
        )
        img, arr, piece, target = detect_state(args.serial, detector, retries=2, delay=args.poll_delay)
        h, w, _ = arr.shape
        cur_score = read_score(reader, arr, default=prev_score + 1)
        died = piece is None or (reader.ready and cur_score <= prev_score)
        reward = -args.death_penalty if died else float(max(1, cur_score - prev_score))
        episode_return += reward
        steps += 1
        mark = "💀 摔死" if died else f"✅ +{reward:.0f}"
        stable = "" if ready else "  (落地未完全稳定)"
        print(f"     跳{steps:<3d}  距 {gap_px:4.0f}px  按 {press_ms:4d}ms  coef {coef:.3f}  {mark}{stable}")
        if died:
            break
        prev_score = cur_score
        if target is None:
            extra = detect_state(args.serial, detector, retries=3, delay=args.poll_delay)
            _, arr, piece, target = extra
            if piece is None:
                episode_return -= args.death_penalty
                break
            if target is None:
                noisy_abort = True
                print("         目标台还没稳定识别出来,本局提前结束")
                break
        time.sleep(args.poll_delay)

    final_score = prev_score
    dropped = steps == 1 and (final_score <= start_score or noisy_abort)
    duration_s = time.time() - started_at
    return {
        "score": int(final_score),
        "return": float(episode_return),
        "steps": int(steps),
        "dropped": bool(dropped),
        "duration_s": round(duration_s, 2),
    }


def load_state(meta_path: Path):
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", help="adb serial;只接一台时可省")
    p.add_argument("--detector", choices=["auto", "heuristic", "yolo"], default="yolo")
    p.add_argument("--yolo-model", default="runs/detect/runs/jump_yolo_synth/weights/best.pt")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-device", default="cpu")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--press-min", type=int, default=200)
    p.add_argument("--press-max", type=int, default=1300)
    p.add_argument("--death-penalty", type=float, default=3.0)
    p.add_argument("--coef-start", type=float, default=1.375)
    p.add_argument("--coef-step", type=float, default=0.025)
    p.add_argument("--coef-radius", type=int, default=6)
    p.add_argument("--coef-min", type=float, default=1.00)
    p.add_argument("--coef-max", type=float, default=1.70)
    p.add_argument("--explore-bonus", type=float, default=1.4)
    p.add_argument("--min-air-time", type=float, default=0.34)
    p.add_argument("--poll-delay", type=float, default=0.05)
    p.add_argument("--ready-streak", type=int, default=2)
    p.add_argument("--ready-timeout", type=float, default=1.1)
    p.add_argument("--out-dir", default="runs/jump_auto_coef")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--calibrate-score", action="store_true")
    p.add_argument("--score-region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   default=[0.095, 0.165, 0.455, 0.240],
                   help="分数区域占全屏比例(默认按 Pixel 8 标定)")
    args = p.parse_args()

    reader = ScoreReader(region_frac=tuple(args.score_region))
    if args.calibrate_score:
        arr = np.array(capture_screen(args.serial))
        reader.calibrate(arr, DIGIT_DIR)
        return

    detector_name = "heuristic"
    detector = None
    if args.detector != "heuristic":
        try:
            detector = YoloJumpDetector(args.yolo_model, conf=args.yolo_conf, device=args.yolo_device)
            detector_name = "yolo"
        except Exception as exc:
            if args.detector == "yolo":
                raise SystemExit(f"failed to load YOLO detector: {exc}") from exc
            print(f"YOLO unavailable, fallback to heuristic detector: {exc}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    history_path = out_dir / "history.jsonl"
    episode0 = 0
    best_score = 0
    best_episode = 0
    best_coef = args.coef_start
    bandit = CoefBandit(
        center=args.coef_start,
        step=args.coef_step,
        radius=args.coef_radius,
        coef_min=args.coef_min,
        coef_max=args.coef_max,
        explore_bonus=args.explore_bonus,
    )
    if args.resume:
        saved = load_state(meta_path)
        if saved:
            episode0 = int(saved.get("episode", 0))
            best_score = int(saved.get("best_score", 0))
            best_episode = int(saved.get("best_episode", 0))
            best_coef = float(saved.get("best_coef", best_coef))
            bandit = CoefBandit.from_state_dict(saved["bandit"])

    bar = "─" * 56
    print(f"\n{bar}\n  真机线性调参 · 跳一跳   →  {out_dir}\n"
          f"  press_ms = coef × gap_px   (det={detector_name})\n{bar}")
    print(f"分数读取:{'模板就绪' if reader.ready else '未标定 → 退化为存活步数'}")
    print(f"备份保留: {ROOT / 'adb_jump_dqn_legacy.py'}")

    for ep in range(episode0 + 1, episode0 + args.episodes + 1):
        arm = bandit.choose()
        coef = bandit.candidates[arm]
        print(f"\n第 {ep:<5d} 局 │ 试 coef {coef:.3f} │ 候选均值 {bandit.values[arm]:5.1f} │ 已试 {bandit.counts[arm]} 次")
        result = run_episode(args, detector, reader, coef)
        score = result["score"]
        reward_metric = score if reader.ready else result["steps"]
        if result["dropped"]:
            flag = "⚠️"
            suffix = " │ 丢弃(疑似重开噪声,不更新 coef)"
        else:
            bandit.update(arm, reward_metric)
            flag = "🏆" if score > best_score else "  "
            suffix = ""
            if score > best_score:
                best_score = score
                best_episode = ep
                best_coef = coef
        reco_coef, reco_mean = bandit.best()
        print(f"{flag} 第 {ep:<5d} 局 │ 分数 {score:<4d} │ 步 {result['steps']:<4d} │ "
              f"回报 {result['return']:6.1f} │ 推荐 coef {reco_coef:.3f} "
              f"(均值 {reco_mean:.1f}){suffix}")

        rec = {
            "episode": ep,
            "coef": coef,
            "score": score,
            "steps": result["steps"],
            "return": result["return"],
            "dropped": result["dropped"],
            "duration_s": result["duration_s"],
            "best_score": best_score,
            "best_episode": best_episode,
            "best_coef": best_coef,
            "recommended_coef": reco_coef,
            "recommended_mean": reco_mean,
        }
        append_jsonl(history_path, rec)
        if ep % args.save_every == 0 or ep == episode0 + args.episodes:
            write_json(meta_path, {
                "episode": ep,
                "best_score": best_score,
                "best_episode": best_episode,
                "best_coef": best_coef,
                "bandit": bandit.state_dict(),
            })

    print(f"\n{bar}\n  调参结束 │ 历史最高 {best_score} 分 │ 最佳 coef {best_coef:.3f} (第 {best_episode} 局)\n{bar}")


if __name__ == "__main__":
    main()
