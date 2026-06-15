#!/usr/bin/env python3
"""真机在线 DQN:边玩边学跳一跳的「距离 → 按压时间」映射。

和 `adb_jump_ppo.py`(纯线性 press_ms = coef×gap_px)不同,这里**直接在真手机上
做强化学习**:

    观测  obs   = [目标台框宽, 目标台框高, 棋子→目标中心距离](按屏宽归一,3 维)
    动作  action= 一档按压时间(离散 K 档 → press_ms)
    奖励  reward= 屏幕上的分数增量(读不到分数时退化为「存活 +1」)
    终止        = 分数归零 / 摔死(piece 检测不到)→ 惩罚,重开一局

每一步就是真机上的一次跳跃,所以 DQN 超参按「样本极少」调小(warmup/衰减都很短)。
检测复用 `adb_jump_ppo` 的 YOLO/规则检测;分数用 cv2 模板匹配读取(见 ScoreReader)。

用法:
    # 一次性标定分数区域 + 数字模板(见下方 ScoreReader 说明)
    .venv/bin/python adb_jump_dqn.py --serial <serial> --calibrate-score

    # 开练(真机)
    .venv/bin/python adb_jump_dqn.py --serial <serial> \\
        --yolo-model runs/detect/runs/jump_yolo_synth/weights/best.pt --episodes 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from adb_jump_ppo import (
    YoloJumpDetector,
    capture_and_detect,
    capture_screen,
    long_press,
    tap,
)
from rl_lab.algos import make_agent

ROOT = Path(__file__).resolve().parent
DIGIT_DIR = ROOT / "tools" / "score_digits"   # 0.png .. 9.png 数字模板


# --------------------------------------------------------------------------- #
# 屏幕分数读取(cv2 模板匹配,无需系统 OCR)
# --------------------------------------------------------------------------- #
class ScoreReader:
    """读取屏幕左上角的分数。

    标定一次(`--calibrate-score`):把分数区域裁出来,人工把各个数字裁成
    `tools/score_digits/0.png`…`9.png`(同一字体/字号)。之后每帧:裁区域 →
    二值化 → 按列切分数字 → 每个数字与 0-9 模板做归一化相关匹配 → 拼成整数。

    模板缺失时 `read()` 返回 None,训练循环自动退化为「存活计数」奖励。
    """

    def __init__(self, region_frac=(0.095, 0.165, 0.455, 0.240), thresh=110):
        # region_frac: 分数区域占全屏的比例 (x0, y0, x1, y1)
        self.region_frac = region_frac
        self.thresh = thresh
        self.templates = self._load_templates()

    @staticmethod
    def _norm(gray: np.ndarray):
        """统一成黑底白字、裁到数字紧致 bbox、缩放 24×36 的二值图。

        模板和待匹配 cell 都过这一步,保证可比(Otsu 自适应阈值 + 极性归一)。"""
        import cv2
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if b.mean() > 127:          # 前景应是少数 → 若白多则反相
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
                    n = self._norm(img) if img is not None else None
                    if n is not None:
                        tpl[d] = n
        return tpl

    @property
    def ready(self) -> bool:
        return len(self.templates) == 10

    def _crop(self, arr: np.ndarray) -> np.ndarray:
        h, w, _ = arr.shape
        x0, y0, x1, y1 = self.region_frac
        return arr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]

    def _digit_boxes(self, mask: np.ndarray):
        """按列投影把数字分开,返回从左到右的 (x0, x1) 列区间。"""
        cols = mask.any(axis=0)
        boxes, run = [], None
        for x, on in enumerate(cols.tolist() + [False]):
            if on and run is None:
                run = x
            elif not on and run is not None:
                if x - run >= 4:          # 太窄的当噪声
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
            cell = self._norm(gray[:, x0:x1])      # 同模板一致的归一化
            if cell is None:
                return None
            best_d, best_s = None, -1.0
            for d, tpl in self.templates.items():
                s = float(cv2.matchTemplate(cell, tpl, cv2.TM_CCOEFF_NORMED).max())
                if s > best_s:
                    best_s, best_d = s, d
            if best_s < 0.3:              # 匹配太差,放弃这帧
                return None
            digits.append(str(best_d))
        try:
            return int("".join(digits))
        except ValueError:
            return None

    def calibrate(self, arr: np.ndarray, out_dir: Path):
        """裁出分数区域和切分后的数字,供人工标注成 0.png..9.png。"""
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


# --------------------------------------------------------------------------- #
# 把真手机封装成一个 reset/step 的环境
# --------------------------------------------------------------------------- #
class RealJumpEnv:
    def __init__(self, serial, detector, score_reader,
                 press_min=200, press_max=1300, levels=15,
                 air_time=0.9, death_penalty=3.0, max_steps=500):
        self.serial = serial
        self.detector = detector
        self.score_reader = score_reader
        self.press_min, self.press_max = press_min, press_max
        self.levels = levels
        self.air_time = air_time
        self.death_penalty = death_penalty
        self.max_steps = max_steps
        self.obs_dim = 3                 # [目标框宽, 目标框高, 到目标中心距离],均按屏宽归一
        self.n_actions = levels
        self.prev_score = 0
        self.W = self.H = None

    # 动作档位 → 按压毫秒
    def action_to_press(self, action: int) -> int:
        if self.levels == 1:
            return self.press_min
        return int(round(self.press_min + action / (self.levels - 1) * (self.press_max - self.press_min)))

    # 线性经验公式 press≈coef×距离 → 最接近的动作档(暖启动探索的中心)
    def linear_action(self, dist_px: float, coef: float) -> int:
        press = coef * dist_px
        frac = (press - self.press_min) / max(1, self.press_max - self.press_min)
        return int(round(float(np.clip(frac, 0.0, 1.0)) * (self.levels - 1)))

    def _detect(self, retries=3, delay=0.18):
        """多试几次拿到 (piece, target),容忍偶发漏检。"""
        last_piece = last_target = None
        for _ in range(retries):
            _, arr, piece, target = capture_and_detect(self.serial, detector=self.detector)
            self.H, self.W, _ = arr.shape
            if piece is not None:
                last_piece, last_target = piece, target
                if target is not None:
                    return arr, piece, target
            time.sleep(delay)
        return None, last_piece, last_target

    def _obs(self, piece, target) -> np.ndarray:
        x0, y0, x1, y1 = target.bbox
        box_w = (x1 - x0) / self.W
        box_h = (y1 - y0) / self.W            # 同按屏宽归一,宽高同尺度
        dist = float(np.hypot(target.x - piece.x, target.y - piece.y)) / self.W
        return np.array([box_w, box_h, dist], dtype=np.float32)

    def reset(self) -> np.ndarray | None:
        """确保进入可玩状态(死了就点重开),返回首个观测。"""
        for attempt in range(6):
            arr, piece, target = self._detect()
            if piece is not None and target is not None:
                self.prev_score = self._read_score(arr, default=0)
                return self._obs(piece, target)
            # 没棋子 = 在结算/开始页 → 点屏幕重开
            w = self.W or 1080
            h = self.H or 2400
            tap(self.serial, w // 2, int(h * 0.78))
            time.sleep(1.4 if attempt else 1.0)
        return None

    def _read_score(self, arr, default):
        s = self.score_reader.read(arr) if self.score_reader else None
        return s if s is not None else default

    def step(self, action: int):
        press_ms = self.action_to_press(action)
        long_press(self.serial, self.W // 2, int(self.H * 0.75), press_ms)
        time.sleep(self.air_time)

        arr, piece, target = self._detect()
        # 摔死判定:棋子检测不到,或屏幕分数归零(比上一次小)
        cur_score = self._read_score(arr, default=None) if arr is not None else None
        died = piece is None or (cur_score is not None and cur_score <= self.prev_score)
        info = {"press_ms": press_ms, "cur_score": cur_score}
        if died:
            return np.zeros(self.obs_dim, np.float32), -self.death_penalty, True, False, {**info, "died": True}

        # 存活:奖励 = 屏幕分数增量;读不到分数则记 +1(存活计数)
        if self.score_reader and self.score_reader.ready and cur_score is not None:
            reward = float(max(0, cur_score - self.prev_score))
            self.prev_score = cur_score
        else:
            reward = 1.0
            self.prev_score += 1
        info["reward"] = reward

        if target is None:                       # 还活着但下一块台子还没出现
            return np.zeros(self.obs_dim, np.float32), reward, False, True, {**info, "no_target": True}
        return self._obs(piece, target), reward, False, False, {**info, "score": self.prev_score}


# --------------------------------------------------------------------------- #
# 训练循环
# --------------------------------------------------------------------------- #
def train(env: RealJumpEnv, agent, episodes: int, out_dir: Path, save_every: int,
          warmup_coef: float = 1.35, warmup_window: int = 2, seed: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    best_return = float("-inf")
    best_score = 0
    bar = "─" * 56

    def _save(name):                              # 连 ε 进度(steps/updates)一起存
        agent.save(out_dir / name, env_name="jump_real",
                   extra={"steps": getattr(agent, "steps", 0),
                          "updates": getattr(agent, "updates", 0)})

    def _is_restart_noise(episode_buf) -> bool:
        """首跳即死通常是重开/落点未就绪的脏样本,整局不进 replay。"""
        if len(episode_buf) != 1:
            return False
        _, _, reward, _, terminated, truncated, info = episode_buf[0]
        return bool(terminated and not truncated and info.get("died") and reward <= 0)

    mode = (f"暖启动探索(coef={warmup_coef}±{warmup_window}档)"
            if warmup_coef > 0 else "纯随机探索")
    print(f"\n{bar}\n  真机在线 DQN · 跳一跳   →  {out_dir}\n  {mode}\n{bar}")
    for ep in range(1, episodes + 1):
        obs = env.reset()
        if obs is None:
            print("⚠️  进不去游戏(检测不到棋子),检查 adb / 游戏是否在前台。")
            return
        ep_return, steps = 0.0, 0
        episode_buf = []
        while True:
            dist_px = float(obs[2]) * (env.W or 1080)   # obs = [框宽, 框高, 距离]
            # 自己做 ε 分支:便于标注「探索/利用」,也保证 ε 可续训。
            #   探索:暖启动时取线性估计档 ±window;否则全档随机。利用:DQN 贪心。
            explore = rng.random() < getattr(agent, "epsilon", 0.05)
            if explore and warmup_coef > 0:
                base = env.linear_action(dist_px, warmup_coef)
                action = int(np.clip(base + rng.integers(-warmup_window, warmup_window + 1),
                                     0, env.n_actions - 1))
            elif explore:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(agent.act(obs, deterministic=True))
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_buf.append((obs, action, reward, next_obs, terminated, truncated, info))
            obs = next_obs
            ep_return += reward
            steps += 1
            src = "🎲 探索" if explore else "🧠 利用"
            mark = "💀 摔死" if info.get("died") else f"✅ +{reward:.0f}"
            print(f"     跳{steps:<3d}  距 {dist_px:4.0f}px  按 {info['press_ms']:4d}ms  {src}  {mark}")
            if truncated and not terminated:     # 没台子等一下重新观测,不算死
                got = env.reset()
                if got is None:
                    break
                obs = got
                continue
            if terminated or steps >= env.max_steps:
                break
        dropped = _is_restart_noise(episode_buf)
        if not dropped:
            for step_obs, step_action, step_reward, step_next_obs, step_terminated, step_truncated, _ in episode_buf:
                agent.observe(step_obs, step_action, step_reward,
                              step_next_obs, step_terminated, step_truncated)
                agent.update()
            best_score = max(best_score, env.prev_score)
            new_best = ep_return > best_return
            if new_best:
                best_return = ep_return
                _save("best.pt")
            flag = "🏆" if new_best else "  "
        else:
            new_best = False
            flag = "⚠️"
        suffix = " │ 丢弃(疑似重开噪声,不入训练集)" if dropped else ""
        print(f"{flag} 第 {ep:<5d} 局 │ 步 {steps:<3d} │ 本局分 {env.prev_score:<3d} │ "
              f"回报 {ep_return:6.1f} │ ε {getattr(agent, 'epsilon', 0):.2f} │ 最高分 {best_score}{suffix}")
        if ep % save_every == 0:
            _save("latest.pt")
    _save("latest.pt")
    print(f"{bar}\n  训练结束 → {out_dir}\n{bar}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", help="adb serial;只接一台时可省")
    p.add_argument("--yolo-model", default="runs/detect/runs/jump_yolo_synth/weights/best.pt")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-device", default="cpu")
    p.add_argument("--detector", choices=["yolo", "heuristic"], default="yolo")
    # 动作空间:按压时间档位
    p.add_argument("--press-min", type=int, default=200)
    p.add_argument("--press-max", type=int, default=1300)
    p.add_argument("--levels", type=int, default=15, help="按压时间离散档数 = 动作数")
    # 真机节奏 / 奖励
    p.add_argument("--air-time", type=float, default=0.9, help="按下后等落地再看屏幕的秒数")
    p.add_argument("--death-penalty", type=float, default=3.0)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=500)
    # 暖启动:探索时围绕线性估计 press≈coef×距离 取档(设 0 = 纯随机探索)
    p.add_argument("--warmup-coef", type=float, default=1.35, help="暖启动探索的线性系数,0=纯随机")
    p.add_argument("--warmup-window", type=int, default=2, help="围绕线性估计探索的档位半径")
    # DQN(真机样本少 → 全部调小)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eps-decay-steps", type=int, default=800)
    # 产物 / 续训
    p.add_argument("--out-dir", default="runs/jump_real_dqn")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=5)
    # 分数标定
    p.add_argument("--calibrate-score", action="store_true", help="裁分数区域和数字供人工标注后退出")
    p.add_argument("--score-region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   default=[0.095, 0.165, 0.455, 0.240], help="分数区域占全屏比例(默认按 Pixel 8 标定)")
    args = p.parse_args()

    reader = ScoreReader(region_frac=tuple(args.score_region))

    if args.calibrate_score:
        arr = np.array(capture_screen(args.serial))
        reader.calibrate(arr, DIGIT_DIR)
        return

    detector = None
    if args.detector == "yolo":
        detector = YoloJumpDetector(args.yolo_model, conf=args.yolo_conf, device=args.yolo_device)
    print(f"分数读取:{'模板就绪' if reader.ready else '未标定 → 退化为存活计数奖励'}")

    env = RealJumpEnv(
        args.serial, detector, reader,
        press_min=args.press_min, press_max=args.press_max, levels=args.levels,
        air_time=args.air_time, death_penalty=args.death_penalty, max_steps=args.max_steps,
    )
    agent = make_agent(
        "dqn", obs_dim=env.obs_dim, n_actions=env.n_actions,
        lr=args.lr, warmup=args.warmup, batch_size=args.batch_size,
        eps_decay_steps=args.eps_decay_steps, buffer_size=10_000, target_sync=100,
    )
    out_dir = Path(args.out_dir)
    if args.resume and (out_dir / "latest.pt").exists():
        from rl_lab.algos.base import BaseAgent
        sd = BaseAgent.load_checkpoint(out_dir / "latest.pt")
        agent.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        extra = sd.get("extra", {}) if isinstance(sd, dict) else {}
        agent.steps = int(extra.get("steps", 0))         # 续上 ε 衰减进度
        agent.updates = int(extra.get("updates", 0))
        print(f"从 {out_dir/'latest.pt'} 续训(step={agent.steps}, ε={getattr(agent,'epsilon',0):.2f})")

    train(env, agent, args.episodes, out_dir, args.save_every,
          warmup_coef=args.warmup_coef, warmup_window=args.warmup_window)


if __name__ == "__main__":
    main()
