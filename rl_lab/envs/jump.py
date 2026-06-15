"""跳一跳 —— 纯 Python 自制,"蓄力跳台"游戏。

小人站在当前台子中心,下一块台子在前方随机距离处出现 —— 方向在两条
等距斜轴里随机选一条(**右前**或**左前**,像原版那样左右拐着走),
宽度、底座样式也都随机。每一步,agent 选一个跳跃力度(离散 N 档,
映射到一段跳跃距离),小人沿该方向腾空落下:
- 落在下一块台面上即成功,前进到该台子,刷新更前方的新台子;落点越靠
  台心给的奖励越高(还原原版"踩中心连击得高分"的手感)。
- 落到台子外面(摔进缝里)即失败,回合结束。

随机性来源:每一步的台距、台宽、方向、底座样式都重新随机。方向与样式
纯属画面表现(小人自动朝向目标台),**不影响决策** —— agent 要学的
始终是"看缺口大小调力度"这道带容差的回归题,观测也只给缺口与台宽。

观测 2 维(均归一化):
- 到下一块台子中心的距离(缺口大小)
- 下一块台子的半宽(容差大小)

奖励:落台 +1,踩得越靠中心额外 +0~2,正中靶心再 +1 连击奖励;
摔下去 -5。连续踩中 SUCCESS_SCORE 块台子视为通关,+50 并结束回合。
"""
import numpy as np

from .base import BaseEnv

# 台子中心间距(缺口)与台子半宽的随机范围(世界坐标单位)
GAP_MIN, GAP_MAX = 1.0, 3.0
HALF_MIN, HALF_MAX = 0.25, 0.55
# 跳跃距离档位:两端都留出余量,保证任意缺口都够得着也跳得过
D_MIN, D_MAX = 0.6, 3.4
N_LEVELS = 41                 # 力度档数 → 分辨率 (D_MAX-D_MIN)/40 ≈ 0.07
N_STYLES = 6                  # 底座样式数(仅供前端画图,需与 web 渲染器一致)

SUCCESS_SCORE = 25            # 连续踩中这么多块台子算通关
MISS_PENALTY = -5.0
PERFECT_FRAC = 0.2           # 落点误差 < 半宽的这个比例算"正中靶心"

# 两条前进斜轴的单位向量:0=右前(+a),1=左前(+b)。世界坐标记成 (a, b)。
DIRS = [(1.0, 0.0), (0.0, 1.0)]


class JumpEnv(BaseEnv):
    obs_dim = 2
    n_actions = N_LEVELS
    max_steps = 100

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.a = self.b = 0.0        # 当前所站台子中心的世界坐标
        self.cur_half = 0.0
        self.cur_style = 0
        self.next_a = self.next_b = 0.0
        self.next_half = 0.0
        self.next_style = 0
        self.next_dir = 0
        self.score = 0
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.a = self.b = 0.0
        self.cur_half = float(self.rng.uniform(HALF_MIN, HALF_MAX))
        self.cur_style = int(self.rng.integers(N_STYLES))
        self._spawn_next()
        self.score = 0
        self.t = 0
        self.frames = []
        if self.record:
            self._record_ready()
        return self._obs()

    def step(self, action):
        action = int(action)
        dist = D_MIN + action / (N_LEVELS - 1) * (D_MAX - D_MIN)
        gap = self._gap()
        err = abs(dist - gap)
        half = self.next_half
        landed = err <= half
        self.t += 1

        # 落点世界坐标:沿本次跳跃方向走 dist
        dx, dy = DIRS[self.next_dir]
        land = [round(self.a + dist * dx, 3), round(self.b + dist * dy, 3)]
        p0 = [round(self.a, 3), round(self.b, 3), round(self.cur_half, 3)]
        p1 = [round(self.next_a, 3), round(self.next_b, 3), round(self.next_half, 3)]
        s0, s1 = self.cur_style, self.next_style
        perfect = False
        success = False

        if landed:
            center = 1.0 - err / half             # 0(贴边)~ 1(正中)
            reward = 1.0 + 2.0 * center
            perfect = err < PERFECT_FRAC * half
            if perfect:
                reward += 1.0
            self.score += 1
            # 前进到刚踩中的台子,再刷新下一块
            self.a, self.b = self.next_a, self.next_b
            self.cur_half = self.next_half
            self.cur_style = self.next_style
            self._spawn_next()
            if self.score >= SUCCESS_SCORE:
                success = True
                reward += 50.0
        else:
            reward = MISS_PENALTY

        terminated = success or not landed
        truncated = (not terminated) and self.t >= self.max_steps
        info = {"success": success, "score": self.score}
        if self.record:
            self.frames.append({
                "p0": p0, "p1": p1, "land": land,
                "s0": s0, "s1": s1,
                "ok": bool(landed),
                "perfect": bool(perfect),
                "score": self.score,
            })
        return self._obs(), reward, terminated, truncated, info

    def _gap(self):
        return abs(self.next_a - self.a) + abs(self.next_b - self.b)

    def _spawn_next(self):
        gap = float(self.rng.uniform(GAP_MIN, GAP_MAX))
        self.next_dir = int(self.rng.integers(2))
        dx, dy = DIRS[self.next_dir]
        self.next_a = self.a + gap * dx
        self.next_b = self.b + gap * dy
        self.next_half = float(self.rng.uniform(HALF_MIN, HALF_MAX))
        self.next_style = int(self.rng.integers(N_STYLES))

    def _obs(self):
        gap = self._gap()
        gap_mid, gap_amp = (GAP_MIN + GAP_MAX) / 2, (GAP_MAX - GAP_MIN) / 2
        half_mid, half_amp = (HALF_MIN + HALF_MAX) / 2, (HALF_MAX - HALF_MIN) / 2
        return np.array([
            (gap - gap_mid) / gap_amp,
            (self.next_half - half_mid) / half_amp,
        ], dtype=np.float32)

    def _record_ready(self):
        # 开局静止帧:小人站在 0 号台中心,下一块台子已就位(p0==land 不起跳)
        self.frames.append({
            "p0": [round(self.a, 3), round(self.b, 3), round(self.cur_half, 3)],
            "p1": [round(self.next_a, 3), round(self.next_b, 3),
                   round(self.next_half, 3)],
            "land": [round(self.a, 3), round(self.b, 3)],
            "s0": self.cur_style, "s1": self.next_style,
            "ok": True,
            "perfect": False,
            "score": 0,
        })

    def render_spec(self):
        return {
            "type": "jump",
            "goal": SUCCESS_SCORE,
            "d_min": D_MIN, "d_max": D_MAX,
            "frame_dt": 0.6,
        }


# 渲染观测图的像素尺寸(CnnPolicy 的 NatureCNN 按 84×84 设计)
PIX = 84


class JumpPixelsEnv(JumpEnv):
    """跳一跳的**图像观测**版本:agent 不看 2 维数字,而是看一张 84×84
    的灰度示意图,用 PPO + CNN(CnnPolicy)从像素里自己读出缺口与台宽。

    决策逻辑、奖励、随机性与 ``JumpEnv`` 完全相同 —— 只把观测从向量换成
    图:当前台子(中等亮度)画在左侧固定位置,棋子在其上方,目标台子
    (最亮)按缺口距离画在右边、宽度按台宽画。网络要学的是"看右边那块
    亮条在哪、有多宽 → 该蓄多大力",本质还是带容差的回归,只是输入是像素。

    演示画面仍走父类的 2.5D 等距渲染(``render_spec`` 不变),好看照旧;
    CNN 只是 agent 内部的输入方式。DQN 不支持图像观测,本环境请用
    ``--algo ppo``(SB3 会自动切 CnnPolicy)。
    """
    obs_shape = (1, PIX, PIX)        # 单通道图像,SB3 走 CnnPolicy
    n_actions = N_LEVELS
    parallel_mode = "dummy"          # 单步极轻,同进程并行比子进程快一个数量级

    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.obs_dim = int(np.prod(self.obs_shape))
        self._yy, self._xx = np.mgrid[0:PIX, 0:PIX]
        self._style = {}
        self._sample_style()

    def reset(self, seed=None):
        self._sample_style()
        return super().reset(seed=seed)

    def _sample_style(self):
        self._style = {
            # Fixed-size grayscale input, but with screenshot-like camera jitter.
            "bg_top": float(self.rng.uniform(138, 188)),
            "bg_bottom": float(self.rng.uniform(156, 210)),
            "vignette": float(self.rng.uniform(0.02, 0.14)),
            "cam_zoom": float(self.rng.uniform(0.92, 1.08)),
            "cam_shift_x": float(self.rng.uniform(-0.18, 0.18)),
            "cam_shift_y": float(self.rng.uniform(-0.05, 0.06)),
            "anchor_x": float(self.rng.uniform(0.34, 0.62)),
            "ground": float(self.rng.uniform(0.58, 0.67)),
            "body_depth": int(self.rng.integers(int(PIX * 0.12), int(PIX * 0.20))),
            "body_drop": int(self.rng.integers(int(PIX * 0.08), int(PIX * 0.14))),
            "piece_h": int(self.rng.integers(int(PIX * 0.18), int(PIX * 0.24))),
            "piece_w": int(self.rng.integers(10, 14)),
            "piece_tone": float(self.rng.uniform(45, 92)),
            "piece_head": float(self.rng.uniform(72, 128)),
            "piece_variant": int(self.rng.integers(0, 4)),
            "piece_lean": float(self.rng.uniform(-0.10, 0.10)),
            "piece_head_scale": float(self.rng.uniform(0.74, 1.05)),
            "piece_waist": float(self.rng.uniform(0.28, 0.55)),
            "piece_shoulder": float(self.rng.uniform(0.52, 0.86)),
            "piece_gloss": float(self.rng.uniform(6, 18)),
            "piece_accent": float(self.rng.uniform(-10, 16)),
            "contrast": float(self.rng.uniform(0.90, 1.18)),
            "brightness": float(self.rng.uniform(-10, 10)),
            "noise_sigma": float(self.rng.uniform(1.5, 7.0)),
            "shadow_alpha": float(self.rng.uniform(0.08, 0.24)),
        }

    def _platform_style(self, side):
        base = float(self.rng.uniform(55, 205))
        top = float(np.clip(base + self.rng.uniform(18, 62), 38, 245))
        body = float(np.clip(base - self.rng.uniform(10, 42), 18, 220))
        shape = int((self.cur_style if side == "cur" else self.next_style) % 4)
        deco = int(self.rng.integers(0, 4))
        return {
            "shape": shape,
            "top": top,
            "body": body,
            "edge": float(np.clip(top - self.rng.uniform(10, 32), 40, 235)),
            "deco": deco,
        }

    def _world_to_px(self, x):
        span = D_MAX + HALF_MAX + 0.95
        ppu = self._style["cam_zoom"] * PIX / (2 * span)
        anchor = self._style["anchor_x"] * PIX
        return anchor + (x + self._style["cam_shift_x"]) * ppu

    def _top_mask(self, cx, cy, hw, hh, shape):
        xx, yy = self._xx, self._yy
        if shape == 0:
            return (np.abs(xx - cx) <= hw) & (np.abs(yy - cy) <= hh)
        if shape == 1:
            return ((xx - cx) / max(hw, 1)) ** 2 + ((yy - cy) / max(hh, 1)) ** 2 <= 1.0
        if shape == 2:
            # Rounded rectangle / pill.
            core = (np.abs(xx - cx) <= hw * 0.76) & (np.abs(yy - cy) <= hh)
            left = ((xx - (cx - hw * 0.76)) / max(hw * 0.34, 1)) ** 2 \
                + ((yy - cy) / max(hh, 1)) ** 2 <= 1.0
            right = ((xx - (cx + hw * 0.76)) / max(hw * 0.34, 1)) ** 2 \
                + ((yy - cy) / max(hh, 1)) ** 2 <= 1.0
            return core | left | right
        # Slightly faceted diamond-ish top.
        return (np.abs(xx - cx) / max(hw, 1) + np.abs(yy - cy) / max(hh, 1)) <= 1.0

    def _render_platform(self, img, cx, cy, half_w, style, facing):
        hw = max(7, int(round(half_w)))
        hh = max(4, int(round(hw * self.rng.uniform(0.33, 0.52))))
        depth = self._style["body_depth"]
        drop = self._style["body_drop"]
        top_mask = self._top_mask(cx, cy, hw, hh, style["shape"])

        body_shift = int(np.sign(facing) * max(6, hw * 0.30))
        body_mask = self._top_mask(cx + body_shift, cy + drop, hw, hh, style["shape"])
        body_mask &= self._yy >= cy
        shadow_mask = self._top_mask(cx - hw * 0.78, cy + drop, int(hw * 0.92), int(hh * 1.05), style["shape"])

        img[shadow_mask] = np.minimum(img[shadow_mask], img[shadow_mask] * (1.0 - self._style["shadow_alpha"]))
        img[body_mask] = style["body"]
        img[top_mask] = style["top"]

        edge_band = top_mask & (~self._top_mask(cx, cy + 1, max(hw - 1, 1), max(hh - 1, 1), style["shape"]))
        img[edge_band] = style["edge"]

        if style["deco"] == 0:
            deco = self._top_mask(cx, cy, int(hw * 0.45), int(hh * 0.45), 1)
            ring = deco & (~self._top_mask(cx, cy, int(hw * 0.18), int(hh * 0.18), 1))
            img[ring] = np.clip(style["edge"] + 18, 0, 255)
        elif style["deco"] == 1:
            stripe = top_mask & (np.abs(self._yy - cy) <= max(1, int(hh * 0.16)))
            img[stripe] = np.clip(style["edge"] + 10, 0, 255)
        elif style["deco"] == 2:
            dot = self._top_mask(cx, cy, max(1, int(hw * 0.16)), max(1, int(hh * 0.16)), 1)
            img[dot] = np.clip(style["top"] + 18, 0, 255)

    def _render_piece(self, img, cx, ground):
        tone = self._style["piece_tone"]
        head = self._style["piece_head"]
        body_h = self._style["piece_h"]
        body_w = self._style["piece_w"]
        variant = self._style["piece_variant"]
        base_y = ground - 3
        top_y = base_y - body_h
        xx, yy = self._xx, self._yy

        shadow = (((xx - (cx - body_w * 1.2)) / max(body_w * 1.15, 1)) ** 2
                  + ((yy - (base_y + body_w * 0.45)) / max(body_w * 0.52, 1)) ** 2) <= 1.0
        img[shadow] = np.minimum(img[shadow], img[shadow] * (1.0 - self._style["shadow_alpha"] * 1.4))

        t = np.clip((yy - top_y) / max(body_h, 1), 0, 1)
        spine = cx + self._style["piece_lean"] * body_h * (1.0 - t)
        if variant == 0:
            radius = body_w * (self._style["piece_waist"] + self._style["piece_shoulder"] * t)
        elif variant == 1:
            radius = body_w * (0.42 + 0.22 * np.sin(np.pi * np.clip(t, 0, 1)) + 0.26 * t)
        elif variant == 2:
            radius = body_w * (0.60 - 0.18 * np.abs(t - 0.42) + 0.22 * t)
        else:
            radius = body_w * (0.36 + 0.70 * (t ** 1.45))
        body = (yy >= top_y) & (yy <= base_y) & (np.abs(xx - spine) <= radius)
        shade = tone + self._style["piece_gloss"] * (1.0 - t)
        img[body] = shade[body]

        if variant in (1, 3):
            accent = body & (xx > spine + radius * 0.18)
            img[accent] = np.clip(img[accent] + self._style["piece_accent"], 0, 255)
        else:
            accent = body & (xx < spine - radius * 0.18)
            img[accent] = np.clip(img[accent] + self._style["piece_accent"] * 0.7, 0, 255)

        neck_w = body_w * (0.48 if variant == 2 else 0.62)
        neck_h = body_w * (0.48 if variant == 3 else 0.58)
        neck = (((xx - (cx + self._style["piece_lean"] * body_h * 0.95)) / max(neck_w, 1)) ** 2
                + ((yy - top_y) / max(neck_h, 1)) ** 2) <= 1.0
        img[neck] = np.clip(tone + 14 + self._style["piece_gloss"] * 0.4, 0, 255)

        head_scale = self._style["piece_head_scale"]
        head_cy = top_y - body_w * (0.60 + 0.12 * variant)
        head_rx = max(body_w * 0.60, body_w * 0.82 * head_scale)
        head_ry = max(body_w * 0.60, body_w * (0.72 + 0.08 * (variant == 3)) * head_scale)
        head_cx = cx + self._style["piece_lean"] * body_h * 0.92
        head_mask = (((xx - head_cx) / max(head_rx, 1)) ** 2
                     + ((yy - head_cy) / max(head_ry, 1)) ** 2) <= 1.0
        img[head_mask] = head

        if variant in (0, 2):
            cap = (((xx - head_cx) / max(head_rx * 0.48, 1)) ** 2
                   + ((yy - (head_cy - head_ry * 0.12)) / max(head_ry * 0.34, 1)) ** 2) <= 1.0
            img[cap] = np.clip(head + self._style["piece_accent"] * 0.6, 0, 255)
        else:
            band = head_mask & (np.abs(yy - head_cy) <= max(1, int(head_ry * 0.14)))
            img[band] = np.clip(head - 10 + self._style["piece_accent"] * 0.5, 0, 255)

    def _apply_post(self, img):
        # Grayscale stays grayscale; only style varies.
        cx = PIX * self.rng.uniform(0.35, 0.65)
        cy = PIX * self.rng.uniform(0.25, 0.55)
        dist = ((self._xx - cx) / PIX) ** 2 + ((self._yy - cy) / PIX) ** 2
        vignette = 1.0 - self._style["vignette"] * np.clip(dist * 3.4, 0, 1)
        img *= vignette

        if self._style["noise_sigma"] > 0:
            img += self.rng.normal(0.0, self._style["noise_sigma"], size=img.shape)

        # Mild streak / UI clutter lines to reduce overfitting to clean renders.
        if self.rng.random() < 0.35:
            band_y = int(self.rng.integers(0, PIX))
            band_h = int(self.rng.integers(1, 3))
            img[max(0, band_y - band_h):min(PIX, band_y + band_h + 1)] += self.rng.uniform(-12, 12)
        if self.rng.random() < 0.22:
            x0 = int(self.rng.integers(0, PIX - 8))
            img[: int(PIX * self.rng.uniform(0.12, 0.35)), x0:x0 + int(self.rng.integers(4, 12))] += self.rng.uniform(-10, 16)

        img = (img - 127.5) * self._style["contrast"] + 127.5 + self._style["brightness"]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _obs(self):
        img = np.linspace(self._style["bg_top"], self._style["bg_bottom"], PIX, dtype=np.float32)[:, None]
        img = np.repeat(img, PIX, axis=1)
        gap = self._gap()
        signed_gap = gap if self.next_dir == 0 else -gap
        ground = int(PIX * self._style["ground"] + self._style["cam_shift_y"] * PIX)

        cur_hw = self.cur_half * (PIX / (D_MAX + HALF_MAX + 1.4)) * 6.2
        nxt_hw = self.next_half * (PIX / (D_MAX + HALF_MAX + 1.4)) * 6.2
        cur_x = self._world_to_px(0.0)
        nxt_x = self._world_to_px(signed_gap)

        cur_style = self._platform_style("cur")
        nxt_style = self._platform_style("next")
        facing = 1 if signed_gap >= 0 else -1
        self._render_platform(img, cur_x, ground, cur_hw, cur_style, facing=-facing)
        self._render_platform(img, nxt_x, ground, nxt_hw, nxt_style, facing=facing)
        self._render_piece(img, cur_x, ground)
        if self.rng.random() < 0.25:
            # Extra horizontal mirror augmentation reduces bias to a single jump side.
            img = img[:, ::-1].copy()
        img = self._apply_post(img)
        return img[None, :, :]
