"""跳一跳 —— 纯 Python 自制,一维"蓄力跳台"游戏。

小人站在当前台子中心,下一块台子在前方随机距离处出现,宽度也随机。
每一步,agent 选一个跳跃力度(离散 N 档,映射到一段跳跃距离),小人
腾空落下:
- 落在下一块台面上即成功,前进到该台子,刷新更前方的新台子;落点越靠
  台心给的奖励越高(还原原版"踩中心连击得高分"的手感)。
- 落到台子外面(摔进缝里)即失败,回合结束。

随机性来源:每一步的台距与台宽都重新随机 —— agent 背不下来固定力度,
只能学出"看着缺口大小调力度"的映射,本质是一道带容差的回归题。

观测 2 维(均归一化):
- 到下一块台子中心的水平距离(缺口大小)
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

SUCCESS_SCORE = 25            # 连续踩中这么多块台子算通关
MISS_PENALTY = -5.0
PERFECT_FRAC = 0.2           # 落点误差 < 半宽的这个比例算"正中靶心"


class JumpEnv(BaseEnv):
    obs_dim = 2
    n_actions = N_LEVELS
    max_steps = 100

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.x = 0.0                 # 当前所站台子的中心
        self.cur_half = 0.0
        self.cur_shape = 0           # 0 方台, 1 圆台(仅供前端画图)
        self.next_center = 0.0
        self.next_half = 0.0
        self.next_shape = 0
        self.score = 0
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = 0.0
        self.cur_half = float(self.rng.uniform(HALF_MIN, HALF_MAX))
        self.cur_shape = int(self.rng.integers(2))
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
        land = self.x + dist
        target = self.next_center
        half = self.next_half
        err = abs(land - target)
        landed = err <= half
        self.t += 1

        x0 = self.x
        p0 = (self.x, self.cur_half)
        p1 = (self.next_center, self.next_half)
        s0, s1 = self.cur_shape, self.next_shape
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
            self.x = self.next_center
            self.cur_half = self.next_half
            self.cur_shape = self.next_shape
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
                "x0": round(x0, 3),
                "x1": round(land, 3),
                "p0": [round(p0[0], 3), round(p0[1], 3)],
                "p1": [round(p1[0], 3), round(p1[1], 3)],
                "s0": s0, "s1": s1,
                "ok": bool(landed),
                "perfect": bool(perfect),
                "score": self.score,
            })
        return self._obs(), reward, terminated, truncated, info

    def _spawn_next(self):
        gap = float(self.rng.uniform(GAP_MIN, GAP_MAX))
        self.next_center = self.x + gap
        self.next_half = float(self.rng.uniform(HALF_MIN, HALF_MAX))
        self.next_shape = int(self.rng.integers(2))

    def _obs(self):
        gap = self.next_center - self.x
        gap_mid, gap_amp = (GAP_MIN + GAP_MAX) / 2, (GAP_MAX - GAP_MIN) / 2
        half_mid, half_amp = (HALF_MIN + HALF_MAX) / 2, (HALF_MAX - HALF_MIN) / 2
        return np.array([
            (gap - gap_mid) / gap_amp,
            (self.next_half - half_mid) / half_amp,
        ], dtype=np.float32)

    def _record_ready(self):
        # 开局静止帧:小人站在 0 号台中心,下一块台子已就位(x0==x1 不起跳)
        self.frames.append({
            "x0": round(self.x, 3),
            "x1": round(self.x, 3),
            "p0": [round(self.x, 3), round(self.cur_half, 3)],
            "p1": [round(self.next_center, 3), round(self.next_half, 3)],
            "s0": self.cur_shape, "s1": self.next_shape,
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
