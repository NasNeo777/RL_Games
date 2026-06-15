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

    def _obs(self):
        img = np.zeros((PIX, PIX), dtype=np.uint8)
        gap = self._gap()
        # 世界 x → 像素:展示 [-0.7, D_MAX+HALF_MAX+0.3] 这段
        xmin, xmax = -0.7, D_MAX + HALF_MAX + 0.3
        ppu = PIX / (xmax - xmin)
        px = lambda x: int(round((x - xmin) * ppu))
        clip = lambda v: max(0, min(PIX, v))
        ground, thick = int(PIX * 0.60), int(PIX * 0.22)
        # 当前台子(中等亮度)
        img[ground:ground + thick, clip(px(-self.cur_half)):clip(px(self.cur_half))] = 130
        # 目标台子(最亮),位置=缺口,宽度=台宽
        img[ground:ground + thick,
            clip(px(gap - self.next_half)):clip(px(gap + self.next_half))] = 255
        # 棋子(当前台子正上方的小方块)
        cx = px(0.0)
        img[ground - int(PIX * 0.19):ground - 2, clip(cx - 3):clip(cx + 3)] = 255
        return img[None, :, :]

