"""MountainCar(小车爬山)。

经典欠驱动控制:小车在 U 型谷底,引擎推力不足以直接爬上右侧山顶,
必须先左右晃动积累动能。状态 2 维(位置、速度),3 个离散动作。

物理与 Gymnasium MountainCar-v0 完全一致,直接手写,避免 gym 依赖:
    velocity += (action - 1) * 0.001 - 0.0025 * cos(3 * position)
    velocity 截断到 [-0.07, 0.07]
    position += velocity
    position 截断到 [-1.2, 0.6],碰到左壁 velocity 清零
    位置 >= 0.5 视为登顶成功

原版奖励是每步 -1,稀疏到 DQN 几乎学不会;这里采用塑形:
- 刷新历史最右侧位置时按距离给一笔较大的奖励(鼓励向目标推进);
- 刷新历史最左侧位置时按距离给一笔较小的奖励(鼓励反向蓄力);
- 速度绝对值大于阈值时给小奖励(维持动能);
- 登顶 +100。
"""
import math

import numpy as np

from .base import BaseEnv

MIN_POS, MAX_POS = -1.2, 0.6
MAX_SPEED = 0.07
GOAL = 0.5
FORCE = 0.001
GRAVITY = 0.0025


class MountainCarEnv(BaseEnv):
    obs_dim = 2
    n_actions = 3
    max_steps = 200

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.position = 0.0
        self.velocity = 0.0
        self.t = 0
        self.max_position = None
        self.min_position = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 与 gym 一致:位置 ∈ [-0.6, -0.4],初速度 0
        self.position = float(self.rng.uniform(-0.6, -0.4))
        self.velocity = 0.0
        self.t = 0
        self.max_position = self.position
        self.min_position = self.position
        self.frames = []
        if self.record:
            self._record_frame(1)
        return self._obs()

    def step(self, action):
        action = int(action)
        self.velocity += (action - 1) * FORCE - GRAVITY * math.cos(
            3 * self.position)
        self.velocity = max(-MAX_SPEED, min(MAX_SPEED, self.velocity))
        self.position += self.velocity
        if self.position < MIN_POS:
            self.position = MIN_POS
            self.velocity = 0.0
        if self.position > MAX_POS:
            self.position = MAX_POS
        self.t += 1

        reward = -1.0  # 原版基线
        if self.position > self.max_position:
            reward += 20.0 * (self.position - self.max_position)
            self.max_position = self.position
        if self.position < self.min_position:
            reward += 5.0 * (self.min_position - self.position)
            self.min_position = self.position
        if abs(self.velocity) > 0.01:
            reward += 3.0 * abs(self.velocity)

        success = self.position >= GOAL
        if success:
            reward += 100.0
        truncated = (not success) and self.t >= self.max_steps
        info = {"success": success, "position": self.position,
                "velocity": self.velocity}
        if success:
            info["swingup_steps"] = self.t
            info["swingup_seconds"] = self.t  # 1 步 = 1 单位时间
        if self.record:
            self._record_frame(action)
        return self._obs(), reward, success, truncated, info

    def _obs(self):
        # 归一化到 ~[-1,1] 量级,网络好收敛
        return np.array([
            (self.position - (MIN_POS + MAX_POS) / 2) / ((MAX_POS - MIN_POS) / 2),
            self.velocity / MAX_SPEED,
        ], dtype=np.float32)

    def _record_frame(self, action):
        self.frames.append({
            "x": round(self.position, 4),
            "v": round(self.velocity, 4),
            "a": action,
            "xmax": round(self.max_position, 4),
            "xmin": round(self.min_position, 4),
        })

    def render_spec(self):
        return {
            "type": "mountain_car",
            "min_pos": MIN_POS, "max_pos": MAX_POS, "goal": GOAL,
            "frame_dt": 0.04,
        }
