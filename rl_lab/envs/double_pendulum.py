"""二阶摆(Acrobot)甩摆环境。

两根连杆吊在固定支点上,只有两杆之间的关节有电机(欠驱动),
目标是把末端甩到支点上方。动力学采用 Sutton & Barto 书中的
Acrobot 方程(与 Gymnasium Acrobot-v1 相同的参数)。

状态: theta1(第一杆与竖直向下方向夹角), theta2(第二杆相对第一杆), 及角速度。
观测: [cos t1, sin t1, cos t2, sin t2, dt1/4pi, dt2/9pi]
动作: 关节力矩 {-1, 0, +1}
奖励: 末端高度 h/2 ∈ [-1,1] 的稠密塑形; h > 1.5 视为摆上去,+50 并结束。
"""
import math

import numpy as np

from .base import BaseEnv

# 物理参数(与 Gymnasium Acrobot 的 book 版本一致)
L1, L2 = 1.0, 1.0          # 杆长
M1, M2 = 1.0, 1.0          # 质量
LC1, LC2 = 0.5, 0.5        # 质心位置
I1, I2 = 1.0, 1.0          # 转动惯量
G = 9.8
MAX_VEL1 = 4 * math.pi
MAX_VEL2 = 9 * math.pi

DT = 0.2                   # 一个控制步的时长
SUBSTEPS = 4               # 每个控制步细分 4 次 RK4,顺便给前端记录平滑帧

TORQUES = (-1.0, 0.0, 1.0)
SUCCESS_HEIGHT = 1.5       # 末端高度阈值(最大为 2)
SUCCESS_BONUS = 50.0


def _dynamics(s, tau):
    t1, t2, dt1, dt2 = s
    d1 = (M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * math.cos(t2))
          + I1 + I2)
    d2 = M2 * (LC2**2 + L1 * LC2 * math.cos(t2)) + I2
    phi2 = M2 * LC2 * G * math.cos(t1 + t2 - math.pi / 2)
    phi1 = (-M2 * L1 * LC2 * dt2**2 * math.sin(t2)
            - 2 * M2 * L1 * LC2 * dt2 * dt1 * math.sin(t2)
            + (M1 * LC1 + M2 * L1) * G * math.cos(t1 - math.pi / 2)
            + phi2)
    ddt2 = ((tau + d2 / d1 * phi1
             - M2 * L1 * LC2 * dt1**2 * math.sin(t2) - phi2)
            / (M2 * LC2**2 + I2 - d2**2 / d1))
    ddt1 = -(d2 * ddt2 + phi1) / d1
    return np.array([dt1, dt2, ddt1, ddt2])


def _rk4(s, tau, dt):
    k1 = _dynamics(s, tau)
    k2 = _dynamics(s + dt / 2 * k1, tau)
    k3 = _dynamics(s + dt / 2 * k2, tau)
    k4 = _dynamics(s + dt * k3, tau)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


class DoublePendulumEnv(BaseEnv):
    obs_dim = 6
    n_actions = len(TORQUES)
    max_steps = 500

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros(4)
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s = self.rng.uniform(-0.1, 0.1, size=4)
        self.t = 0
        self.frames = []
        if self.record:
            self._record_frame(0.0)
        return self._obs()

    def step(self, action):
        tau = TORQUES[int(action)]
        for _ in range(SUBSTEPS):
            self.s = _rk4(self.s, tau, DT / SUBSTEPS)
            self.s[0] = _wrap(self.s[0])
            self.s[1] = _wrap(self.s[1])
            self.s[2] = np.clip(self.s[2], -MAX_VEL1, MAX_VEL1)
            self.s[3] = np.clip(self.s[3], -MAX_VEL2, MAX_VEL2)
            if self.record:
                self._record_frame(tau)
        self.t += 1

        h = self._tip_height()
        reward = h / 2.0
        success = h > SUCCESS_HEIGHT
        if success:
            reward += SUCCESS_BONUS
        truncated = (not success) and self.t >= self.max_steps
        info = {"success": success, "height": h}
        return self._obs(), reward, success, truncated, info

    def _tip_height(self):
        t1, t2 = self.s[0], self.s[1]
        return -math.cos(t1) - math.cos(t1 + t2)

    def _obs(self):
        t1, t2, dt1, dt2 = self.s
        return np.array([math.cos(t1), math.sin(t1),
                         math.cos(t2), math.sin(t2),
                         dt1 / MAX_VEL1, dt2 / MAX_VEL2], dtype=np.float32)

    def _record_frame(self, tau):
        self.frames.append({
            "t1": round(float(self.s[0]), 4),
            "t2": round(float(self.s[1]), 4),
            "tau": tau,
            "h": round(self._tip_height(), 3),
        })

    def render_spec(self):
        return {
            "type": "double_pendulum",
            "l1": L1, "l2": L2,
            "success_height": SUCCESS_HEIGHT,
            "frame_dt": DT / SUBSTEPS,
        }
