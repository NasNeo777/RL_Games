"""二阶摆(Acrobot)甩起 + 稳定倒立环境。

两根连杆吊在固定支点上,只有两杆之间的关节有电机(欠驱动)。
目标不是甩过某个高度,而是:**尽快甩到倒立位置并稳定保持**。
连续在倒立区(末端高度 > 1.9 且角速度足够小)保持 HOLD_STEPS 步
(5 秒)才算成功;成功奖金随用时减少而增加,逼智能体学快速甩起。

倒立是不稳定平衡点,所以控制频率取 20Hz(dt=0.05),
力矩 5 档 {-2,-1,0,1,2},否则稳不住。

状态: theta1(第一杆与竖直向下方向夹角), theta2(第二杆相对第一杆), 及角速度。
观测: [cos t1, sin t1, cos t2, sin t2, dt1/4pi, dt2/9pi]
奖励: 高度塑形(小) - 顶部超速惩罚 + 倒立区每步 +1 + 成功奖金(含速度加成)。
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

DT = 0.05                  # 控制步长(20Hz,倒立平衡必须够快)
# ±0.25 两档微调力矩是平衡的关键:只有 ±1/±2 粗档时,
# 策略在倒立点附近只能反复过度修正,陷入小幅晃动的极限环。
TORQUES = (-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0)

UPRIGHT_H = 1.9            # 倒立区:末端高度阈值(满高 2.0)
UPRIGHT_VEL1 = 3.0         # 倒立区:第一关节角速度上限
UPRIGHT_VEL2 = 5.0         # 倒立区:第二关节角速度上限
HOLD_STEPS = 100           # 连续保持 5 秒才算成功
SUCCESS_BONUS = 100.0      # 成功基础奖金;另按剩余时间最多再翻一倍

# 课程学习:训练时一定比例的回合直接从"接近倒立"开始,
# 让 agent 把"稳"这件事单独学会(评估永远从下垂开始,任务不变)。
CURRICULUM_PROB = 0.3


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
    max_steps = 600            # 30 秒

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros(4)
        self.t = 0
        self.hold = 0

    def reset(self, seed=None):
        # 评估总是传 seed,训练第一步也传一次;之后训练 reset() 不带 seed,
        # 这时按概率切到"近倒立"起手,做课程学习。
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.s = self.rng.uniform(-0.1, 0.1, size=4)
        elif self.rng.random() < CURRICULUM_PROB:
            # 在完全倒立 (theta1=pi, theta2=0) 附近撒一些角度/角速度噪声
            self.s = np.array([math.pi, 0.0, 0.0, 0.0]) \
                + self.rng.uniform(-0.4, 0.4, size=4)
        else:
            self.s = self.rng.uniform(-0.1, 0.1, size=4)
        self.t = 0
        self.hold = 0
        self.frames = []
        if self.record:
            self._record_frame(0.0)
        return self._obs()

    def step(self, action):
        tau = TORQUES[int(action)]
        self.s = _rk4(self.s, tau, DT)
        self.s[0] = _wrap(self.s[0])
        self.s[1] = _wrap(self.s[1])
        self.s[2] = np.clip(self.s[2], -MAX_VEL1, MAX_VEL1)
        self.s[3] = np.clip(self.s[3], -MAX_VEL2, MAX_VEL2)
        self.t += 1

        h = self._tip_height()
        dt1, dt2 = self.s[2], self.s[3]

        # 高度塑形(量级小,只起引导作用)
        reward = 0.05 * (h / 2.0)
        # 接近顶部时惩罚角速度:要减速到达,而不是高速甩过
        if h > 1.6:
            reward -= 0.005 * (dt1 * dt1 + dt2 * dt2)
        # 顶部精度奖励:1.8 以上连续递增,晃动幅度收得越紧拿得越多。
        # 没有这个梯度,策略学到"差不多直立"后就再无改进动力。
        if h > 1.8:
            reward += 2.0 * (h - 1.8) / 0.2

        upright = (h > UPRIGHT_H
                   and abs(dt1) < UPRIGHT_VEL1
                   and abs(dt2) < UPRIGHT_VEL2)
        if upright:
            self.hold += 1
            # 关键:奖励随连续保持步数线性递增,而不是恒定 +1。
            # 这样 agent 选择"持续静止"严格优于"反复进出倒立区刷短奖"。
            reward += 1.0 + 0.01 * self.hold
        else:
            self.hold = 0

        success = self.hold >= HOLD_STEPS
        if success:
            # 速度加成:越早稳住,奖金越高(1~2 倍)
            reward += SUCCESS_BONUS * (1.0 + (self.max_steps - self.t)
                                       / self.max_steps)
        truncated = (not success) and self.t >= self.max_steps
        info = {"success": success, "height": h, "hold": self.hold}
        if success:
            info["swingup_steps"] = self.t - HOLD_STEPS   # 摆起用的步数
            info["swingup_seconds"] = round((self.t - HOLD_STEPS) * DT, 2)
        if self.record:
            self._record_frame(tau, upright)
        return self._obs(), reward, success, truncated, info

    def _tip_height(self):
        t1, t2 = self.s[0], self.s[1]
        return -math.cos(t1) - math.cos(t1 + t2)

    def _obs(self):
        t1, t2, dt1, dt2 = self.s
        return np.array([math.cos(t1), math.sin(t1),
                         math.cos(t2), math.sin(t2),
                         dt1 / MAX_VEL1, dt2 / MAX_VEL2], dtype=np.float32)

    def _record_frame(self, tau, upright=False):
        self.frames.append({
            "t1": round(float(self.s[0]), 4),
            "t2": round(float(self.s[1]), 4),
            "tau": tau,
            "h": round(self._tip_height(), 3),
            "up": int(upright),
            "hold": self.hold,                 # 已连续稳定的步数,前端做倒计时
        })

    def render_spec(self):
        return {
            "type": "double_pendulum",
            "l1": L1, "l2": L2,
            "success_height": UPRIGHT_H,
            "hold_steps": HOLD_STEPS,
            "hold_seconds": HOLD_STEPS * DT,
            "frame_dt": DT,
        }
