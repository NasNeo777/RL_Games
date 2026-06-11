"""超级马里奥(世界 1-1)—— 来自 gym-super-mario-bros 集成包。

agent **不直接看原始游戏画面**(240x256 RGB),观测经过经典的
Atari 式预处理流水线(本文件自己实现,方便学习):

1. 跳帧:同一动作连按 SKIP=4 帧,只决策一次(NES 60fps -> agent 15Hz),
   并对最后两帧逐像素取 max,消除 NES 精灵隔帧闪烁;
2. 灰度化:RGB 按亮度加权成单通道,丢掉颜色信息;
3. 栅格化缩小:240x256 双线性缩到 84x84,只留轮廓和布局;
4. 叠帧:最近 STACK=4 个处理后的帧叠成 (4, 84, 84),让网络能感知速度。

最终观测是 (4, 84, 84) uint8,SB3 的 CnnPolicy 会自动 /255 归一化。

动作用 SIMPLE_MOVEMENT 7 键组合(不动/右/右跳/右加速/右加速跳/跳/左)。
奖励直接用包内置的(x 前进量 + 时间流逝惩罚 + 死亡 -15,包内已裁剪
到 ±15,不再缩放);拿到旗子(flag_get)视为成功,加奖金并结束回合。
"""
import base64
import io
from collections import deque

import numpy as np

from .base import BaseEnv

SKIP = 4          # 跳帧数:每个 agent 步 = 4 个游戏帧
SIZE = 84         # 缩小后的边长
STACK = 4         # 叠帧数


def _patch_numpy2():
    """gym-super-mario-bros 7.4.0 在 numpy 2 下读 RAM 做 uint8 运算会
    溢出(报错或刷警告),打补丁换成 Python int 运算。"""
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
    SuperMarioBrosEnv._x_position = property(
        lambda self: int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86]))
    SuperMarioBrosEnv._y_position = property(
        lambda self: (255 + (255 - int(self._y_pixel)))
        if self._y_viewport < 1 else 255 - int(self._y_pixel))
    return SuperMarioBrosEnv


class MarioEnv(BaseEnv):
    obs_shape = (STACK, SIZE, SIZE)   # 图像观测,SB3 走 CnnPolicy
    n_actions = 7                     # len(SIMPLE_MOVEMENT)
    # 游戏时钟 400 约合 2400 个 agent 步,超时马里奥会死(terminated),
    # 这里只是兜底
    max_steps = 3000
    success_bonus = 50.0

    render_every = 1                  # 每个 agent 步抓一帧(已含 4 倍跳帧)
    jpeg_quality = 75
    max_record_frames = 1500

    def __init__(self, seed=None):
        super().__init__()
        SuperMarioBrosEnv = _patch_numpy2()
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
        # 直接构造(不走 gym.make),固定世界 1-1
        env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False,
                                target=(1, 1))
        self._env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.obs_dim = int(np.prod(self.obs_shape))
        self._stack = deque(maxlen=STACK)
        self._seed = seed
        self.t = 0

    # ---- 观测预处理 ----

    @staticmethod
    def _preprocess(rgb):
        """RGB 游戏帧 -> 84x84 灰度 uint8。"""
        from PIL import Image
        gray = Image.fromarray(rgb).convert("L")          # 灰度化
        small = gray.resize((SIZE, SIZE), Image.BILINEAR)  # 栅格化缩小
        return np.asarray(small, dtype=np.uint8)

    def _observe(self, frame):
        self._stack.append(self._preprocess(frame))
        return np.stack(self._stack)                       # (4, 84, 84)

    # ---- gym 风格接口 ----

    def reset(self, seed=None):
        if seed is None and self._seed is not None:
            seed, self._seed = self._seed, None
        frame, _ = self._env.reset(seed=seed)
        self.t = 0
        self.frames = []
        self._stack.clear()
        for _ in range(STACK):                # 初始时把同一帧复制满栈
            self._stack.append(self._preprocess(frame))
        if self.record:
            self._record_frame({})
        return np.stack(self._stack)

    def step(self, action):
        total_reward, terminated, truncated, info = 0.0, False, False, {}
        last_two = deque(maxlen=2)            # 跳帧期间的最后两帧
        for _ in range(SKIP):
            frame, r, terminated, truncated, info = self._env.step(int(action))
            total_reward += float(r)
            last_two.append(frame)
            if terminated or truncated:
                break
        # 相邻两帧取 max,消除 NES 精灵隔帧闪烁
        frame = np.max(np.stack(last_two), axis=0) if len(last_two) == 2 \
            else last_two[0]
        obs = self._observe(frame)

        self.t += 1
        reward = total_reward
        success = bool(info.get("flag_get"))
        if success:
            reward += self.success_bonus
            terminated = True
        if not terminated and self.t >= self.max_steps:
            truncated = True
        info = dict(info)
        info["success"] = success
        if (self.record and len(self.frames) < self.max_record_frames
                and (self.t % self.render_every == 0
                     or terminated or truncated)):
            self._record_frame(info)
        return obs, reward, bool(terminated), bool(truncated), info

    # ---- 网页演示:每帧录原始彩色画面 + agent 实际看到的预处理观测 ----

    def _record_frame(self, info):
        from PIL import Image
        img = Image.fromarray(np.asarray(self._env.unwrapped.screen))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        frame = {"img": "data:image/jpeg;base64,"
                        + base64.b64encode(buf.getvalue()).decode()}
        # 叠帧栈里最新的 84x84 灰度帧,PNG 无损(本来就是给网络看的小图)
        obs_buf = io.BytesIO()
        Image.fromarray(self._stack[-1], mode="L").save(obs_buf, format="PNG")
        frame["obs"] = ("data:image/png;base64,"
                        + base64.b64encode(obs_buf.getvalue()).decode())
        if "score" in info:
            frame["score"] = int(info["score"])
        self.frames.append(frame)

    def render_spec(self):
        # 每个 agent 步 = SKIP 个 60fps 游戏帧
        return {"type": "video",
                "frame_dt": self.render_every * SKIP / 60.0,
                "obs_label": f"agent 观测 {SIZE}×{SIZE} 灰度×{STACK}帧(最新帧)"}
