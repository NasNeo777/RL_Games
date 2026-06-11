"""Gymnasium 通用适配器。

把任何 Gymnasium 环境包装成本框架的 BaseEnv 接口。
网页演示不再需要逐环境写 canvas 渲染器:适配器直接抓
`render_mode="rgb_array"` 的画面,压缩成 JPEG base64 帧,
前端用通用的 "video" 渲染器播放。

接入一个新 gym 游戏只需要子类三五行:

    class FlappyBirdEnv(GymEnv):
        env_id = "FlappyBird-v0"
        import_module = "flappy_bird_gymnasium"   # 注册环境的包,可省略
        gym_kwargs = {"use_lidar": False}
        max_steps = 1000

        def is_success(self, info):
            return info.get("score", 0) >= 20

再到 envs/__init__.py 的 ENVS 登记即可。
"""
import base64
import importlib
import io

import numpy as np

from .base import BaseEnv


class GymEnv(BaseEnv):
    env_id = None             # 必填:gym.make 的环境 id
    import_module = None      # 选填:import 后才会注册环境的包名
    gym_kwargs = {}           # 选填:传给 gym.make 的参数
    max_steps = 1000
    success_bonus = 50.0      # 达成 is_success 时的额外奖金(0 表示不加)
    render_every = 2          # 每隔几步抓一帧演示画面(权衡流畅度和体积)
    render_height = 512       # 高于此才缩;多数小游戏原生分辨率本就不高,保留原生
    jpeg_quality = 75

    def __init__(self, seed=None):
        super().__init__()
        if self.import_module:
            importlib.import_module(self.import_module)
        import gymnasium as gym
        self._env = gym.make(self.env_id, render_mode="rgb_array",
                             **self.gym_kwargs)
        space = self._env.observation_space
        self.obs_dim = int(np.prod(space.shape))
        self.n_actions = int(self._env.action_space.n)
        self._fps = self._env.metadata.get("render_fps", 30)
        self._seed = seed
        self.t = 0

    def is_success(self, info):
        """子类按游戏定义"成功";默认永不成功(只看回报)。"""
        return False

    def reset(self, seed=None):
        if seed is None and self._seed is not None:
            seed, self._seed = self._seed, None    # 构造时的种子只用一次
        obs, _ = self._env.reset(seed=seed)
        self.t = 0
        self.frames = []
        if self.record:
            self._record_frame({})
        return np.asarray(obs, dtype=np.float32).ravel()

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self.t += 1
        reward = float(reward)
        success = self.is_success(info)
        if success:
            reward += self.success_bonus
            terminated = True
        if not terminated and self.t >= self.max_steps:
            truncated = True
        info = dict(info)
        info["success"] = bool(success)
        if self.record and (self.t % self.render_every == 0
                            or terminated or truncated):
            self._record_frame(info)
        return (np.asarray(obs, dtype=np.float32).ravel(),
                reward, bool(terminated), bool(truncated), info)

    def _record_frame(self, info):
        from PIL import Image
        rgb = self._env.render()
        img = Image.fromarray(np.asarray(rgb))
        if img.height > self.render_height:
            w = round(img.width * self.render_height / img.height)
            img = img.resize((w, self.render_height), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        frame = {"img": "data:image/jpeg;base64,"
                        + base64.b64encode(buf.getvalue()).decode()}
        if "score" in info:
            frame["score"] = info["score"]
        self.frames.append(frame)

    def render_spec(self):
        return {
            "type": "video",
            "frame_dt": self.render_every / self._fps,
        }
