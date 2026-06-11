"""把本框架的 BaseEnv 包装成标准 gymnasium.Env,供 SB3 等外部库使用。"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BaseEnvToGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_env):
        super().__init__()
        self.base = base_env
        if base_env.obs_shape:    # 图像观测(通道, 高, 宽) uint8
            self.observation_space = spaces.Box(
                0, 255, shape=base_env.obs_shape, dtype=np.uint8)
        else:                     # 向量观测
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(base_env.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(base_env.n_actions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.base.reset(seed=seed), {}

    def step(self, action):
        return self.base.step(action)
