"""算法(Agent)基类与公共组件。

统一接口让 train.py 不关心具体算法:

- act(obs, deterministic): 选动作。
- observe(...): 接收一条转移,算法自行决定存回放还是攒 rollout。
- update(): 每个环境步后调用,算法自行决定是否真的训练;返回训练指标 dict 或 None。
- save/load: 检查点包含重建所需的全部信息(env 名、算法名、权重)。

新算法:实现本接口后在 algos/__init__.py 的 ALGOS 注册表里登记。
"""
import torch
import torch.nn as nn


def mlp(in_dim, out_dim, hidden=(128, 128)):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class BaseAgent:
    name = "base"

    def __init__(self, obs_dim, n_actions, device="cpu"):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)

    def act(self, obs, deterministic=False) -> int:
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        raise NotImplementedError

    def update(self):
        return None

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, sd):
        raise NotImplementedError

    def save(self, path, env_name, extra=None):
        payload = {
            "algo": self.name,
            "env": env_name,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path):
        return torch.load(path, map_location="cpu", weights_only=False)
