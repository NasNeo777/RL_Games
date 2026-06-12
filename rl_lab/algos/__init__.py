"""算法注册表。新算法在这里登记即可被 train.py / server.py 使用。"""
from .dqn import DQNAgent
from .ppo import PPOAgent
from .sb3_ppo import SB3PPOAgent
from .td2048 import TD2048Agent

ALGOS = {
    "dqn": DQNAgent,
    "ppo": SB3PPOAgent,        # PPO 用 Stable-Baselines3 实现
    "ppo_custom": PPOAgent,    # 旧的手写版,保留以兼容历史检查点
    "td2048": TD2048Agent,     # 2048 专用 afterstate TD + N-tuple 查表
}


def make_agent(name, obs_dim, n_actions, **kwargs):
    if name not in ALGOS:
        raise KeyError(f"未知算法 {name!r},可选: {sorted(ALGOS)}")
    return ALGOS[name](obs_dim, n_actions, **kwargs)
