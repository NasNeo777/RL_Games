"""算法注册表。新算法在这里登记即可被 train.py / server.py 使用。"""
from .dqn import DQNAgent
from .ppo import PPOAgent

ALGOS = {
    "dqn": DQNAgent,
    "ppo": PPOAgent,
}


def make_agent(name, obs_dim, n_actions, **kwargs):
    if name not in ALGOS:
        raise KeyError(f"未知算法 {name!r},可选: {sorted(ALGOS)}")
    return ALGOS[name](obs_dim, n_actions, **kwargs)
