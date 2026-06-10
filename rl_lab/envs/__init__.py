"""环境注册表。新游戏在这里登记即可被 train.py / server.py 使用。"""
from .double_pendulum import DoublePendulumEnv
from .mountain_car import MountainCarEnv

ENVS = {
    "double_pendulum": DoublePendulumEnv,
    "mountain_car": MountainCarEnv,
}


def make_env(name, **kwargs):
    if name not in ENVS:
        raise KeyError(f"未知环境 {name!r},可选: {sorted(ENVS)}")
    return ENVS[name](**kwargs)
