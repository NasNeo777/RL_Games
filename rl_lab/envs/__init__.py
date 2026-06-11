"""环境注册表。新游戏在这里登记即可被 train.py / server.py 使用。"""
from .double_pendulum import DoublePendulumEnv
from .flappy_bird import FlappyBirdEnv
from .mountain_car import MountainCarEnv

ENVS = {
    "double_pendulum": DoublePendulumEnv,
    "flappy_bird": FlappyBirdEnv,
    "mountain_car": MountainCarEnv,
}


def make_env(name, **kwargs):
    if name not in ENVS:
        raise KeyError(f"未知环境 {name!r},可选: {sorted(ENVS)}")
    return ENVS[name](**kwargs)
