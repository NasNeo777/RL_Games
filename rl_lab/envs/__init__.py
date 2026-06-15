"""环境注册表。新游戏在这里登记即可被 train.py / server.py 使用。"""
from .double_pendulum import DoublePendulumEnv
from .flappy_bird import FlappyBirdEnv
from .game_2048 import Game2048Env
from .jump import JumpEnv
from .mario import MarioEnv
from .mountain_car import MountainCarEnv
from .snake import SnakeEnv
from .tetris import TetrisEnv

ENVS = {
    "2048": Game2048Env,
    "double_pendulum": DoublePendulumEnv,
    "flappy_bird": FlappyBirdEnv,
    "jump": JumpEnv,
    "mario": MarioEnv,
    "mountain_car": MountainCarEnv,
    "snake": SnakeEnv,
    "tetris": TetrisEnv,
}


def make_env(name, **kwargs):
    if name not in ENVS:
        raise KeyError(f"未知环境 {name!r},可选: {sorted(ENVS)}")
    return ENVS[name](**kwargs)
