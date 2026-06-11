"""Flappy Bird —— 来自 flappy-bird-gymnasium 集成包(不自己写游戏逻辑)。

观测 12 维(管道位置、小鸟位置/速度等,包内已归一化),动作 2 个。
包内奖励:存活每步 +0.1,过管 +1,死亡 -1。
连过 SUCCESS_SCORE 根管道视为成功,加奖金并结束回合。
"""
from .gym_adapter import GymEnv

SUCCESS_SCORE = 777


class FlappyBirdEnv(GymEnv):
    env_id = "FlappyBird-v0"
    import_module = "flappy_bird_gymnasium"
    gym_kwargs = {"use_lidar": False}
    # 777 根管道约需 3 万步,步数上限要给够,否则永远到不了成功
    max_steps = 40000
    success_bonus = 50.0

    def is_success(self, info):
        return info.get("score", 0) >= SUCCESS_SCORE
