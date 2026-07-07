"""训练进度与 ETA 估算工具。"""
from __future__ import annotations

from typing import Optional


BUDGETS = {
    "mountain_car_dqn": {
        "steps": 16_000,
        "src": "实测",
        "rough": "1 分钟",
    },
    "tetris_dqn": {
        "steps": 570_000,
        "src": "实测",
        "rough": "8 分钟",
    },
    "mario_ppo": {
        "steps": 4_300_000,
        "src": "实测",
        "rough": "一夜",
    },
    "flappy_bird_ppo": {
        "steps": 23_200_000,
        "src": "实测",
        "rough": "数小时",
    },
    "double_pendulum_ppo": {
        "steps": 155_000_000,
        "src": "实测",
        "rough": "数小时",
    },
    "snake_dqn": {
        "steps": 3_000_000,
        "src": "估算",
        "rough": "2 分钟",
    },
    "snake_gate_dqn": {
        "steps": 80_000,
        "src": "估算",
        "rough": "几分钟",
    },
    "snake_gate_ppo": {
        "steps": 120_000,
        "src": "估算",
        "rough": "几分钟",
    },
    "2048_td2048": {
        "steps": 10_000_000,
        "src": "估算",
        "rough": "7 分钟",
    },
    "2048_dqn": {
        "tip": "DQN 在 2048 上极慢,强烈建议改用 --algo td2048。",
    },
    "flappy_bird_dqn": {
        "tip": "DQN 能较快学个大概,但要冲 777 管道的练成标准建议改用 --algo ppo。",
    },
}


def budget_for(env_name: str, algo: str):
    return BUDGETS.get(f"{env_name}_{algo}")


def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} 秒"
    if seconds < 5400:
        return f"{round(seconds / 60)} 分钟"
    return f"{seconds / 3600:.1f} 小时"


def startup_message(env_name: str, algo: str,
                    env_steps: int = 0) -> Optional[str]:
    budget = budget_for(env_name, algo)
    if not budget:
        return None
    if "tip" in budget:
        return f"进度提示: {budget['tip']}"
    pct = min(99, env_steps / budget["steps"] * 100) if env_steps else 0
    return ("进度提示: 这组配置参考要跑 "
            f"{budget['steps'] / 1000:.0f}k 步({budget['src']}),"
            f"通常约 {budget['rough']}能接近练成。"
            + (f" 当前已到 {pct:.0f}% 左右。" if env_steps else ""))


def progress_message(env_name: str, algo: str, env_steps: int,
                     session_started: float, session_env_steps0: int,
                     solved: bool = False) -> Optional[str]:
    budget = budget_for(env_name, algo)
    if not budget:
        return None
    if "tip" in budget:
        return f"进度提示: {budget['tip']}"
    pct = min(100 if solved else 99, env_steps / budget["steps"] * 100)
    base = (f"进度 {pct:.0f}%"
            f" ({env_steps / 1000:.0f}k/{budget['steps'] / 1000:.0f}k 步,"
            f" {budget['src']})")
    if solved:
        return base + "，已达到练成标准。"
    if env_steps >= budget["steps"]:
        return base + "，已超过参考步数；这类任务波动较大，继续跑并观察成功率。"
    elapsed = max(0.0, session_started)
    delta_steps = env_steps - session_env_steps0
    if elapsed > 1 and delta_steps > 0:
        rate = delta_steps / elapsed
        eta = (budget["steps"] - env_steps) / rate if rate > 0 else None
        if eta:
            return (base + f"，当前约 {rate:.0f} 步/秒，"
                    f"预计还需 {fmt_duration(eta)}。")
    return base + f"，通常约 {budget['rough']}能接近练成。"
