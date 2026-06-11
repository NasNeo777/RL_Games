"""评估工具:train.py 与各自带训练循环的算法(如 SB3)共用。"""


def evaluate(env, agent, episodes):
    """跑 episodes 局贪心策略,返回 (平均回报, 成功率, 平均摆起秒数或 None)。"""
    returns, swingups, successes = [], [], 0
    for i in range(episodes):
        obs = env.reset(seed=10_000 + i)
        total, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, info = env.step(
                agent.act(obs, deterministic=True))
            total += r
            done = terminated or truncated
        returns.append(total)
        successes += int(info.get("success", False))
        if "swingup_seconds" in info:
            swingups.append(info["swingup_seconds"])
    avg_swingup = round(sum(swingups) / len(swingups), 2) if swingups else None
    return sum(returns) / len(returns), successes / episodes, avg_swingup
