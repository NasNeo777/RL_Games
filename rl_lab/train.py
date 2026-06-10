"""持续训练入口。

用法:
    python -m rl_lab.train --env double_pendulum --algo ppo
    python -m rl_lab.train --env double_pendulum --algo dqn --forever

默认一直训练,直到评估中连续摆上去(成功率达到 --solve-rate)才停;
加 --forever 则解决后也继续优化。检查点写入 runs/<env>_<algo>/:

    latest.pt      每次评估都覆盖(最新模型)
    best.pt        评估回报创新高时覆盖(最优模型)
    metrics.jsonl  每次评估追加一行,网页训练曲线的数据源
    meta.json      运行状态摘要

server.py 读取这些文件做演示,训练和展示互不阻塞。
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

from .algos import ALGOS, make_agent
from .envs import ENVS, make_env

ROOT = Path(__file__).resolve().parent.parent


def evaluate(env, agent, episodes):
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


def main():
    p = argparse.ArgumentParser(description="持续训练 RL 智能体")
    p.add_argument("--env", default="double_pendulum", choices=sorted(ENVS))
    p.add_argument("--algo", default="ppo", choices=sorted(ALGOS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-every", type=int, default=20,
                   help="每多少个训练回合评估一次")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--solve-rate", type=float, default=1.0,
                   help="评估成功率达到该值视为解决")
    p.add_argument("--forever", action="store_true",
                   help="解决后也继续训练")
    p.add_argument("--resume", action="store_true",
                   help="从 latest.pt 恢复权重继续训练")
    args = p.parse_args()

    env = make_env(args.env, seed=args.seed)
    eval_env = make_env(args.env)
    agent = make_agent(args.algo, env.obs_dim, env.n_actions,
                       device=args.device, seed=args.seed)

    run_dir = ROOT / "runs" / f"{args.env}_{args.algo}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest, best = run_dir / "latest.pt", run_dir / "best.pt"
    metrics_path = run_dir / "metrics.jsonl"
    meta_path = run_dir / "meta.json"

    if args.resume and latest.exists():
        ckpt = agent.load_checkpoint(latest)
        agent.load_state_dict(ckpt["state_dict"])
        print(f"已从 {latest} 恢复权重")

    best_return = float("-inf")
    if args.resume and metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            rec = json.loads(line)
            best_return = max(best_return, rec.get("eval_return", best_return))

    started = time.time()
    episode, env_steps, solved = 0, 0, False
    recent_returns = deque(maxlen=20)
    obs = env.reset(seed=args.seed)
    ep_ret = 0.0

    print(f"开始训练: env={args.env} algo={args.algo} -> {run_dir}")
    while True:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated, truncated)
        train_metrics = agent.update()
        obs = next_obs
        ep_ret += reward
        env_steps += 1

        if not (terminated or truncated):
            continue

        episode += 1
        recent_returns.append(ep_ret)
        obs = env.reset()
        ep_ret = 0.0

        if episode % args.eval_every != 0:
            continue

        eval_return, success_rate, avg_swingup = evaluate(
            eval_env, agent, args.eval_episodes)
        train_avg = sum(recent_returns) / len(recent_returns)
        is_best = eval_return > best_return
        if is_best:
            best_return = eval_return
            agent.save(best, args.env, extra={"eval_return": eval_return})
        agent.save(latest, args.env, extra={"eval_return": eval_return})

        if success_rate >= args.solve_rate:
            solved = True

        record = {
            "time": round(time.time() - started, 1),
            "episode": episode,
            "env_steps": env_steps,
            "train_return": round(train_avg, 2),
            "eval_return": round(eval_return, 2),
            "success_rate": round(success_rate, 3),
            "swingup_seconds": avg_swingup,
            "best_return": round(best_return, 2),
            "solved": solved,
        }
        if train_metrics:
            record["algo_metrics"] = train_metrics
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        meta_path.write_text(json.dumps({
            "env": args.env, "algo": args.algo,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            **record,
        }, indent=2))

        flag = " [BEST]" if is_best else ""
        print(f"回合 {episode:>5} | 步数 {env_steps:>8} | "
              f"训练回报 {train_avg:8.1f} | 评估回报 {eval_return:8.1f} | "
              f"成功率 {success_rate:.0%}"
              + (f" | 摆起 {avg_swingup}s" if avg_swingup else "")
              + flag + ("  ✅ 已解决" if solved else ""))

        if solved and not args.forever:
            print(f"评估成功率达到 {args.solve_rate:.0%},训练完成。"
                  f"最优模型: {best}")
            break


if __name__ == "__main__":
    main()
