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

如果训练目录已存在 latest.pt,**默认会自动续练**(回合数/步数从上次的累计往下加);
要重头训,加 --restart;`--resume` 兼容旧用法,与默认行为相同。

server.py 读取这些文件做演示,训练和展示互不阻塞。
"""
import argparse
import json
import shutil
import time
from collections import deque
from pathlib import Path

from .algos import ALGOS, make_agent
from .envs import ENVS, make_env
from .evaluation import evaluate

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description="持续训练 RL 智能体")
    p.add_argument("--env", default="double_pendulum", choices=sorted(ENVS))
    p.add_argument("--algo", default="ppo", choices=sorted(ALGOS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-envs", type=int, default=0,
                   help="SB3 并行采样环境数;0=自动(图像观测 8,其余 1)")
    p.add_argument("--eval-every", type=int, default=20,
                   help="每多少个训练回合评估一次")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--solve-rate", type=float, default=1.0,
                   help="评估成功率达到该值视为解决")
    p.add_argument("--forever", action="store_true",
                   help="解决后也继续训练")
    p.add_argument("--resume", action="store_true",
                   help="(已是默认)有 latest.pt 就接着练;保留是为了兼容旧脚本")
    p.add_argument("--restart", action="store_true",
                   help="忽略已有训练目录,从头训练(会备份旧目录为 *_old)")
    args = p.parse_args()

    env = make_env(args.env, seed=args.seed)
    eval_env = make_env(args.env)
    agent = make_agent(args.algo, env.obs_dim, env.n_actions,
                       device=args.device, seed=args.seed)
    if env.obs_shape and not getattr(agent, "supports_image_obs", False):
        raise SystemExit(f"环境 {args.env} 是图像观测,算法 {args.algo} "
                         f"只支持向量观测;请用 --algo ppo")

    run_dir = ROOT / "runs" / f"{args.env}_{args.algo}"
    if args.restart and run_dir.exists():
        backup = run_dir.with_name(run_dir.name + "_old")
        if backup.exists():
            shutil.rmtree(backup)
        run_dir.rename(backup)
        print(f"--restart: 旧训练目录已备份为 {backup.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest, best = run_dir / "latest.pt", run_dir / "best.pt"
    metrics_path = run_dir / "metrics.jsonl"
    meta_path = run_dir / "meta.json"

    # 默认自动续练:只要 latest.pt 存在(且没 --restart),就接着练
    resumed = False
    episode, env_steps = 0, 0
    best_return = float("-inf")
    if latest.exists():
        try:
            ckpt = agent.load_checkpoint(latest)
            if (ckpt.get("env") == args.env
                    and ckpt.get("algo") == args.algo):
                agent.load_state_dict(ckpt["state_dict"])
                resumed = True
                print(f"🔄 继续训练:已从 {latest} 恢复权重")
            else:
                print(f"⚠ {latest} 的 env/algo 与当前不一致,改从头训练")
        except Exception as e:
            print(f"⚠ 恢复检查点失败 ({e}),改从头训练")
    if resumed and metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            rec = json.loads(line)
            best_return = max(best_return, rec.get("eval_return", best_return))
            episode = max(episode, rec.get("episode", 0))
            env_steps = max(env_steps, rec.get("env_steps", 0))
        if episode or env_steps:
            print(f"   累计已训练 {episode} 回合 / {env_steps} 步,"
                  f"历史最优回报 {best_return:.1f}")

    print(f"{'继续训练' if resumed else '开始训练'}: "
          f"env={args.env} algo={args.algo} -> {run_dir}")

    # SB3 这类算法自带训练循环,评估/检查点/metrics 由其回调按同一格式落盘
    if getattr(agent, "trains_itself", False):
        agent.train_loop(env=env, eval_env=eval_env, args=args,
                         run_dir=run_dir, episode0=episode,
                         env_steps0=env_steps, best_return=best_return)
        return

    started = time.time()
    solved = False
    recent_returns = deque(maxlen=20)
    obs = env.reset(seed=args.seed)
    ep_ret = 0.0
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
