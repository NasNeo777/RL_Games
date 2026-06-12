"""PPO —— 由 Stable-Baselines3 提供实现。

SB3 自带训练循环(model.learn),所以本 Agent 设 trains_itself = True:
train.py 不再逐步驱动,而是调用 train_loop(),由 SB3 跑采样/更新,
评估、检查点、metrics.jsonl 通过回调以与框架完全相同的格式落盘,
网页界面无感知差异。

检查点兼容:state_dict() 把 SB3 模型 zip 打进 torch.save 负载里,
load_state_dict() 原样恢复,演示服务器加载路径不变。
"""
import io
import json
import time
from collections import deque
from functools import partial

from .base import BaseAgent
from ..progress import progress_message


def _make_sb3_env(env_name, seed):
    """SubprocVecEnv 子进程里的环境工厂(模块级函数才能被 pickle)。"""
    from stable_baselines3.common.monitor import Monitor

    from ..envs import make_env
    from ..envs.to_gym import BaseEnvToGym
    return Monitor(BaseEnvToGym(make_env(env_name, seed=seed)))


class SB3PPOAgent(BaseAgent):
    name = "ppo"
    trains_itself = True
    supports_image_obs = True   # 图像观测自动切 CnnPolicy

    # 向量观测(MlpPolicy)超参,与之前手写版调优一致
    HP = dict(learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
              gamma=0.995, gae_lambda=0.95, clip_range=0.2,
              ent_coef=0.003, vf_coef=0.5, max_grad_norm=0.5)
    POLICY_KWARGS = dict(net_arch=[128, 128])

    # 图像观测(CnnPolicy)超参,参考已验证的马里奥 PPO 配方:
    # 小 gamma(0.95)+ 紧裁剪(0.1)+ target_kl 早停,多环境并行采样
    HP_IMAGE = dict(learning_rate=2.5e-4, n_steps=2048, batch_size=2048,
                    n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.1,
                    ent_coef=0.1, vf_coef=0.5, max_grad_norm=0.8,
                    target_kl=0.03)
    # 熵系数从 HP_IMAGE["ent_coef"] 线性衰减到 end(按累计环境步数),
    # 前期多探索,后期降噪声帮收敛;end 不要太低,否则后期会
    # 探索不出关卡后段的难点(如 1-1 的悬崖跳)
    ENT_DECAY = dict(end=0.02, steps=2_000_000)
    N_ENVS_IMAGE = 8            # 图像环境默认并行数(--n-envs 可覆盖)

    def __init__(self, obs_dim, n_actions, device="cpu", seed=None):
        super().__init__(obs_dim, n_actions, device)
        self.seed = seed
        self.model = None      # 训练时由 train_loop 构建,演示时由 load_state_dict 恢复

    def act(self, obs, deterministic=False):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def state_dict(self):
        buf = io.BytesIO()
        self.model.save(buf)
        return {"sb3_zip": buf.getvalue()}

    def load_state_dict(self, sd):
        from stable_baselines3 import PPO
        self.model = PPO.load(io.BytesIO(sd["sb3_zip"]), device="cpu")

    # ---- 训练(由 train.py 调用) ----

    def train_loop(self, env, eval_env, args, run_dir,
                   episode0=0, env_steps0=0, best_return=float("-inf")):
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from ..envs.to_gym import BaseEnvToGym
        from ..evaluation import evaluate

        image_obs = env.obs_shape is not None
        n_envs = getattr(args, "n_envs", 0) \
            or (self.N_ENVS_IMAGE if image_obs else 1)
        if n_envs > 1:
            # 多环境并行采样:图像环境单局慢,靠并行喂饱大 rollout
            from stable_baselines3.common.vec_env import SubprocVecEnv
            venv = SubprocVecEnv([
                partial(_make_sb3_env, args.env, (args.seed or 0) + i)
                for i in range(n_envs)])
            print(f"并行采样: SubprocVecEnv x {n_envs}")
        else:
            venv = Monitor(BaseEnvToGym(env))

        if self.model is not None:                     # 续练
            if self.model.n_envs != getattr(venv, "num_envs", 1):
                # set_env 不允许改并行数,经由 zip 重载来换
                buf = io.BytesIO()
                self.model.save(buf)
                buf.seek(0)
                self.model = PPO.load(buf, env=venv, device=self.device)
            else:
                self.model.set_env(venv)
        else:
            # 图像观测(如 mario 的 4x84x84)用 CNN 策略,SB3 自动 /255;
            # CnnPolicy 的特征提取器(NatureCNN)后面不再需要额外 MLP 层
            policy = "CnnPolicy" if image_obs else "MlpPolicy"
            policy_kwargs = {} if image_obs else dict(self.POLICY_KWARGS)
            hp = self.HP_IMAGE if image_obs else self.HP
            self.model = PPO(policy, venv, device=self.device,
                             seed=self.seed, verbose=0,
                             policy_kwargs=policy_kwargs,
                             **hp)

        agent = self
        latest, best = run_dir / "latest.pt", run_dir / "best.pt"
        metrics_path, meta_path = run_dir / "metrics.jsonl", run_dir / "meta.json"
        started = time.time()

        class EntropyDecay(BaseCallback):
            """熵系数线性衰减(按累计环境步数,续练时接着衰减)。"""

            def _on_step(self):
                start = agent.HP_IMAGE["ent_coef"]
                frac = min(1.0, (env_steps0 + self.num_timesteps)
                           / agent.ENT_DECAY["steps"])
                self.model.ent_coef = \
                    start + frac * (agent.ENT_DECAY["end"] - start)
                return True

        class EvalCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode = episode0
                self.recent = deque(maxlen=20)
                self.best_return = best_return
                self.solved = False

            def _on_step(self):
                for info in self.locals["infos"]:
                    ep = info.get("episode")
                    if not ep:
                        continue
                    self.episode += 1
                    self.recent.append(ep["r"])
                    if self.episode % args.eval_every == 0 \
                            and not self._evaluate():
                        return False
                return True

            def _evaluate(self):
                eval_return, success_rate, avg_swingup = evaluate(
                    eval_env, agent, args.eval_episodes)
                train_avg = sum(self.recent) / max(1, len(self.recent))
                is_best = eval_return > self.best_return
                if is_best:
                    self.best_return = eval_return
                    agent.save(best, args.env,
                               extra={"eval_return": eval_return})
                agent.save(latest, args.env,
                           extra={"eval_return": eval_return})
                if success_rate >= args.solve_rate:
                    self.solved = True
                record = {
                    "time": round(time.time() - started, 1),
                    "episode": self.episode,
                    "env_steps": env_steps0 + self.num_timesteps,
                    "train_return": round(train_avg, 2),
                    "eval_return": round(eval_return, 2),
                    "success_rate": round(success_rate, 3),
                    "swingup_seconds": avg_swingup,
                    "best_return": round(self.best_return, 2),
                    "solved": self.solved,
                }
                with metrics_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                meta_path.write_text(json.dumps({
                    "env": args.env, "algo": agent.name,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **record,
                }, indent=2))
                print(f"回合 {self.episode:>5} | 步数 {record['env_steps']:>8} | "
                      f"训练回报 {train_avg:8.1f} | 评估回报 {eval_return:8.1f} | "
                      f"成功率 {success_rate:.0%}"
                      + (f" | 摆起 {avg_swingup}s" if avg_swingup else "")
                      + (" [BEST]" if is_best else "")
                      + ("  ✅ 已解决" if self.solved else ""))
                msg = progress_message(args.env, agent.name,
                                       record["env_steps"],
                                       time.time() - started, env_steps0,
                                       solved=self.solved)
                if msg:
                    print(msg)
                if self.solved and not args.forever:
                    print(f"评估成功率达到 {args.solve_rate:.0%},训练完成。"
                          f"最优模型: {best}")
                    return False
                return True

        callbacks = [EvalCallback()]
        if image_obs:
            callbacks.append(EntropyDecay())
        self.model.learn(total_timesteps=int(1e12),
                         callback=callbacks,
                         reset_num_timesteps=True)
