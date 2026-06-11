"""从检查点重建 env + agent,供演示服务器使用。"""
from .algos import make_agent
from .algos.base import BaseAgent
from .envs import make_env


def load_agent_for_demo(ckpt_path):
    ckpt = BaseAgent.load_checkpoint(ckpt_path)
    algo = ckpt["algo"]
    # 兼容:SB3 接入前的 "ppo" 检查点(权重是手写版的 pi/v)
    if algo == "ppo" and "sb3_zip" not in ckpt["state_dict"]:
        algo = "ppo_custom"
    env = make_env(ckpt["env"])
    agent = make_agent(algo, ckpt["obs_dim"], ckpt["n_actions"])
    agent.load_state_dict(ckpt["state_dict"])
    return env, agent, ckpt
