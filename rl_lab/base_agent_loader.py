"""从检查点重建 env + agent,供演示服务器使用。"""
from .algos import make_agent
from .algos.base import BaseAgent
from .envs import make_env


def load_agent_for_demo(ckpt_path):
    ckpt = BaseAgent.load_checkpoint(ckpt_path)
    env = make_env(ckpt["env"])
    agent = make_agent(ckpt["algo"], ckpt["obs_dim"], ckpt["n_actions"])
    agent.load_state_dict(ckpt["state_dict"])
    return env, agent, ckpt
