"""PPO(裁剪目标 + GAE),离散动作版。"""
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent, mlp


class PPOAgent(BaseAgent):
    name = "ppo"

    # gamma 0.995(视野约 10 秒):甩起→稳住→成功奖金的因果链长达数百步,
    # 0.99 时奖金折扣到几乎不可见。ent_coef 0.003:倒立是不稳定平衡,
    # 熵奖励太高会让策略在平衡点附近持续随机抖动,训练中体验不到长 hold。
    def __init__(self, obs_dim, n_actions, device="cpu",
                 lr=3e-4, gamma=0.995, lam=0.95, clip=0.2,
                 rollout=2048, epochs=10, minibatch=256,
                 ent_coef=0.003, vf_coef=0.5, max_grad_norm=0.5,
                 seed=None):
        super().__init__(obs_dim, n_actions, device)
        self.pi = mlp(obs_dim, n_actions, hidden=(128, 128)).to(self.device)
        self.v = mlp(obs_dim, 1, hidden=(128, 128)).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.pi.parameters()) + list(self.v.parameters()), lr=lr)
        if seed is not None:
            torch.manual_seed(seed)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.rollout, self.epochs, self.minibatch = rollout, epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm
        self._buf = []          # (obs, act, logp, value, reward, terminated)
        self._last = None       # act() 时缓存的 (logp, value)
        self._pending_metrics = None

    def _dist(self, obs_t):
        return torch.distributions.Categorical(logits=self.pi(obs_t))

    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self._dist(obs_t)
            if deterministic:
                a = dist.logits.argmax(dim=1)
            else:
                a = dist.sample()
            self._last = (float(dist.log_prob(a).item()),
                          float(self.v(obs_t).item()))
        return int(a.item())

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        logp, value = self._last
        # 截断时把末状态价值并入奖励,等价于 bootstrap
        if truncated:
            with torch.no_grad():
                nv = float(self.v(torch.as_tensor(
                    next_obs, device=self.device).unsqueeze(0)).item())
            reward = reward + self.gamma * nv
        self._buf.append((obs, action, logp, value, reward,
                          float(terminated or truncated)))
        if len(self._buf) >= self.rollout:
            last_v = 0.0
            if not (terminated or truncated):
                with torch.no_grad():
                    last_v = float(self.v(torch.as_tensor(
                        next_obs, device=self.device).unsqueeze(0)).item())
            self._pending_metrics = self._train(last_v)

    def update(self):
        m, self._pending_metrics = self._pending_metrics, None
        return m

    def _train(self, last_value):
        obs = torch.as_tensor(np.array([b[0] for b in self._buf]),
                              device=self.device)
        act = torch.as_tensor([b[1] for b in self._buf], device=self.device)
        logp_old = torch.as_tensor([b[2] for b in self._buf],
                                   dtype=torch.float32, device=self.device)
        values = np.array([b[3] for b in self._buf], dtype=np.float32)
        rewards = np.array([b[4] for b in self._buf], dtype=np.float32)
        dones = np.array([b[5] for b in self._buf], dtype=np.float32)
        self._buf = []

        # GAE
        adv = np.zeros_like(rewards)
        gae = 0.0
        next_v = last_value
        for t in range(len(rewards) - 1, -1, -1):
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
            next_v = values[t]
        ret = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.as_tensor(adv, device=self.device)
        ret = torch.as_tensor(ret, device=self.device)

        n = len(rewards)
        idx = np.arange(n)
        pi_losses, v_losses, entropies = [], [], []
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = idx[s:s + self.minibatch]
                dist = self._dist(obs[mb])
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                a_mb = adv[mb]
                pi_loss = -torch.min(
                    ratio * a_mb,
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_mb
                ).mean()
                v_loss = nn.functional.mse_loss(
                    self.v(obs[mb]).squeeze(1), ret[mb])
                ent = dist.entropy().mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.pi.parameters()) + list(self.v.parameters()),
                    self.max_grad_norm)
                self.opt.step()
                pi_losses.append(float(pi_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(ent.item()))
        return {"pi_loss": round(float(np.mean(pi_losses)), 4),
                "v_loss": round(float(np.mean(v_losses)), 4),
                "entropy": round(float(np.mean(entropies)), 4)}

    def state_dict(self):
        return {"pi": self.pi.state_dict(), "v": self.v.state_dict()}

    def load_state_dict(self, sd):
        self.pi.load_state_dict(sd["pi"])
        self.v.load_state_dict(sd["v"])
