"""Double DQN,带经验回放与 epsilon 贪心探索。"""
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent, mlp


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = terminated
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.terminated[idx])


class DQNAgent(BaseAgent):
    name = "dqn"

    def __init__(self, obs_dim, n_actions, device="cpu",
                 lr=3e-4, gamma=0.99, buffer_size=100_000, batch_size=128,
                 warmup=2_000, target_sync=1_000,
                 eps_start=1.0, eps_end=0.02, eps_decay_steps=150_000,
                 seed=None):
        super().__init__(obs_dim, n_actions, device)
        self.q = mlp(obs_dim, n_actions).to(self.device)
        self.q_target = mlp(obs_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_sync = target_sync
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.steps = 0
        self.updates = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, deterministic=False):
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        # 截断(超时)不算终止:目标值仍然 bootstrap
        self.buffer.add(obs, action, reward, next_obs, float(terminated))
        self.steps += 1

    def update(self):
        if self.buffer.size < max(self.warmup, self.batch_size):
            return None
        obs, act, rew, nxt, term = self.buffer.sample(self.batch_size, self.rng)
        obs = torch.as_tensor(obs, device=self.device)
        act = torch.as_tensor(act, device=self.device)
        rew = torch.as_tensor(rew, device=self.device)
        nxt = torch.as_tensor(nxt, device=self.device)
        term = torch.as_tensor(term, device=self.device)

        with torch.no_grad():
            best = self.q(nxt).argmax(dim=1, keepdim=True)        # Double DQN
            q_next = self.q_target(nxt).gather(1, best).squeeze(1)
            target = rew + self.gamma * (1.0 - term) * q_next
        q_pred = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_pred, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {"loss": float(loss.item()), "epsilon": round(self.epsilon, 3)}

    def state_dict(self):
        return {"q": self.q.state_dict()}

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q"])
