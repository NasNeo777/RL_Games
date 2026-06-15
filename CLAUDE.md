# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

强化学习实验室 (RL lab): pick a game, run one command, watch an AI learn it in the browser with live demo animations and training curves. Python 3.13, PyTorch (hand-written DQN), Stable-Baselines3 (PPO), zero-dependency stdlib server + single-file web UI.

## Commands

All Python runs through the project venv at `.venv/bin/python` (no global Python).

```bash
# Train + demo server together (opens browser at localhost:8000)
./start.sh --env snake --algo dqn
./start.sh                          # default: double_pendulum + ppo
./start.sh --env 2048 --algo td2048 --forever   # keep optimizing after solved
./start.sh --env tetris --algo dqn --restart    # retrain from scratch (old dir backed up to *_old)

# Run the two halves separately
.venv/bin/python -m rl_lab.train --env tetris --algo dqn
.venv/bin/python -m rl_lab.server --port 8000

PORT=8888 ./start.sh                 # change demo port
NO_OPEN=1 ./start.sh                 # don't auto-open browser

# Install deps (some games are opt-in, commented out in requirements.txt)
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install flappy-bird-gymnasium      # for --env flappy_bird
.venv/bin/pip install gym-super-mario-bros gym   # for --env mario
```

By default training **auto-resumes** from `latest.pt` if the run dir exists (step/episode counts continue from the checkpoint). `--restart` forces a fresh start. There is no test suite, linter, or build step.

> ⚠️ After editing `envs/` or `algos/` code, **restart the demo server** — it imports those modules and won't pick up changes otherwise.

## Architecture

Two decoupled processes communicate only through files in `runs/<env>_<algo>/`:

- **`train.py`** writes checkpoints (`latest.pt` every eval, `best.pt` on new high), `metrics.jsonl` (one JSON line per eval — the curve data source), and `meta.json` (status summary).
- **`server.py`** (stdlib `http.server` only, no deps) reads those files on demand and serves the web UI. It re-runs a full episode to produce demo frames, cached per checkpoint mtime. APIs: `GET /api/runs`, `GET /api/metrics?run=&limit=N`, `GET /api/demo?run=&which=best|latest`.

### Two registries drive everything

- **Envs**: `rl_lab/envs/__init__.py` `ENVS` dict maps name → class. `make_env(name)`.
- **Algos**: `rl_lab/algos/__init__.py` `ALGOS` dict maps name → class. `make_agent(name, obs_dim, n_actions)`.

`train.py` and `server.py` are env/algo-agnostic — they only touch the base interfaces.

### Two kinds of algorithms (the key control-flow split)

`train.py` checks `agent.trains_itself`:

1. **Manual loop** (DQN, td2048, custom PPO): `train.py` owns the loop — `agent.act → env.step → agent.observe → agent.update()` per step, evaluates every `--eval-every` episodes, saves checkpoints, appends metrics.
2. **Self-training** (`SB3PPOAgent`, `trains_itself=True`): `train.py` calls `agent.train_loop(...)` once and the external library (Stable-Baselines3) runs its own loop. Evaluation, checkpointing, and metrics are emitted from SB3 callbacks in the **same file format** so the server can't tell the difference. SB3 needs gym envs, so `envs/to_gym.py` wraps a `BaseEnv` into a standard `gymnasium.Env`.

Implement a new algo against `algos/base.py` `BaseAgent` (`act / observe / update / state_dict / load_state_dict`) and register it. To wrap an external library, set `trains_itself = True` and implement `train_loop()` (see `algos/sb3_ppo.py`).

### Two kinds of environments

Implement `envs/base.py` `BaseEnv` (`reset / step` 5-tuple, gym-style; `render_spec()` for the frontend). Key class attributes:
- `obs_dim` / `n_actions`: discrete action spaces only.
- `obs_shape`: `None` for vector obs (float32, `obs_dim`-wide); `(C, H, W)` uint8 for **image obs**. Image-obs envs (`mario`, `jump_pixels`) only work with `--algo ppo` (SB3 auto-switches to a CNN); `train.py` raises if you pair image obs with a vector-only algo.
- `parallel_mode`: `"subproc"` (true multiprocess, heavy single steps) or `"dummy"` (in-process, ultralight steps) — controls how SB3 vectorizes envs.

Two integration paths:
- **Gymnasium-backed** (recommended, no game logic): subclass `envs/gym_adapter.py` `GymEnv`, set `env_id` / `import_module` / `max_steps` / `is_success(info)`. The adapter grabs `rgb_array` frames → JPEG and the frontend plays them with the generic `video` renderer — **no frontend code needed** (this is how `flappy_bird` works).
- **Custom physics/logic** (`snake`, `2048`, `tetris`, `jump`, `double_pendulum`): record state frames in `step` when `self.record` is on, then add a matching canvas renderer to `RENDERERS` in `web/index.html`.

### Other modules

- `evaluation.py` — `evaluate(env, agent, n)` returns `(eval_return, success_rate, avg_swingup)`; `success_rate` comes from each env's `is_success(info)`.
- `progress.py` — startup/progress messages estimating "time to mastery" from per-env step budgets measured in this repo.
- `base_agent_loader.py` — rebuilds env+agent from a checkpoint for the demo server; handles the legacy `ppo` → `ppo_custom` checkpoint compatibility (pre-SB3 checkpoints).
- `web/index.html` — single-file, zero-dependency UI. `RENDERERS` map demo frame data to canvas drawings per env type.

### Algorithm notes

- `dqn`: hand-written Double DQN (replay + ε-greedy), all vector-obs games.
- `ppo`: Stable-Baselines3 PPO via the `sb3_ppo.py` adapter; only choice for image obs. `HP_IMAGE` holds image-specific hyperparams.
- `td2048`: **2048-only** afterstate TD(0) with N-tuple lookup tables (no neural net) — evaluates the board *after* the deterministic merge but *before* the random spawn, sidestepping the spawn randomness that cripples generic Q-learning here. ~20× DQN's score at equal steps.
- `ppo_custom`: deprecated hand-written PPO, kept only for loading old checkpoints.
