# 大蛇关卡游戏代码框架与交接文档

本文档按接手开发的顺序组织：

1. 先给出代码框架：下一步应该新增哪些文件、每个类怎么拆、环境怎么接入本仓库。
2. 再给出交接文档：当前状态、未完成事项、验证命令、风险点。
3. 最后保留原始规则与数学建模，作为实现时查阅的设计附录。

---

# 1. 代码框架

## 1.1 命名与接入位置

本仓库里已经有一个 `rl_lab/envs/snake.py`，它是“贪吃蛇”环境。这个新游戏不要继续叫 `snake`，建议命名为：

| 项目 | 建议命名 | 原因 |
| --- | --- | --- |
| 环境 key | `snake_gate` | 避免和现有贪吃蛇 `snake` 冲突 |
| 环境类 | `SnakeGateEnv` | 表达“数值门 + 大蛇 Boss” |
| 主文件 | `rl_lab/envs/snake_gate.py` | 纯 Python 数值环境 |
| 前端渲染器 | `snake_gate` | 对应 `render_spec()["type"]` |
| 训练目录 | `runs/snake_gate_dqn/`、`runs/snake_gate_ppo/` | 沿用本仓库训练产物规则 |

第一版只做“纯数值模拟 + 网页演示”，不要先做完整游戏画面。这样可以更快接入 DQN/PPO，并验证策略是否真的能学会打门、打石头、打大蛇。

---

## 1.2 推荐文件树

```text
rl_lab/
  envs/
    snake_gate.py          # 新增: 大蛇关卡核心环境
    __init__.py            # 修改: 注册 snake_gate
  web/
    index.html             # 修改: 增加 snake_gate 渲染器
  progress.py              # 可选修改: 增加 snake_gate_dqn / snake_gate_ppo 进度预算
README.md                  # 可选修改: 游戏厅里加“大蛇关卡”
docs/
  大蛇plan.md              # 本文档
```

如果后续配置变复杂，再拆：

```text
rl_lab/envs/snake_gate_config.py   # 关卡配置、默认数值、难度曲线
rl_lab/envs/snake_gate_policy.py   # 贪心 baseline / 规则 bot
```

MVP 阶段可以先把配置、环境、baseline 都放进 `snake_gate.py`，等跑通后再拆。

---

## 1.3 `rl_lab/envs/snake_gate.py` 骨架

本仓库自制环境继承 `BaseEnv`，约定是：

```python
obs = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

不是直接返回 Gymnasium 的 `(obs, info)`。如果给 SB3 PPO 用，本仓库已有 `BaseEnvToGym` 适配层。

建议骨架如下：

```python
"""大蛇关卡:射击升级门 + 障碍 + Boss 的纯数值 RL 环境。

动作:选择当前攻击哪个目标。
观测:玩家数值、门状态、障碍状态、Boss/基地血量、剩余时间。
目标:在基地被打爆或超时之前击败大蛇。
"""
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .base import BaseEnv


class GateType(IntEnum):
    ATTACK_ADD = 0
    ATTACK_MULT = 1
    FIRE_RATE_MULT = 2


class TargetKind(IntEnum):
    GATE = 0
    OBSTACLE = 1
    CHEST = 2
    BOSS = 3


@dataclass
class PlayerState:
    attack: float = 10.0
    fire_rate: float = 1.0

    @property
    def dps(self) -> float:
        return self.attack * self.fire_rate


@dataclass
class GateConfig:
    gate_type: GateType
    base_cost: float
    cost_growth: float
    base_reward: float
    reward_growth: float
    max_level: int = 20


@dataclass
class GateState:
    id: str
    config: GateConfig
    level: int = 0
    remaining_cost: float = 0.0
    current_reward: float = 0.0
    unlocked: bool = True

    def refresh(self) -> None:
        self.remaining_cost = self.config.base_cost * (
            self.config.cost_growth ** self.level
        )
        self.current_reward = self.config.base_reward * (
            self.config.reward_growth ** self.level
        )


@dataclass
class ObstacleState:
    id: str
    hp: float
    max_hp: float
    unlocked: bool = True
    cleared: bool = False
    unlock_targets: list[str] = field(default_factory=list)


@dataclass
class ChestState:
    id: str
    hp: float
    max_hp: float
    unlocked: bool = False
    opened: bool = False
    attack_bonus: float = 0.0
    fire_rate_mult: float = 1.0


@dataclass
class BossState:
    hp: float
    max_hp: float
    attack_damage: float
    attack_interval: float
    timer: float = 0.0
    unlocked: bool = False


@dataclass
class LevelConfig:
    dt: float = 0.1
    time_limit: float = 120.0
    base_hp: float = 1000.0
    gates: tuple[GateConfig, ...] = (
        GateConfig(GateType.ATTACK_ADD, 50, 1.8, 25, 1.35),
        GateConfig(GateType.ATTACK_MULT, 180, 2.2, 1.5, 1.05),
        GateConfig(GateType.FIRE_RATE_MULT, 120, 2.0, 1.35, 1.04),
    )
    obstacle_hps: tuple[float, ...] = (300, 900)
    chest_hps: tuple[float, ...] = (500,)
    boss_hp: float = 15000.0
    boss_attack_damage: float = 20.0
    boss_attack_interval: float = 1.0


class SnakeGateEnv(BaseEnv):
    """本仓库训练入口可直接使用的环境。"""

    max_steps = 1200
    parallel_mode = "dummy"

    def __init__(self, seed=None, config: LevelConfig | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.config = config or LevelConfig()
        self.n_gates = len(self.config.gates)
        self.n_obstacles = len(self.config.obstacle_hps)
        self.n_chests = len(self.config.chest_hps)
        self.n_actions = self.n_gates + self.n_obstacles + self.n_chests + 1
        self.obs_dim = self._calc_obs_dim()
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.steps = 0
        self.base_hp = self.config.base_hp
        self.player = PlayerState()
        self.gates = self._make_gates()
        self.obstacles = self._make_obstacles()
        self.chests = self._make_chests()
        self.boss = BossState(
            hp=self.config.boss_hp,
            max_hp=self.config.boss_hp,
            attack_damage=self.config.boss_attack_damage,
            attack_interval=self.config.boss_attack_interval,
            unlocked=False,
        )
        self.frames = []
        if self.record:
            self._record_frame(event="reset")
        return self._obs()

    def step(self, action):
        action = int(action)
        self.t += self.config.dt
        self.steps += 1

        event = None
        reward = -0.01
        invalid = not self._is_action_valid(action)
        if invalid:
            reward -= 1.0
            event = "invalid_action"
        else:
            event, shaped = self._apply_player_damage(action)
            reward += shaped

        base_damage = self._boss_tick()
        reward -= 0.1 * base_damage

        terminated = self.boss.hp <= 0 or self.base_hp <= 0
        truncated = (not terminated) and self.steps >= self.max_steps

        if self.boss.hp <= 0:
            reward += 1000.0
        elif self.base_hp <= 0:
            reward -= 1000.0
        elif truncated:
            reward -= 300.0

        info = {
            "success": self.boss.hp <= 0,
            "score": int(max(0, self.config.boss_hp - self.boss.hp)),
            "event": event,
            "attack": self.player.attack,
            "fire_rate": self.player.fire_rate,
            "dps": self.player.dps,
            "base_hp": self.base_hp,
            "boss_hp": self.boss.hp,
            "action_mask": self.action_mask(),
        }
        if self.record:
            self._record_frame(event=event)
        return self._obs(), float(reward), terminated, truncated, info

    def _make_gates(self):
        gates = []
        for i, cfg in enumerate(self.config.gates):
            gate = GateState(id=f"gate_{i}", config=cfg)
            gate.refresh()
            gates.append(gate)
        return gates

    def _make_obstacles(self):
        return [
            ObstacleState(id=f"stone_{i}", hp=hp, max_hp=hp, unlocked=(i == 0))
            for i, hp in enumerate(self.config.obstacle_hps)
        ]

    def _make_chests(self):
        return [
            ChestState(
                id=f"chest_{i}",
                hp=hp,
                max_hp=hp,
                unlocked=False,
                attack_bonus=50.0 * (i + 1),
            )
            for i, hp in enumerate(self.config.chest_hps)
        ]

    def _apply_player_damage(self, action):
        damage = self.player.dps * self.config.dt
        target = self._target_for_action(action)
        before_dps = self.player.dps

        if isinstance(target, GateState):
            target.remaining_cost -= damage
            if target.remaining_cost <= 0:
                self._resolve_gate(target)
                return "gate_upgraded", 0.01 * (self.player.dps - before_dps) + 10.0
            return "gate_damaged", 0.0

        if isinstance(target, ObstacleState):
            target.hp -= damage
            if target.hp <= 0 and not target.cleared:
                self._resolve_obstacle(target)
                return "obstacle_cleared", 20.0
            return "obstacle_damaged", 0.0

        if isinstance(target, ChestState):
            target.hp -= damage
            if target.hp <= 0 and not target.opened:
                self._resolve_chest(target)
                return "chest_opened", 30.0
            return "chest_damaged", 0.0

        if isinstance(target, BossState):
            target.hp -= damage
            return "boss_damaged", 0.001 * damage

        return "unknown", 0.0

    def _resolve_gate(self, gate: GateState) -> None:
        cfg = gate.config
        reward = gate.current_reward
        if cfg.gate_type == GateType.ATTACK_ADD:
            self.player.attack += reward
        elif cfg.gate_type == GateType.ATTACK_MULT:
            self.player.attack *= reward
        elif cfg.gate_type == GateType.FIRE_RATE_MULT:
            self.player.fire_rate *= reward

        gate.level += 1
        if gate.level >= cfg.max_level:
            gate.unlocked = False
        else:
            gate.refresh()

    def _resolve_obstacle(self, obstacle: ObstacleState) -> None:
        obstacle.cleared = True
        obstacle.hp = 0.0

        # MVP 规则:清掉全部石头后解锁 Boss；清掉第一个石头后解锁宝箱。
        if self.chests:
            self.chests[0].unlocked = True
        if all(o.cleared for o in self.obstacles):
            self.boss.unlocked = True

    def _resolve_chest(self, chest: ChestState) -> None:
        chest.opened = True
        chest.hp = 0.0
        self.player.attack += chest.attack_bonus
        self.player.fire_rate *= chest.fire_rate_mult

    def _boss_tick(self) -> float:
        if not self.boss.unlocked:
            return 0.0
        self.boss.timer += self.config.dt
        hits = int(self.boss.timer / self.boss.attack_interval)
        if hits <= 0:
            return 0.0
        self.boss.timer -= hits * self.boss.attack_interval
        damage = hits * self.boss.attack_damage
        self.base_hp = max(0.0, self.base_hp - damage)
        return damage

    def _target_for_action(self, action):
        if action < self.n_gates:
            return self.gates[action]
        action -= self.n_gates
        if action < self.n_obstacles:
            return self.obstacles[action]
        action -= self.n_obstacles
        if action < self.n_chests:
            return self.chests[action]
        return self.boss

    def _is_action_valid(self, action) -> bool:
        if action < 0 or action >= self.n_actions:
            return False
        target = self._target_for_action(action)
        if isinstance(target, GateState):
            return target.unlocked
        if isinstance(target, ObstacleState):
            return target.unlocked and not target.cleared
        if isinstance(target, ChestState):
            return target.unlocked and not target.opened
        if isinstance(target, BossState):
            return target.unlocked and target.hp > 0
        return False

    def action_mask(self):
        return [self._is_action_valid(a) for a in range(self.n_actions)]

    def _calc_obs_dim(self) -> int:
        # player: attack, fire_rate, dps, time_left, base_hp, boss_hp, boss_unlocked
        # gates: type, level, remaining_cost, reward, roi, unlocked
        # obstacles: hp, unlocked, cleared
        # chests: hp, unlocked, opened
        return 7 + self.n_gates * 6 + self.n_obstacles * 3 + self.n_chests * 3

    def _obs(self):
        values = [
            np.log1p(self.player.attack) / 10.0,
            np.log1p(self.player.fire_rate) / 5.0,
            np.log1p(self.player.dps) / 12.0,
            max(0.0, 1.0 - self.steps / self.max_steps),
            self.base_hp / self.config.base_hp,
            self.boss.hp / self.boss.max_hp,
            float(self.boss.unlocked),
        ]

        for gate in self.gates:
            roi = self._gate_roi(gate)
            values.extend([
                gate.config.gate_type / 2.0,
                gate.level / max(1, gate.config.max_level),
                np.log1p(max(0.0, gate.remaining_cost)) / 12.0,
                np.log1p(gate.current_reward) / 8.0,
                np.clip(roi, 0.0, 10.0) / 10.0,
                float(gate.unlocked),
            ])

        for obstacle in self.obstacles:
            values.extend([
                max(0.0, obstacle.hp) / obstacle.max_hp,
                float(obstacle.unlocked),
                float(obstacle.cleared),
            ])

        for chest in self.chests:
            values.extend([
                max(0.0, chest.hp) / chest.max_hp,
                float(chest.unlocked),
                float(chest.opened),
            ])

        return np.array(values, dtype=np.float32)

    def _gate_roi(self, gate: GateState) -> float:
        if gate.remaining_cost <= 0:
            return 0.0
        old_dps = self.player.dps
        if gate.config.gate_type == GateType.ATTACK_ADD:
            new_dps = (self.player.attack + gate.current_reward) * self.player.fire_rate
        elif gate.config.gate_type == GateType.ATTACK_MULT:
            new_dps = (self.player.attack * gate.current_reward) * self.player.fire_rate
        else:
            new_dps = self.player.attack * (self.player.fire_rate * gate.current_reward)
        return (new_dps - old_dps) / gate.remaining_cost

    def _record_frame(self, event=None):
        self.frames.append({
            "t": round(self.t, 2),
            "attack": round(self.player.attack, 2),
            "fireRate": round(self.player.fire_rate, 2),
            "dps": round(self.player.dps, 2),
            "baseHp": round(self.base_hp, 2),
            "bossHp": round(max(0.0, self.boss.hp), 2),
            "bossMaxHp": self.boss.max_hp,
            "bossUnlocked": self.boss.unlocked,
            "gates": [
                {
                    "type": int(g.config.gate_type),
                    "level": g.level,
                    "cost": round(max(0.0, g.remaining_cost), 2),
                    "reward": round(g.current_reward, 2),
                    "unlocked": g.unlocked,
                }
                for g in self.gates
            ],
            "obstacles": [
                {
                    "hp": round(max(0.0, o.hp), 2),
                    "maxHp": o.max_hp,
                    "unlocked": o.unlocked,
                    "cleared": o.cleared,
                }
                for o in self.obstacles
            ],
            "chests": [
                {
                    "hp": round(max(0.0, c.hp), 2),
                    "maxHp": c.max_hp,
                    "unlocked": c.unlocked,
                    "opened": c.opened,
                }
                for c in self.chests
            ],
            "event": event,
        })

    def render_spec(self):
        return {
            "type": "snake_gate",
            "frame_dt": self.config.dt,
            "gateTypes": ["攻击+", "攻击x", "攻速x"],
        }
```

---

## 1.4 环境注册

修改 `rl_lab/envs/__init__.py`：

```python
from .snake_gate import SnakeGateEnv

ENVS = {
    # ...
    "snake_gate": SnakeGateEnv,
}
```

注册后可以直接跑：

```bash
.venv/bin/python -m rl_lab.train --env snake_gate --algo dqn --restart
```

---

## 1.5 贪心 baseline 骨架

在 RL 之前先写一个规则 bot，验证关卡数值不是死局。MVP 可以先放在 `snake_gate.py` 末尾，稳定后拆到 `snake_gate_policy.py`。

```python
def greedy_action(env: SnakeGateEnv) -> int:
    """选择当前最值得攻击的目标。用于调参和生成 imitation 数据。"""
    boss_action = env.n_actions - 1

    best_gate_action = None
    best_roi = -1.0
    for i, gate in enumerate(env.gates):
        if not env._is_action_valid(i):
            continue
        roi = env._gate_roi(gate)
        if roi > best_roi:
            best_roi = roi
            best_gate_action = i

    # ROI 足够高时先升级；阈值是调参旋钮，不是数学真理。
    if best_gate_action is not None and best_roi >= 0.05:
        return best_gate_action

    if env.boss.unlocked:
        return boss_action

    # Boss 未解锁时，优先清石头推进地图，再开宝箱。
    first_chest_action = None
    for action in range(env.n_actions):
        if not env._is_action_valid(action):
            continue
        target = env._target_for_action(action)
        if isinstance(target, ObstacleState):
            return action
        if isinstance(target, ChestState) and first_chest_action is None:
            first_chest_action = action

    if first_chest_action is not None:
        return first_chest_action

    if best_gate_action is not None:
        return best_gate_action

    return int(np.argmax(env.action_mask()))
```

建议先用这个 bot 连跑 100 局。如果贪心都完全打不过，先调数值，不要急着上 RL。

---

## 1.6 前端渲染器骨架

`server.py` 已经会把 `env.frames` 和 `env.render_spec()` 发给网页。只需要在 `rl_lab/web/index.html` 的 `RENDERERS` 中增加：

```javascript
snake_gate(ctx, W, H, spec, frames, fIdx, frac) {
  const f = frames[fIdx];
  ctx.clearRect(0, 0, W, H);

  const pad = 20;
  const laneY = H * 0.55;
  const panelW = Math.min(W - pad * 2, 760);
  const ox = (W - panelW) / 2;

  ctx.fillStyle = "#e6edf3";
  ctx.font = "bold 18px -apple-system, 'PingFang SC', sans-serif";
  ctx.fillText(`攻击 ${f.attack}  攻速 ${f.fireRate}  DPS ${f.dps}`, ox, 30);
  ctx.fillText(`基地 ${f.baseHp}  大蛇 ${f.bossHp}/${f.bossMaxHp}`, ox, 56);

  // 门
  f.gates.forEach((g, i) => {
    const x = ox + i * 120;
    ctx.fillStyle = g.unlocked ? "#1f6feb" : "#30363d";
    ctx.fillRect(x, laneY - 110, 90, 86);
    ctx.fillStyle = "#fff";
    ctx.font = "12px -apple-system, 'PingFang SC', sans-serif";
    const label = spec.gateTypes[g.type] || "门";
    ctx.fillText(`${label} ${g.reward}`, x + 8, laneY - 82);
    ctx.fillText(`Lv ${g.level}`, x + 8, laneY - 58);
    ctx.fillText(`成本 ${g.cost}`, x + 8, laneY - 34);
  });

  // 石头
  f.obstacles.forEach((o, i) => {
    const x = ox + i * 110;
    ctx.fillStyle = o.cleared ? "#2ea043" : o.unlocked ? "#8b949e" : "#30363d";
    ctx.beginPath();
    ctx.roundRect(x, laneY + 20, 82, 54, 8);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.fillText(o.cleared ? "已清除" : `${o.hp}/${o.maxHp}`, x + 8, laneY + 52);
  });

  // 宝箱
  f.chests.forEach((c, i) => {
    const x = ox + panelW - 220 + i * 90;
    ctx.fillStyle = c.opened ? "#2ea043" : c.unlocked ? "#d29922" : "#30363d";
    ctx.fillRect(x, laneY + 12, 68, 52);
    ctx.fillStyle = "#fff";
    ctx.fillText(c.opened ? "已开" : "宝箱", x + 10, laneY + 43);
  });

  // 大蛇 Boss
  const bossX = ox + panelW - 150;
  ctx.fillStyle = f.bossUnlocked ? "#f85149" : "#30363d";
  ctx.beginPath();
  ctx.roundRect(bossX, laneY - 125, 130, 92, 16);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 15px -apple-system, 'PingFang SC', sans-serif";
  ctx.fillText("大蛇", bossX + 45, laneY - 88);
  ctx.font = "12px -apple-system, 'PingFang SC', sans-serif";
  ctx.fillText(f.bossUnlocked ? `${f.bossHp}/${f.bossMaxHp}` : "未出现", bossX + 16, laneY - 58);

  if (f.event) {
    ctx.fillStyle = "#8b98a5";
    ctx.fillText(`事件: ${f.event}`, ox, H - 20);
  }
}
```

这个画法先保证能看清数值变化。美术可以后补，训练验证不等美术。

---

## 1.7 Observation / Action 约定

### Action

动作是离散整数：

| 范围 | 含义 |
| --- | --- |
| `0 .. n_gates-1` | 攻击第 N 个门 |
| `n_gates .. n_gates+n_obstacles-1` | 攻击第 N 个石头 |
| 接着的 `n_chests` 个动作 | 攻击宝箱 |
| 最后一个动作 | 攻击大蛇 |

第一版不要让动作数量动态变化。目标未解锁时，动作仍然存在，但 `action_mask` 为 `False`，误选给小惩罚。

### Observation

第一版用固定长度向量：

```text
player:
  log_attack, log_fire_rate, log_dps, time_left, base_hp, boss_hp, boss_unlocked

per gate:
  gate_type, level, remaining_cost, reward, roi, unlocked

per obstacle:
  hp_ratio, unlocked, cleared

per chest:
  hp_ratio, unlocked, opened
```

注意：攻击、DPS、成本、奖励容易指数爆炸，进入观测前用 `log1p` 压缩。

---

## 1.8 训练路线

Step 1：环境自测。

```bash
.venv/bin/python -m py_compile rl_lab/envs/snake_gate.py
```

再临时写一个小脚本或用 REPL 跑随机策略，确认 `reset/step` 不报错、无 NaN、回合能结束。

Step 2：贪心 bot 调参。

目标是简单配置下贪心能稳定击败大蛇。否则说明门成本、奖励、Boss 血量或基地血量不合理。

Step 3：DQN 训练。

```bash
.venv/bin/python -m rl_lab.train --env snake_gate --algo dqn --restart --eval-every 20 --eval-episodes 10
```

Step 4：PPO 对照。

```bash
.venv/bin/python -m rl_lab.train --env snake_gate --algo ppo --restart --eval-every 20 --eval-episodes 10
```

Step 5：接网页演示。

```bash
.venv/bin/python -m rl_lab.server --port 8000
```

打开 `http://localhost:8000`，确认能看到门、石头、宝箱和大蛇的状态变化。

---

# 2. 交接文档

## 2.1 当前状态

当前已经落地第一版代码框架：`snake_gate` 环境可以被训练入口识别，也能给网页演示输出 `frames`。这一版重点是纯数值环境闭环，还不是完整游戏美术版。

已有基础：

| 已有内容 | 位置 | 说明 |
| --- | --- | --- |
| 大蛇关卡环境 | `rl_lab/envs/snake_gate.py` | 已实现 reset/step、奖励、录像帧、贪心 baseline |
| 自制环境基类 | `rl_lab/envs/base.py` | 环境实现必须继承 `BaseEnv` |
| 环境注册表 | `rl_lab/envs/__init__.py` | 已注册 `"snake_gate": SnakeGateEnv` |
| 训练入口 | `rl_lab/train.py` | 会创建 env、agent，写入 `runs/<env>_<algo>/` |
| 演示服务 | `rl_lab/server.py` | 读取 checkpoint，开启 `env.record`，返回 frames |
| 网页渲染 | `rl_lab/web/index.html` | 已新增 `RENDERERS.snake_gate` 和训练预算 |
| 训练进度提示 | `rl_lab/progress.py` | 已新增 `snake_gate_dqn` / `snake_gate_ppo` 估算预算 |
| Gymnasium 适配 | `rl_lab/envs/to_gym.py` | SB3 PPO 通过适配层使用自制环境 |

重要约定：

```python
BaseEnv.reset(seed=None) -> obs
BaseEnv.step(action) -> obs, reward, terminated, truncated, info
```

不要在 `SnakeGateEnv.reset()` 里返回 `(obs, info)`，否则现有 `train.py`、`server.py` 会不匹配。

---

## 2.2 下一位接手者的任务清单

### 必做

1. 用 DQN 跑一轮短训，确认 `runs/snake_gate_dqn/metrics.jsonl`、`latest.pt` 能生成。
2. 打开网页演示，确认 `snake_gate` 录像帧显示正常。
3. 根据训练曲线调门成本、倍率、Boss 血量和奖励权重。
4. 决定是否接入真正的 action mask 算法；当前 DQN/PPO 不会自动消费 `info["action_mask"]`。

### 可选

1. 在 `README.md` 游戏厅里增加“大蛇关卡”的入口。
2. 把关卡数值拆到 `snake_gate_config.py`，方便后续多难度。
3. 把贪心策略拆到 `snake_gate_policy.py`，用于模仿学习或回归测试。
4. 增加简单回归测试，固定贪心策略应能通关。

---

## 2.3 验收标准

环境层验收：

| 项目 | 标准 |
| --- | --- |
| reset | 返回 `np.float32`，shape 等于 `env.obs_dim` |
| step | 返回五元组，reward 是 float，info 含 `success` |
| 数值 | 观测无 NaN/Inf，DPS 不在前几级爆炸 |
| 回合结束 | 随机策略、贪心策略都能自然 terminated 或 truncated |
| 胜负 | 击败大蛇时 `info["success"] == True` |

训练层验收：

| 项目 | 标准 |
| --- | --- |
| DQN | 能创建 `runs/snake_gate_dqn/latest.pt` |
| metrics | `metrics.jsonl` 有 `eval_return`、`success_rate` |
| server | `/api/demo?run=snake_gate_dqn&which=latest` 能返回 frames |
| 前端 | 页面能画出门、石头、宝箱、Boss、关键数值 |

---

## 2.4 推荐验证命令

实现后按顺序跑：

```bash
.venv/bin/python -m py_compile rl_lab/envs/snake_gate.py
```

```bash
.venv/bin/python - <<'PY'
from rl_lab.envs import make_env

env = make_env("snake_gate", seed=0)
obs = env.reset(seed=0)
print("obs", obs.shape, obs.dtype, "actions", env.n_actions)
total = 0
for i in range(1000):
    action = i % env.n_actions
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        print("done", i + 1, total, info)
        break
else:
    print("not done", total, info)
PY
```

```bash
.venv/bin/python -m rl_lab.train --env snake_gate --algo dqn --restart --eval-every 5 --eval-episodes 3
```

```bash
.venv/bin/python -m rl_lab.server --port 8000
```

---

## 2.5 已知风险

1. 当前算法不会自动使用 action mask。第一版要么让未解锁目标给小惩罚，要么保持所有目标从一开始都可攻击；不要以为 `info["action_mask"]` 已经生效。
2. 乘法门会导致数值爆炸。观测必须 `log1p`，并限制 `max_level`。
3. 奖励函数如果过度奖励 DPS，Agent 可能一直刷门不打 Boss。必须保留时间惩罚、Boss 伤害奖励、通关大奖励。
4. 如果 Boss 太晚解锁，Agent 早期几乎看不到通关奖励。MVP 建议只放 2 个门、1 个石头、1 个 Boss。
5. 如果 invalid action 惩罚太大，DQN 可能先学会保守不探索；如果太小，又会浪费很多步。建议从 `-1.0` 开始调。
6. 前端渲染只用于演示，不应该影响训练逻辑。所有胜负和奖励都必须在环境里闭环。

---

## 2.6 建议第一版默认数值

| 参数 | 建议值 |
| --- | --- |
| `dt` | `0.1` |
| `max_steps` | `1200` |
| 初始攻击 | `10` |
| 初始攻速 | `1.0` |
| 基地血量 | `1000` |
| 大蛇血量 | `15000` |
| 大蛇攻击 | 每 `1.0s` 打 `20` |
| 门数量 | 3 |
| 石头数量 | 2 |
| 宝箱数量 | 1 |

第一版目标不是好玩，而是“能学会”。等 DQN/PPO 能稳定通关，再提高成本增长、增加石头、增加宝箱和路线选择。

---

# 3. 规则与数学设计附录

下面保留原始**开发文档 v0.1**。目标是两个：

1. **程序员能根据它搭出这个游戏原型**；
2. **后续能写一个强化学习 Agent 自动玩，并训练到通关。**

---

# 大蛇关卡游戏开发文档 v0.1

## 1. 游戏定位

本游戏是一个：

[
\boxed{\text{射击升级}+\text{数值门}+\text{障碍破坏}+\text{最终 Boss/大蛇战}}
]

玩家控制一个角色持续攻击屏幕上的目标。目标包括：

1. 升级门；
2. 石头/障碍；
3. 宝箱；
4. 最终大蛇 Boss。

核心玩法是：

> 玩家通过攻击升级门，把门的负数成本打成正数，获得攻击力、攻速、倍率等强化；强化后继续打更高级的门和障碍，最后击败大蛇或守住终点。

---

# 2. 核心游戏规则

## 2.1 玩家属性

玩家有两个最核心属性：

| 变量  | 含义       |
| --- | -------- |
| (A) | 攻击力      |
| (F) | 攻击速度     |
| (D) | 每秒输出，DPS |

核心公式：

[
D=A \times F
]

例如：

[
A=100,\quad F=5
]

则：

[
D=500
]

玩家每秒能造成 500 点有效伤害。

---

## 2.2 升级门规则

每个门由两部分组成：

上方是奖励，例如：

[
\text{攻击}+100
]

[
\text{攻击}\times2
]

[
\text{攻速}\times5
]

下方是当前解锁成本，例如：

[
-50,\quad -200,\quad -12
]

负数表示还没解锁。玩家攻击门后，门的数值逐渐增加。

例如一个门当前是：

[
-50
]

每受到 10 点有效伤害，数值增加 10：

[
-50 \rightarrow -40 \rightarrow -30 \rightarrow \cdots \rightarrow 0 \rightarrow 10
]

当门达到正值或成本归零时，门解锁。

解锁后立即获得上方奖励。

---

## 2.3 门解锁后升级

每次打完一个门，门不会消失，而是刷新成更强版本。

例如：

第 1 级：

[
\text{攻击}+100,\quad 成本50
]

打完后变成第 2 级：

[
\text{攻击}+250,\quad 成本300
]

再打完变成第 3 级：

[
\text{攻击}+600,\quad 成本1500
]

所以每个门都有等级：

[
L_i=0,1,2,3,\cdots
]

等级越高：

[
\text{奖励越强，成本也越大}
]

---

## 2.4 门的类型

游戏中至少设计三类核心门。

### 1. 攻击加法门

显示：

[
\text{攻击}+R
]

效果：

[
A \leftarrow A+R
]

例如：

[
A=100
]

打完：

[
\text{攻击}+100
]

则：

[
A=200
]

---

### 2. 攻击倍率门

显示：

[
\text{攻击}\times M
]

效果：

[
A \leftarrow A \times M
]

例如：

[
A=200
]

打完：

[
\text{攻击}\times2
]

则：

[
A=400
]

---

### 3. 攻速倍率门

显示：

[
\text{攻速}\times Q
]

效果：

[
F \leftarrow F \times Q
]

例如：

[
F=2
]

打完：

[
\text{攻速}\times5
]

则：

[
F=10
]

---

# 3. 石头、宝箱和大蛇规则

## 3.1 石头/障碍

地图上方有很多石头，每个石头有血量：

[
H_j
]

例如：

[
450,\quad 750,\quad 2550,\quad 13500,\quad 22500
]

玩家打掉石头后可以：

1. 解锁新区域；
2. 获得金币；
3. 打开通往大蛇的路径；
4. 解锁宝箱。

打掉石头所需时间：

[
t_j=\frac{H_j}{D}
]

其中 (D) 是当前 DPS。

---

## 3.2 宝箱

宝箱也可以视为特殊目标。

宝箱有开启成本：

[
C_{\text{chest}}
]

玩家攻击宝箱，成本归零后开启。

宝箱奖励可以是：

1. 攻击力；
2. 攻速；
3. 金币；
4. 临时技能；
5. 一次性大伤害；
6. 解锁新门。

---

## 3.3 大蛇 Boss

大蛇是最终目标。

大蛇有：

| 变量         | 含义            |
| ---------- | ------------- |
| (H_s)      | 大蛇血量          |
| (D_s)      | 大蛇攻击力         |
| (F_s)      | 大蛇攻击频率        |
| (HP_g)     | 玩家需要守住的门/基地血量 |
| (T_{\max}) | 关卡最大时间        |

胜利条件：

[
H_s \leq 0
]

失败条件可以有两种：

[
HP_g \leq 0
]

或者：

[
t > T_{\max}
]

也就是说，玩家需要在基地被大蛇打爆之前击败大蛇。

---

# 4. 数学建模

## 4.1 游戏状态

游戏在任意时刻的状态可以表示为：

[
S_t=(A_t,F_t,L_1,L_2,\ldots,L_m,H_1,H_2,\ldots,H_n,H_s,HP_g,t)
]

其中：

| 变量     | 含义             |
| ------ | -------------- |
| (A_t)  | 当前攻击力          |
| (F_t)  | 当前攻速           |
| (L_i)  | 第 (i) 个门的等级    |
| (H_j)  | 第 (j) 个石头的剩余血量 |
| (H_s)  | 大蛇剩余血量         |
| (HP_g) | 基地/门血量         |
| (t)    | 当前时间           |

当前输出为：

[
D_t=A_tF_t
]

---

## 4.2 门成本模型

第 (i) 个门在等级 (L_i) 时的成本：

[
C_i(L_i)=C_{i0}\gamma_i^{L_i}
]

其中：

| 变量         | 含义     |
| ---------- | ------ |
| (C_{i0})   | 初始成本   |
| (\gamma_i) | 成本增长倍率 |
| (L_i)      | 当前等级   |

例如：

[
C_{i0}=50,\quad \gamma_i=2
]

则：

[
C_i(0)=50
]

[
C_i(1)=100
]

[
C_i(2)=200
]

[
C_i(3)=400
]

---

## 4.3 门奖励模型

第 (i) 个门在等级 (L_i) 时的奖励：

[
R_i(L_i)=R_{i0}\lambda_i^{L_i}
]

其中：

| 变量          | 含义     |
| ----------- | ------ |
| (R_{i0})    | 初始奖励   |
| (\lambda_i) | 奖励增长倍率 |

例如攻击加法门：

[
R_{i0}=100,\quad \lambda_i=2
]

则：

[
R_i(0)=100
]

[
R_i(1)=200
]

[
R_i(2)=400
]

---

## 4.4 打门所需时间

如果当前 DPS 为：

[
D_t=A_tF_t
]

第 (i) 个门当前成本为：

[
C_i(L_i)
]

则打完这个门所需时间为：

[
\boxed{
\tau_i=\frac{C_i(L_i)}{A_tF_t}
}
]

这是游戏建模中最重要的公式之一。

---

# 5. 门收益分析结果

## 5.1 攻击加法门收益

如果门奖励是：

[
\text{攻击}+R
]

则：

[
A' = A+R
]

[
F'=F
]

原 DPS：

[
D=AF
]

新 DPS：

[
D'=(A+R)F
]

DPS 提升：

[
\Delta D=D'-D=RF
]

所以：

[
\boxed{
\Delta D_{\text{攻击加法}}=RF
}
]

结论：

> 攻击加法门的收益取决于当前攻速。攻速越高，攻击 +100 越值钱。

---

## 5.2 攻击倍率门收益

如果门奖励是：

[
\text{攻击}\times M
]

则：

[
A'=AM
]

[
F'=F
]

新 DPS：

[
D'=AMF
]

DPS 提升：

[
\Delta D=AMF-AF
]

[
\boxed{
\Delta D_{\text{攻击倍率}}=(M-1)AF=(M-1)D
}
]

结论：

> 攻击倍率门越到后期越强，因为当前 DPS 越高，倍率收益越大。

---

## 5.3 攻速倍率门收益

如果门奖励是：

[
\text{攻速}\times Q
]

则：

[
A'=A
]

[
F'=FQ
]

新 DPS：

[
D'=AFQ
]

DPS 提升：

[
\Delta D=AFQ-AF
]

[
\boxed{
\Delta D_{\text{攻速倍率}}=(Q-1)AF=(Q-1)D
}
]

结论：

> 攻速倍率门也是乘法门，通常优先级很高。尤其是低成本高倍率攻速门，会带来极强滚雪球。

---

# 6. 性价比模型

为了让 Agent 判断先打哪个门，可以定义性价比：

[
\rho_i=\frac{\Delta D_i}{C_i}
]

其中：

| 变量           | 含义                 |
| ------------ | ------------------ |
| (\Delta D_i) | 打完第 (i) 个门后 DPS 提升 |
| (C_i)        | 当前门成本              |
| (\rho_i)     | 单位成本换来的 DPS 提升     |

[
\rho_i
]

越大，说明越值得先打。

---

## 6.1 回本时间模型

也可以定义回本时间：

[
B_i=\frac{C_i}{\Delta D_i}
]

回本时间越短，越值得先打。

但更严格的判断是：

假设后面还剩总工作量：

[
W
]

比如还要打石头、宝箱、大蛇，合计血量为 (W)。

不打这个门，后续耗时：

[
\frac{W}{D}
]

打完这个门后，后续耗时：

[
\frac{W}{D'}
]

这个门节省的时间是：

[
\frac{W}{D}-\frac{W}{D'}
]

打门本身花费时间：

[
\frac{C_i}{D}
]

因此，门值得打的条件是：

[
\boxed{
\frac{W}{D}-\frac{W}{D'}>\frac{C_i}{D}
}
]

翻译成人话：

> 这个门后面帮你省下来的时间，要大于你现在打它花掉的时间。

---

# 7. 最优目标函数

如果目标是最快通关，设最终大蛇血量为：

[
H_s
]

如果当前已经升级了若干门，最终 DPS 是：

[
D_n=A_nF_n
]

那么通关总时间为：

[
T_{\text{clear}}=t_n+\frac{H_s}{A_nF_n}
]

目标函数：

[
\boxed{
\min T_{\text{clear}}
=====================

\min
\left[
t_n+\frac{H_s}{A_nF_n}
\right]
}
]

其中：

[
t_n
]

是前面打门、打石头、开宝箱消耗的时间。

这个模型说明：

> 升级不是越多越好。升级太多会浪费时间；升级太少又打不动大蛇。最优策略是在“升级耗时”和“后续节省时间”之间找平衡。

---

# 8. 基础策略结论

根据数学建模，可以得到几个初步策略。

## 8.1 低成本高倍率攻速门优先

例如：

[
\text{攻速}\times5,\quad 成本12
]

这种门通常极强，因为它会让后续所有门、石头和 Boss 都更快被打掉。

---

## 8.2 攻击加法门适合前中期

例如：

[
\text{攻击}+100
]

当前攻击力低的时候，直接加攻击很明显。

但是后期如果攻击力已经很高，攻击加法门可能不如倍率门。

---

## 8.3 攻击倍率门适合当前攻击力较高时

例如：

[
\text{攻击}\times2
]

当 (A) 很低时，(\times2) 收益不一定比 (+100) 高。

当 (A) 已经较高时，(\times2) 会非常强。

---

## 8.4 最终 Boss 不一定最后才打

如果继续升级的回本时间太长，就应该停止打门，直接攻击大蛇。

判断标准：

[
\text{继续升级节省的时间} < \text{升级消耗的时间}
]

此时应该打 Boss。

---

# 9. 游戏开发结构

## 9.1 核心模块

建议游戏拆成以下模块：

| 模块             | 功能                 |
| -------------- | ------------------ |
| PlayerSystem   | 管理攻击力、攻速、射击        |
| TargetSystem   | 管理可攻击目标            |
| GateSystem     | 管理升级门              |
| UpgradeSystem  | 应用门奖励              |
| ObstacleSystem | 管理石头和障碍            |
| ChestSystem    | 管理宝箱               |
| BossSystem     | 管理大蛇               |
| LevelSystem    | 管理关卡流程             |
| RewardSystem   | 管理金币、掉落、奖励         |
| RLInterface    | 给强化学习 Agent 提供环境接口 |

---

## 9.2 推荐数据结构

### PlayerConfig

```json
{
  "baseAttack": 10,
  "baseFireRate": 1.0,
  "attackRange": 999,
  "targetMode": "manual_or_agent"
}
```

---

### GateConfig

```json
{
  "id": "atk_add_01",
  "type": "ATTACK_ADD",
  "baseCost": 50,
  "costGrowth": 2.0,
  "baseReward": 100,
  "rewardGrowth": 1.8,
  "maxLevel": 20
}
```

---

### Gate 类型

```json
{
  "ATTACK_ADD": "攻击力加法",
  "ATTACK_MULT": "攻击力倍率",
  "FIRE_RATE_MULT": "攻速倍率"
}
```

---

### ObstacleConfig

```json
{
  "id": "stone_01",
  "hp": 450,
  "reward": {
    "gold": 10
  },
  "unlockTargets": ["stone_02", "chest_01"]
}
```

---

### BossConfig

```json
{
  "id": "snake_boss",
  "hp": 100000,
  "attackDamage": 100,
  "attackInterval": 2.0,
  "timeLimit": 120
}
```

---

# 10. 游戏主循环

每帧执行：

```python
while game_running:
    dt = get_delta_time()

    target = get_current_target()

    damage = player.attack * player.fire_rate * dt

    target.hp -= damage

    if target.hp <= 0:
        resolve_target(target)

    boss_update(dt)

    check_win_or_lose()
```

---

## 10.1 打门逻辑

```python
def attack_gate(gate, damage):
    gate.remaining_cost -= damage

    if gate.remaining_cost <= 0:
        apply_gate_reward(gate)
        gate.level += 1
        refresh_gate(gate)
```

---

## 10.2 刷新门逻辑

```python
def refresh_gate(gate):
    gate.remaining_cost = gate.base_cost * (gate.cost_growth ** gate.level)
    gate.current_reward = gate.base_reward * (gate.reward_growth ** gate.level)
```

---

## 10.3 应用奖励逻辑

```python
def apply_gate_reward(gate):
    reward = gate.current_reward

    if gate.type == "ATTACK_ADD":
        player.attack += reward

    elif gate.type == "ATTACK_MULT":
        player.attack *= reward

    elif gate.type == "FIRE_RATE_MULT":
        player.fire_rate *= reward
```

注意：如果倍率门写的是 (\times2)，那么 `reward = 2`。

---

# 11. 强化学习 Agent 建模

## 11.1 MDP 定义

强化学习环境可以定义为：

[
\boxed{
(S,A,P,R,\gamma)
}
]

其中：

| 项        | 含义   |
| -------- | ---- |
| (S)      | 状态空间 |
| (A)      | 动作空间 |
| (P)      | 状态转移 |
| (R)      | 奖励函数 |
| (\gamma) | 折扣因子 |

---

## 11.2 状态空间 Observation

Agent 每一步看到的信息：

[
S_t=(A_t,F_t,D_t,t_{\text{remain}},G_1,\ldots,G_m,O_1,\ldots,O_n,H_s,HP_g)
]

其中每个门 (G_i) 包括：

[
G_i=(type_i,level_i,cost_i,reward_i,\rho_i)
]

每个障碍 (O_j) 包括：

[
O_j=(hp_j,isUnlocked_j,reward_j)
]

推荐实际输入向量：

```python
obs = [
    normalized_attack,
    normalized_fire_rate,
    normalized_dps,
    normalized_time_remaining,

    gate_1_type_onehot,
    gate_1_level,
    gate_1_cost,
    gate_1_reward,
    gate_1_roi,

    gate_2_type_onehot,
    gate_2_level,
    gate_2_cost,
    gate_2_reward,
    gate_2_roi,

    stone_1_hp,
    stone_2_hp,

    boss_hp,
    base_hp
]
```

---

## 11.3 动作空间 Action

最简单的动作空间：

[
a_t \in {0,1,2,\ldots,m+n}
]

含义：

| 动作  | 含义       |
| --- | -------- |
| 0   | 攻击第 1 个门 |
| 1   | 攻击第 2 个门 |
| 2   | 攻击第 3 个门 |
| ... | ...      |
| m   | 攻击石头     |
| m+1 | 攻击宝箱     |
| m+2 | 攻击大蛇     |

也就是说，Agent 每一步只需要决定：

[
\boxed{\text{当前攻击哪个目标}}
]

这是离散动作空间，适合 DQN 或 PPO。

---

## 11.4 环境 step 设计

每次 step 可以表示一小段时间，例如：

[
\Delta t=0.1s
]

Agent 选择目标：

```python
action = agent.select_action(obs)
```

环境执行：

```python
damage = player.attack * player.fire_rate * dt
target.hp -= damage
```

如果目标被打完，就结算奖励或升级。

然后返回：

```python
next_obs, reward, done, info
```

---

# 12. 奖励函数设计

奖励函数非常关键。

## 12.1 稀疏奖励

最简单：

[
R=
\begin{cases}
+1000, & 击败大蛇 \
-1000, & 失败 \
0, & 其他情况
\end{cases}
]

问题是训练很慢。

---

## 12.2 推荐使用塑形奖励

为了让 Agent 更容易学，可以设计：

[
R_t=
w_1\Delta D_t
+w_2U_t
+w_3B_t
+w_4K_t
-w_5\Delta t
-w_6 damage_{\text{base}}
]

其中：

| 变量                     | 含义       |
| ---------------------- | -------- |
| (\Delta D_t)           | DPS 提升   |
| (U_t)                  | 是否解锁升级门  |
| (B_t)                  | 是否打开奖励宝箱 |
| (K_t)                  | 对大蛇造成的伤害 |
| (\Delta t)             | 时间消耗     |
| (damage_{\text{base}}) | 基地受到伤害   |

可以具体写成：

```python
reward = 0
reward += 0.01 * dps_increase
reward += 10 if gate_unlocked else 0
reward += 20 if chest_opened else 0
reward += 0.001 * boss_damage
reward -= 0.01
reward -= 0.1 * base_damage

if boss_dead:
    reward += 1000

if base_dead or timeout:
    reward -= 1000
```

---

## 12.3 防止奖励作弊

注意不要让 Agent 只刷升级门不打 Boss。

所以要加入：

1. 时间惩罚；
2. Boss 伤害奖励；
3. 通关大奖励；
4. 超时失败惩罚。

否则 Agent 可能学成：

[
\text{一直打门升级，不去打大蛇}
]

---

# 13. 推荐训练路线

## 13.1 第一阶段：规则环境测试

先不用 RL，写一个贪心 Bot。

贪心策略：

[
\rho_i=\frac{\Delta D_i}{C_i}
]

每次选择：

[
\arg\max_i \rho_i
]

也就是攻击性价比最高的门。

当所有门回本时间都太长时，开始攻击 Boss。

---

## 13.2 第二阶段：模仿学习

用贪心 Bot 打出一些轨迹：

[
(s_t,a_t)
]

先训练 Agent 模仿这个 Bot。

这样比纯随机探索更快。

---

## 13.3 第三阶段：强化学习

推荐算法：

| 算法           | 适用情况        |
| ------------ | ----------- |
| DQN          | 离散动作，目标数量固定 |
| PPO          | 更稳定，适合复杂环境  |
| Maskable PPO | 有些目标未解锁时很好用 |
| Rainbow DQN  | DQN 的增强版    |

本游戏动作是“选择攻击哪个目标”，天然是离散动作，所以：

[
\boxed{\text{DQN 或 PPO 都可以}}
]

如果你想快速稳定，建议：

[
\boxed{\text{PPO + action mask}}
]

---

# 14. Action Mask

有些目标当前不能打，比如：

1. 石头还没解锁；
2. 门暂时不可见；
3. Boss 还没出现；
4. 宝箱前面有障碍挡着。

所以要给 Agent 一个动作掩码：

```python
action_mask = [
    True,   # 门1可打
    True,   # 门2可打
    False,  # 门3暂不可打
    True,   # 石头1可打
    False,  # Boss暂不可打
]
```

Agent 只能从 `True` 的动作中选择。

---

# 15. 环境接口设计

建议环境接口接近 Gymnasium：

```python
class SnakeGateEnv:
    def reset(self):
        return obs, info

    def step(self, action):
        # action = 选择攻击哪个目标
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
```

---

## 15.1 step 伪代码

```python
def step(action):
    dt = self.dt

    target = self.targets[action]

    damage = self.player.attack * self.player.fire_rate * dt
    target.take_damage(damage)

    event = None

    if target.is_dead():
        event = self.resolve_target(target)

    self.boss_update(dt)

    reward = self.compute_reward(event)

    obs = self.get_observation()

    terminated = self.boss.hp <= 0 or self.base.hp <= 0
    truncated = self.time >= self.time_limit

    info = {
        "attack": self.player.attack,
        "fire_rate": self.player.fire_rate,
        "dps": self.player.attack * self.player.fire_rate,
        "event": event
    }

    return obs, reward, terminated, truncated, info
```

---

# 16. 贪心 Bot 基线

开发 Agent 前，先写一个数学 Bot，作为 baseline。

## 16.1 计算每个门的收益

```python
def calc_dps_gain(player, gate):
    A = player.attack
    F = player.fire_rate
    D = A * F
    R = gate.current_reward

    if gate.type == "ATTACK_ADD":
        D2 = (A + R) * F

    elif gate.type == "ATTACK_MULT":
        D2 = (A * R) * F

    elif gate.type == "FIRE_RATE_MULT":
        D2 = A * (F * R)

    return D2 - D
```

---

## 16.2 计算性价比

```python
def calc_roi(player, gate):
    gain = calc_dps_gain(player, gate)
    cost = gate.remaining_cost
    return gain / cost
```

---

## 16.3 选择动作

```python
def greedy_policy(env):
    best_gate = None
    best_roi = -1

    for gate in env.available_gates:
        roi = calc_roi(env.player, gate)

        if roi > best_roi:
            best_roi = roi
            best_gate = gate

    boss_time_now = env.boss.hp / env.player.dps

    if best_gate is None:
        return env.action_attack_boss

    gate_time = best_gate.remaining_cost / env.player.dps

    # 如果门的性价比太低，直接打 Boss
    if gate_time > boss_time_now * 0.3:
        return env.action_attack_boss

    return env.action_attack(best_gate)
```

这个 Bot 不一定最优，但能作为 RL 训练前的基准。

---

# 17. 调参建议

为了让游戏有策略性，不能让某一个门永远最优。

## 17.1 成本增长不能太慢

如果成本增长太慢，Agent 会一直刷同一个门。

例如：

[
C_i(L)=C_{i0}1.1^L
]

可能太便宜。

建议：

[
\gamma_i \in [1.5,3.0]
]

---

## 17.2 奖励增长不能太夸张

如果奖励增长太快，会数值爆炸。

例如：

[
R_i(L)=R_{i0}5^L
]

可能几级后直接失控。

建议：

[
\lambda_i \in [1.1,2.0]
]

倍率门尤其要谨慎。

---

## 17.3 乘法门需要高成本

例如：

[
\text{攻速}\times5
]

这种非常强，成本不能太低。

否则最优策略永远是先打它。

---

# 18. 最小可玩版本 MVP

第一版只需要这些内容：

## 玩家

* 攻击力 (A)
* 攻速 (F)
* 自动攻击目标

## 门

* 攻击 +X
* 攻击 ×X
* 攻速 ×X
* 成本随等级增长
* 奖励随等级增长

## 目标

* 若干石头
* 一个大蛇 Boss
* 一个基地血量

## 胜负

胜利：

[
H_s \leq 0
]

失败：

[
HP_g \leq 0
]

或超时：

[
t>T_{\max}
]

---

# 19. 开发里程碑

## Milestone 1：纯数值模拟

不做画面，只做 Python/脚本环境。

目标：

* 能 reset；
* 能 step；
* 能选择攻击目标；
* 能打门升级；
* 能击败 Boss 或失败。

---

## Milestone 2：贪心 Bot

实现：

[
\rho_i=\frac{\Delta D_i}{C_i}
]

让 Bot 自动选择性价比最高的门。

验证数学模型是否合理。

---

## Milestone 3：游戏原型

用 Unity/Godot/网页做可视化：

* 玩家角色；
* 升级门；
* 石头；
* Boss 大蛇；
* 数值变化；
* 自动攻击。

---

## Milestone 4：强化学习环境

实现 Gymnasium 风格接口：

```python
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)
```

---

## Milestone 5：训练 Agent

先训练简单关卡：

* 2 个门；
* 1 个石头；
* 1 个 Boss。

然后逐渐增加复杂度：

* 更多门；
* 更多石头；
* 更多宝箱；
* 更高成本增长；
* 更复杂地图。

---

# 20. 最终总结

这个游戏的核心数学模型是：

[
\boxed{
D=A\times F
}
]

[
\boxed{
\tau_i=\frac{C_i(L_i)}{A F}
}
]

[
\boxed{
C_i(L_i)=C_{i0}\gamma_i^{L_i}
}
]

[
\boxed{
R_i(L_i)=R_{i0}\lambda_i^{L_i}
}
]

最终目标是：

[
\boxed{
\min
\left[
t_n+\frac{H_s}{A_nF_n}
\right]
}
]

强化学习 Agent 的核心任务是：

[
\boxed{
\text{在每一刻选择攻击哪个目标，使最终通关时间最短或胜率最高}
}
]

开发上，它可以先做成一个纯数值环境，再接入强化学习，最后再做成完整游戏画面。
