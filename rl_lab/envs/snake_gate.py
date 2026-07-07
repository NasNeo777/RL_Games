"""Snake Gate: numeric upgrade-gate combat environment.

The player deals continuous DPS to one selected target. Targets include
upgrade gates, stones, chests, and the final snake boss. The environment keeps
the action space fixed so DQN/PPO can train without dynamic network heads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import ceil

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


@dataclass(frozen=True)
class GateConfig:
    gate_type: GateType
    base_cost: float
    cost_growth: float
    base_reward: float
    reward_growth: float
    max_level: int


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


@dataclass(frozen=True)
class LevelConfig:
    dt: float = 0.1
    time_limit: float = 120.0
    base_hp: float = 1200.0
    player_attack: float = 10.0
    player_fire_rate: float = 1.0
    gates: tuple[GateConfig, ...] = field(default_factory=lambda: (
        GateConfig(GateType.ATTACK_ADD, 35.0, 1.75, 15.0, 1.32, 10),
        GateConfig(GateType.ATTACK_MULT, 140.0, 2.35, 1.4, 1.04, 5),
        GateConfig(GateType.FIRE_RATE_MULT, 90.0, 2.15, 1.3, 1.04, 5),
    ))
    obstacle_hps: tuple[float, ...] = (250.0, 700.0)
    chest_hps: tuple[float, ...] = (420.0,)
    boss_hp: float = 5000.0
    boss_attack_damage: float = 14.0
    boss_attack_interval: float = 1.0


class SnakeGateEnv(BaseEnv):
    """Upgrade gates + stones + boss, implemented as a pure numeric env."""

    parallel_mode = "dummy"

    def __init__(self, seed=None, config: LevelConfig | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.config = config or LevelConfig()
        self.max_steps = int(ceil(self.config.time_limit / self.config.dt))
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
        self.player = PlayerState(
            attack=self.config.player_attack,
            fire_rate=self.config.player_fire_rate,
        )
        self.gates = self._make_gates()
        self.obstacles = self._make_obstacles()
        self.chests = self._make_chests()
        self.boss = BossState(
            hp=self.config.boss_hp,
            max_hp=self.config.boss_hp,
            attack_damage=self.config.boss_attack_damage,
            attack_interval=self.config.boss_attack_interval,
        )
        self.frames = []
        if self.record:
            self._record_frame(event="reset", action=None)
        return self._obs()

    def step(self, action):
        action = int(action)
        self.t += self.config.dt
        self.steps += 1

        reward = -0.01
        event = "idle"
        invalid = not self._is_action_valid(action)
        if invalid:
            reward -= 1.0
            event = "invalid_action"
        else:
            event, shaped = self._apply_player_damage(action)
            reward += shaped

        base_damage = self._boss_tick()
        reward -= 0.1 * base_damage

        boss_dead = self.boss.hp <= 0
        base_dead = self.base_hp <= 0
        terminated = boss_dead or base_dead
        truncated = (not terminated) and self.steps >= self.max_steps

        if boss_dead:
            reward += 1000.0
            event = "boss_defeated"
        elif base_dead:
            reward -= 1000.0
            event = "base_destroyed"
        elif truncated:
            reward -= 300.0
            event = "timeout"

        info = {
            "success": boss_dead,
            "score": int(max(0.0, self.config.boss_hp - self.boss.hp)),
            "event": event,
            "attack": self.player.attack,
            "fire_rate": self.player.fire_rate,
            "dps": self.player.dps,
            "base_hp": self.base_hp,
            "boss_hp": max(0.0, self.boss.hp),
            "action_mask": self.action_mask(),
        }
        if self.record:
            self._record_frame(event=event, action=action)
        return self._obs(), float(reward), terminated, truncated, info

    def action_mask(self):
        return [self._is_action_valid(a) for a in range(self.n_actions)]

    def greedy_action(self) -> int:
        """Small deterministic baseline for tuning and smoke tests."""
        boss_action = self.n_actions - 1

        best_gate_action = None
        best_roi = -1.0
        for i, gate in enumerate(self.gates):
            if not self._is_action_valid(i):
                continue
            roi = self._gate_roi(gate)
            if roi > best_roi:
                best_roi = roi
                best_gate_action = i

        if best_gate_action is not None and best_roi >= 0.05:
            return best_gate_action

        if self.boss.unlocked:
            return boss_action

        first_chest_action = None
        for action in range(self.n_actions):
            if not self._is_action_valid(action):
                continue
            target = self._target_for_action(action)
            if isinstance(target, ObstacleState):
                return action
            if isinstance(target, ChestState) and first_chest_action is None:
                first_chest_action = action

        if first_chest_action is not None:
            return first_chest_action
        if best_gate_action is not None:
            return best_gate_action

        mask = self.action_mask()
        return next((i for i, ok in enumerate(mask) if ok), 0)

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
                attack_bonus=55.0 * (i + 1),
                fire_rate_mult=1.05,
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
                dps_gain = max(0.0, self.player.dps - before_dps)
                return "gate_upgraded", 10.0 + 0.01 * dps_gain
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
                dps_gain = max(0.0, self.player.dps - before_dps)
                return "chest_opened", 30.0 + 0.01 * dps_gain
            return "chest_damaged", 0.0

        if isinstance(target, BossState):
            target.hp -= damage
            return "boss_damaged", 0.001 * damage

        return "unknown", 0.0

    def _resolve_gate(self, gate: GateState) -> None:
        reward = gate.current_reward
        if gate.config.gate_type == GateType.ATTACK_ADD:
            self.player.attack += reward
        elif gate.config.gate_type == GateType.ATTACK_MULT:
            self.player.attack *= reward
        elif gate.config.gate_type == GateType.FIRE_RATE_MULT:
            self.player.fire_rate *= reward

        gate.level += 1
        if gate.level >= gate.config.max_level:
            gate.unlocked = False
            gate.remaining_cost = 0.0
        else:
            gate.refresh()

    def _resolve_obstacle(self, obstacle: ObstacleState) -> None:
        obstacle.cleared = True
        obstacle.hp = 0.0

        try:
            idx = self.obstacles.index(obstacle)
        except ValueError:
            idx = -1
        if 0 <= idx + 1 < len(self.obstacles):
            self.obstacles[idx + 1].unlocked = True
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

    def _calc_obs_dim(self) -> int:
        return 7 + self.n_gates * 6 + self.n_obstacles * 3 + self.n_chests * 3

    def _obs(self):
        values = [
            np.log1p(self.player.attack) / 10.0,
            np.log1p(self.player.fire_rate) / 5.0,
            np.log1p(self.player.dps) / 12.0,
            max(0.0, 1.0 - self.steps / self.max_steps),
            self.base_hp / self.config.base_hp,
            max(0.0, self.boss.hp) / self.boss.max_hp,
            float(self.boss.unlocked),
        ]

        for gate in self.gates:
            roi = self._gate_roi(gate)
            values.extend([
                float(gate.config.gate_type) / 2.0,
                gate.level / max(1, gate.config.max_level),
                np.log1p(max(0.0, gate.remaining_cost)) / 12.0,
                np.log1p(max(0.0, gate.current_reward)) / 8.0,
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

        obs = np.array(values, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _gate_roi(self, gate: GateState) -> float:
        if not gate.unlocked or gate.remaining_cost <= 0:
            return 0.0
        old_dps = self.player.dps
        if gate.config.gate_type == GateType.ATTACK_ADD:
            new_dps = (self.player.attack + gate.current_reward) * self.player.fire_rate
        elif gate.config.gate_type == GateType.ATTACK_MULT:
            new_dps = (self.player.attack * gate.current_reward) * self.player.fire_rate
        else:
            new_dps = self.player.attack * (self.player.fire_rate * gate.current_reward)
        return max(0.0, new_dps - old_dps) / gate.remaining_cost

    def _record_frame(self, event=None, action=None):
        self.frames.append({
            "t": round(self.t, 2),
            "action": action,
            "attack": round(self.player.attack, 2),
            "fireRate": round(self.player.fire_rate, 2),
            "dps": round(self.player.dps, 2),
            "baseHp": round(self.base_hp, 2),
            "baseMaxHp": self.config.base_hp,
            "bossHp": round(max(0.0, self.boss.hp), 2),
            "bossMaxHp": self.boss.max_hp,
            "bossUnlocked": self.boss.unlocked,
            "gates": [
                {
                    "id": g.id,
                    "type": int(g.config.gate_type),
                    "level": g.level,
                    "maxLevel": g.config.max_level,
                    "cost": round(max(0.0, g.remaining_cost), 2),
                    "reward": round(g.current_reward, 2),
                    "roi": round(self._gate_roi(g), 3),
                    "unlocked": g.unlocked,
                }
                for g in self.gates
            ],
            "obstacles": [
                {
                    "id": o.id,
                    "hp": round(max(0.0, o.hp), 2),
                    "maxHp": o.max_hp,
                    "unlocked": o.unlocked,
                    "cleared": o.cleared,
                }
                for o in self.obstacles
            ],
            "chests": [
                {
                    "id": c.id,
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
            "goal": "击败大蛇",
            "gateTypes": ["攻击+", "攻击x", "攻速x"],
            "actionLabels": [
                *(f"门{i + 1}" for i in range(self.n_gates)),
                *(f"石头{i + 1}" for i in range(self.n_obstacles)),
                *(f"宝箱{i + 1}" for i in range(self.n_chests)),
                "大蛇",
            ],
        }


def greedy_action(env: SnakeGateEnv) -> int:
    return env.greedy_action()
