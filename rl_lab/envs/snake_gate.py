"""Snake Gate: vertical shooting gate-survival environment.

The snake enters from the top and gradually covers the screen. The player fires
from the bottom, routes bullets through fixed or moving upgrade gates, and has
to destroy snake body segments one by one before the snake reaches the base.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import ceil, sin, tau

import numpy as np

from .base import BaseEnv


class GateType(IntEnum):
    ATTACK_ADD = 0
    ATTACK_MULT = 1
    FIRE_RATE_MULT = 2


@dataclass
class PlayerState:
    attack: float = 12.0
    fire_rate: float = 1.0

    @property
    def dps(self) -> float:
        return self.attack * self.fire_rate


@dataclass(frozen=True)
class GateConfig:
    gate_type: GateType
    lane: int
    x: float
    y: float
    base_cost: float
    cost_growth: float
    base_reward: float
    reward_growth: float
    max_level: int
    moving: bool = False
    amplitude: float = 0.0
    speed: float = 0.0


@dataclass
class GateState:
    id: str
    config: GateConfig
    level: int = 0
    remaining_cost: float = 0.0
    current_reward: float = 0.0
    unlocked: bool = True

    def refresh(self) -> None:
        self.remaining_cost = (
            self.config.base_cost * self.config.cost_growth**self.level
        )
        self.current_reward = (
            self.config.base_reward * self.config.reward_growth**self.level
        )


@dataclass
class SnakeSegment:
    id: int
    front_order: int
    lane: int
    row: int
    path_x: float
    path_y: float
    hp: float
    max_hp: float
    depth: float
    status: str = "pending"
    status_age: float = 0.0
    chest: bool = False
    chest_opened: bool = False


@dataclass(frozen=True)
class LevelConfig:
    dt: float = 0.1
    time_limit: float = 120.0
    lanes: int = 3
    player_attack: float = 12.0
    player_fire_rate: float = 1.0
    snake_speed: float = 0.0065
    snake_push_per_segment: float = 0.024
    snake_retreat_on_kill: float = 0.035
    dead_clear_delay: float = 0.45
    fail_coverage: float = 0.88
    start_coverage: float = 0.18
    target_kills: int = 18
    segment_hp: float = 95.0
    segment_hp_growth: float = 1.18
    chest_every: int = 4
    gates: tuple[GateConfig, ...] = (
        GateConfig(GateType.FIRE_RATE_MULT, 0, 0.22, 0.62, 48.0, 1.65, 1.35, 1.06, 6),
        GateConfig(
            GateType.ATTACK_ADD,
            1,
            0.50,
            0.50,
            42.0,
            1.72,
            24.0,
            1.20,
            8,
            True,
            0.18,
            0.55,
        ),
        GateConfig(GateType.ATTACK_MULT, 2, 0.78, 0.66, 72.0, 1.88, 1.45, 1.05, 6),
    )


class SnakeGateEnv(BaseEnv):
    """Top-down snake pressure + upgrade-gate shooting game."""

    parallel_mode = "dummy"
    obs_dim = 34
    n_actions = 7

    def __init__(self, seed=None, config: LevelConfig | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.config = config or LevelConfig()
        self.max_steps = int(ceil(self.config.time_limit / self.config.dt))
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.steps = 0
        self.coverage = self.config.start_coverage
        self.player = PlayerState(
            self.config.player_attack, self.config.player_fire_rate
        )
        self.gates = self._make_gates()
        self.snake_segments = self._make_snake_body()
        self.next_entry_index = 0
        self.entry_meter = 0.0
        self.kills = 0
        self.cleared = 0
        self.last_bullets = []
        self.last_target_id = None
        for _ in range(min(self.config.lanes * 2, len(self.snake_segments))):
            self._enter_next_segment()
        self.frames = []
        if self.record:
            self._record_frame(event="reset", action=None)
        return self._obs()

    def step(self, action):
        action = int(action)
        self.t += self.config.dt
        self.steps += 1
        self.last_bullets = []
        self.last_target_id = None

        self._advance_snake_lifecycle()
        reward = -0.02 - 0.35 * self.coverage
        event, shaped = self._fire(action)
        reward += shaped

        success = self._snake_defeated()
        failed = self.coverage >= self.config.fail_coverage
        terminated = success or failed
        truncated = (not terminated) and self.steps >= self.max_steps
        if success:
            reward += 1000.0
            event = "snake_defeated"
        elif failed:
            reward -= 1000.0
            event = "screen_covered"
        elif truncated:
            reward -= 250.0
            event = "timeout"

        info = {
            "success": success,
            "score": int(
                self.kills * 100 + self.player.dps * 3 + (1 - self.coverage) * 100
            ),
            "event": event,
            "attack": self.player.attack,
            "fire_rate": self.player.fire_rate,
            "dps": self.player.dps,
            "coverage": self.coverage,
            "kills": self.kills,
            "cleared": self.cleared,
            "pending": self._count_status("pending"),
            "alive": self._active_segment_count(),
            "action_mask": self.action_mask(),
        }
        if self.record:
            self._record_frame(event=event, action=action)
        return self._obs(), float(reward), terminated, truncated, info

    def action_mask(self):
        return [True] * self.n_actions

    def greedy_action(self) -> int:
        best = max(range(3), key=lambda lane: self._lane_threat(lane))
        for i, gate in enumerate(self.gates):
            if gate.config.lane == best and self._gate_roi(gate) > 0.15:
                return i
        chest_lane = self._best_chest_lane()
        if chest_lane is not None:
            return 6
        return 3 + best

    def _make_gates(self):
        gates = []
        for i, cfg in enumerate(self.config.gates):
            gate = GateState(id=f"gate_{i}", config=cfg)
            gate.refresh()
            gates.append(gate)
        return gates

    def _make_snake_body(self) -> list[SnakeSegment]:
        body = []
        total = self.config.target_kills
        rows = max(1, ceil(total / self.config.lanes))
        for idx in range(total):
            row = idx // self.config.lanes
            display_row = rows - 1 - row
            col = idx % self.config.lanes
            lane = col if row % 2 == 0 else self.config.lanes - 1 - col
            progress = idx / max(1, total - 1)
            hp_scale = 1.0 + (self.config.segment_hp_growth - 1.0) * idx
            hp_scale *= 1.0 + 0.28 * progress * progress
            hp = self.config.segment_hp * hp_scale
            chest = idx > 0 and idx % self.config.chest_every == 0
            depth = 0.08 + 0.78 * (display_row / max(1, rows - 1))
            path_x = (lane + 0.5) / self.config.lanes
            path_y = (display_row + 0.5) / rows
            body.append(
                SnakeSegment(
                    idx,
                    idx,
                    lane,
                    row,
                    path_x,
                    path_y,
                    hp,
                    hp,
                    depth,
                    "pending",
                    0.0,
                    chest,
                )
            )
        return body

    def _enter_next_segment(self) -> bool:
        if self.next_entry_index >= len(self.snake_segments):
            return False
        segment = self.snake_segments[self.next_entry_index]
        segment.status = "entered"
        segment.status_age = 0.0
        self.next_entry_index += 1
        self.coverage = min(
            self.config.fail_coverage,
            self.coverage
            + self.config.snake_push_per_segment * self._segment_pressure(segment),
        )
        return True

    def _advance_snake_lifecycle(self) -> None:
        for segment in self.snake_segments:
            if segment.status in {"entered", "alive", "dead"}:
                segment.status_age += self.config.dt
            if segment.status == "entered" and segment.status_age >= self.config.dt:
                segment.status = "alive"
                segment.status_age = 0.0
            elif (
                segment.status == "dead"
                and segment.status_age >= self.config.dead_clear_delay
            ):
                segment.status = "cleared"
                segment.status_age = 0.0
                self.cleared += 1

        active_pressure = sum(
            self._segment_pressure(s)
            for s in self.snake_segments
            if s.status in {"entered", "alive"}
        ) / max(1, self.config.target_kills)
        self.coverage = min(
            1.0,
            self.coverage
            + self.config.snake_speed * self.config.dt * (1.0 + 2.4 * active_pressure),
        )

        if self.next_entry_index < len(self.snake_segments):
            progress = self.next_entry_index / max(1, len(self.snake_segments) - 1)
            rate = 0.72 + 0.76 * progress + 0.36 * self.coverage
            self.entry_meter += rate * self.config.dt
            while self.entry_meter >= 1.0 and self._enter_next_segment():
                self.entry_meter -= 1.0

    def _segment_pressure(self, segment: SnakeSegment) -> float:
        progress = segment.id / max(1, self.config.target_kills - 1)
        return 1.0 + 0.8 * progress

    def _count_status(self, status: str) -> int:
        return sum(1 for s in self.snake_segments if s.status == status)

    def _active_segment_count(self) -> int:
        return sum(1 for s in self.snake_segments if s.status in {"entered", "alive"})

    def _snake_defeated(self) -> bool:
        return self.next_entry_index >= len(self.snake_segments) and all(
            s.status in {"dead", "cleared"} for s in self.snake_segments
        )

    def _fire(self, action: int):
        lane = self._lane_for_action(action)
        damage = self.player.dps * self.config.dt
        shaped = 0.05 * damage
        event = "snake_hit"
        gate = self.gates[action] if 0 <= action < len(self.gates) else None

        if gate is not None:
            gate.remaining_cost -= damage
            event = "gate_hit"
            shaped += 0.1
            if gate.remaining_cost <= 0 and gate.unlocked:
                before = self.player.dps
                self._resolve_gate(gate)
                damage *= 1.0 + min(
                    2.0, max(0.0, self.player.dps - before) / max(1.0, before)
                )
                event = "gate_upgraded"
                shaped += 18.0 + 0.02 * max(0.0, self.player.dps - before)

        target = self._target_segment(lane, chest_priority=(action == 6))
        if target is not None:
            target.hp -= damage
            self.last_target_id = target.id
            if target.hp <= 0:
                shaped += self._resolve_segment(target)
                event = "chest_opened" if target.chest else "segment_destroyed"
        else:
            event = "empty_lane" if gate is None else event
            shaped -= 0.4

        self.last_bullets.append(
            {
                "lane": lane,
                "gate": gate.id if gate else None,
                "target": self.last_target_id,
            }
        )
        return event, shaped

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

    def _resolve_segment(self, segment: SnakeSegment) -> float:
        self.kills += 1
        segment.status = "dead"
        segment.status_age = 0.0
        segment.hp = 0.0
        self.coverage = max(0.08, self.coverage - self.config.snake_retreat_on_kill)
        progress_bonus = 1.0 + 0.45 * segment.id / max(1, self.config.target_kills - 1)
        reward = 30.0 + 0.18 * segment.max_hp * progress_bonus
        if segment.chest and not segment.chest_opened:
            segment.chest_opened = True
            self.player.attack += 35.0 + self.kills * 4.5
            self.player.fire_rate *= 1.09
            reward += 70.0
        return reward

    def _target_segment(self, lane: int, chest_priority=False):
        targetable = [
            s for s in self.snake_segments if s.status in {"entered", "alive"}
        ]
        candidates = [s for s in targetable if s.lane == lane]
        if not candidates:
            candidates = list(targetable)
        if not candidates:
            return None
        if chest_priority:
            chests = [s for s in candidates if s.chest]
            if chests:
                return max(chests, key=lambda s: s.front_order)
        return max(candidates, key=lambda s: s.front_order)

    def _lane_for_action(self, action: int) -> int:
        if 0 <= action < len(self.gates):
            return self.gates[action].config.lane
        if 3 <= action <= 5:
            return action - 3
        return self._best_chest_lane() or int(
            np.argmax([self._lane_threat(i) for i in range(3)])
        )

    def _best_chest_lane(self):
        chests = [
            s
            for s in self.snake_segments
            if s.chest and s.status in {"entered", "alive"}
        ]
        if not chests:
            return None
        return max(chests, key=lambda s: s.front_order).lane

    def _lane_threat(self, lane: int) -> float:
        return sum(
            (s.depth + 0.5) * (s.hp / max(1.0, s.max_hp))
            for s in self.snake_segments
            if s.lane == lane and s.status in {"entered", "alive"}
        )

    def _gate_x(self, gate: GateState) -> float:
        cfg = gate.config
        if not cfg.moving:
            return cfg.x
        return cfg.x + cfg.amplitude * sin(tau * cfg.speed * self.t)

    def _gate_roi(self, gate: GateState) -> float:
        if not gate.unlocked or gate.remaining_cost <= 0:
            return 0.0
        before = self.player.dps
        if gate.config.gate_type == GateType.ATTACK_ADD:
            after = (self.player.attack + gate.current_reward) * self.player.fire_rate
        elif gate.config.gate_type == GateType.ATTACK_MULT:
            after = self.player.attack * gate.current_reward * self.player.fire_rate
        else:
            after = self.player.attack * self.player.fire_rate * gate.current_reward
        return max(0.0, after - before) / gate.remaining_cost

    def _obs(self):
        values = [
            np.log1p(self.player.attack) / 10.0,
            np.log1p(self.player.fire_rate) / 5.0,
            np.log1p(self.player.dps) / 12.0,
            self.coverage,
            self.kills / max(1, self.config.target_kills),
            max(0.0, 1.0 - self.steps / self.max_steps),
            self._active_segment_count() / 10.0,
        ]
        for gate in self.gates:
            values.extend(
                [
                    float(gate.config.gate_type) / 2.0,
                    gate.level / max(1, gate.config.max_level),
                    np.log1p(max(0.0, gate.remaining_cost)) / 12.0,
                    np.log1p(max(0.0, gate.current_reward)) / 8.0,
                    self._gate_roi(gate) / 10.0,
                    self._gate_x(gate),
                ]
            )
        for lane in range(3):
            lane_segments = [
                s
                for s in self.snake_segments
                if s.lane == lane and s.status in {"entered", "alive"}
            ]
            front = max((s.depth for s in lane_segments), default=0.0)
            hp = sum(max(0.0, s.hp) / max(1.0, s.max_hp) for s in lane_segments)
            chest = any(s.chest for s in lane_segments)
            values.extend([front, min(1.0, hp / 4.0), float(chest)])
        values = values[: self.obs_dim]
        values.extend([0.0] * (self.obs_dim - len(values)))
        obs = np.array(values, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _record_frame(self, event=None, action=None):
        self.frames.append(
            {
                "t": round(self.t, 2),
                "action": action,
                "attack": round(self.player.attack, 2),
                "fireRate": round(self.player.fire_rate, 2),
                "dps": round(self.player.dps, 2),
                "coverage": round(self.coverage, 4),
                "failCoverage": self.config.fail_coverage,
                "kills": self.kills,
                "targetKills": self.config.target_kills,
                "pendingSegments": self._count_status("pending"),
                "aliveSegments": self._active_segment_count(),
                "clearedSegments": self.cleared,
                "snakeSegments": [
                    {
                        "id": s.id,
                        "frontOrder": s.front_order,
                        "lane": s.lane,
                        "row": s.row,
                        "pathX": round(s.path_x, 3),
                        "pathY": round(s.path_y, 3),
                        "hp": round(max(0.0, s.hp), 2),
                        "maxHp": round(s.max_hp, 2),
                        "depth": round(s.depth, 3),
                        "status": s.status,
                        "chest": s.chest,
                        "chestOpened": s.chest_opened,
                        "hit": self.last_target_id == s.id,
                    }
                    for s in self.snake_segments
                    if s.status != "cleared"
                ],
                "gates": [
                    {
                        "id": g.id,
                        "type": int(g.config.gate_type),
                        "lane": g.config.lane,
                        "x": round(self._gate_x(g), 3),
                        "y": g.config.y,
                        "moving": g.config.moving,
                        "level": g.level,
                        "maxLevel": g.config.max_level,
                        "cost": round(max(0.0, g.remaining_cost), 2),
                        "reward": round(g.current_reward, 2),
                        "roi": round(self._gate_roi(g), 3),
                        "unlocked": g.unlocked,
                    }
                    for g in self.gates
                ],
                "bullets": self.last_bullets,
                "event": event,
            }
        )

    def render_spec(self):
        return {
            "type": "snake_gate",
            "frame_dt": self.config.dt,
            "goal": "阻止大蛇覆盖屏幕",
            "gateTypes": ["攻击+", "攻击x", "攻速x"],
            "lanes": self.config.lanes,
            "actionLabels": [
                "左门",
                "移动门",
                "右门",
                "左路",
                "中路",
                "右路",
                "宝箱优先",
            ],
        }


def greedy_action(env: SnakeGateEnv) -> int:
    return env.greedy_action()
