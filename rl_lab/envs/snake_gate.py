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
    entry_progress: float = 0.0


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
    initial_entry_distance: float = 1.2
    segment_entry_duration: float = 0.85
    entry_start_y: float = -0.08
    bullet_start_y: float = 0.83
    bullet_end_y: float = -0.08
    snake_body_spacing: float = 0.72
    snake_track_left: float = 0.14
    snake_track_right: float = 0.86
    snake_track_row_gap: float = 0.105
    snake_track_straight_distance: float = 4.2
    snake_track_turn_distance: float = 0.9
    snake_track_turn_bulge: float = 0.045
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
            0.30,
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
        self.entry_meter = self.config.initial_entry_distance
        self.kills = 0
        self.cleared = 0
        self.last_bullets = []
        self.last_target_id = None
        self.last_floaters = []
        self._sync_snake_entry_states()
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

        # age existing floaters, remove expired ones (> 1.5s)
        for fl in self.last_floaters:
            fl["age"] = fl.get("age", 0.0) + self.config.dt
        self.last_floaters = [fl for fl in self.last_floaters if fl["age"] < fl.get("max_age", 1.5)]

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

    def _advance_snake_lifecycle(self) -> None:
        for segment in self.snake_segments:
            if segment.status == "dead":
                segment.status_age += self.config.dt
            if (
                segment.status == "dead"
                and segment.status_age >= self.config.dead_clear_delay
            ):
                segment.status = "cleared"
                segment.status_age = 0.0
                self.cleared += 1

        previous_entry = self.entry_meter
        if self._body_segments():
            progress = min(
                1.0, self.entry_meter / max(1.0, len(self.snake_segments) - 1)
            )
            rate = 0.72 + 0.76 * progress + 0.36 * self.coverage
            self.entry_meter += rate * self.config.dt
            self._apply_continuous_entry_pressure(previous_entry, self.entry_meter)

        self._sync_snake_entry_states()

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

    def _segment_pressure(self, segment: SnakeSegment) -> float:
        progress = segment.id / max(1, self.config.target_kills - 1)
        return 1.0 + 0.8 * progress

    def _apply_continuous_entry_pressure(self, previous: float, current: float) -> None:
        all_started = self._tail_start_distance()
        previous = min(previous, all_started)
        current = min(current, all_started)
        delta = max(0.0, current - previous)
        if delta <= 0.0:
            return
        idx = max(
            0,
            min(
                len(self.snake_segments) - 1,
                int(current / max(0.1, self.config.snake_body_spacing)),
            ),
        )
        segment = self._segment_at_body_slot(idx)
        if segment is None:
            return
        self.coverage = min(
            self.config.fail_coverage,
            self.coverage
            + self.config.snake_push_per_segment
            * delta
            * self._segment_pressure(segment),
        )

    def _body_segments(self) -> list[SnakeSegment]:
        return [
            s
            for s in sorted(self.snake_segments, key=lambda x: x.front_order)
            if s.status != "cleared"
        ]

    def _segment_at_body_slot(self, slot: int) -> SnakeSegment | None:
        body = self._body_segments()
        if not body:
            return None
        return min(body, key=lambda segment: abs(self._segment_body_slot(segment) - slot))

    def _segment_body_slot(self, segment: SnakeSegment) -> int:
        slot = 0
        for candidate in sorted(self.snake_segments, key=lambda x: x.front_order):
            if candidate is segment:
                break
            slot += 1
        cleared_after = sum(
            1
            for candidate in self.snake_segments
            if candidate.status == "cleared"
            and candidate.front_order > segment.front_order
        )
        return slot + cleared_after

    def _slot_start_distance(self, slot: int) -> float:
        return slot * self.config.snake_body_spacing

    def _tail_start_distance(self) -> float:
        body = self._body_segments()
        if not body:
            return 0.0
        return self._slot_start_distance(
            max(self._segment_body_slot(segment) for segment in body)
        )

    def _tail_fully_entered_distance(self) -> float:
        return self._tail_start_distance() + self.config.segment_entry_duration

    def _segment_path_distance(self, segment: SnakeSegment) -> float:
        return self.entry_meter - self._slot_start_distance(
            self._segment_body_slot(segment)
        )

    def _entry_progress_from_distance(self, distance: float) -> float:
        return max(
            0.0,
            min(
                1.0,
                distance / max(self.config.dt, self.config.segment_entry_duration),
            ),
        )

    def _snake_path_position(self, distance: float) -> tuple[float, float]:
        distance = max(0.0, distance)
        straight = max(0.1, self.config.snake_track_straight_distance)
        turn = max(0.1, self.config.snake_track_turn_distance)
        cycle = straight + turn
        row = int(distance / cycle)
        offset = distance - row * cycle
        left = self.config.snake_track_left
        right = self.config.snake_track_right
        row_y = (
            self.config.entry_start_y
            + self.coverage * 0.12
            + row * self.config.snake_track_row_gap
        )

        left_to_right = row % 2 == 0
        if offset <= straight:
            u = offset / straight
            if left_to_right:
                x = left + (right - left) * u
            else:
                x = right - (right - left) * u
            y = row_y
        else:
            u = self._smoothstep((offset - straight) / turn)
            if left_to_right:
                x = right + self.config.snake_track_turn_bulge * (1.0 - abs(2.0 * u - 1.0))
            else:
                x = left - self.config.snake_track_turn_bulge * (1.0 - abs(2.0 * u - 1.0))
            y = row_y + self.config.snake_track_row_gap * u

        return max(0.08, min(0.92, x)), y

    def _sync_snake_entry_states(self) -> None:
        started = 0
        for segment in self.snake_segments:
            if segment.status in {"dead", "cleared"}:
                started += 1
                continue

            distance = self._segment_path_distance(segment)
            progress = self._entry_progress_from_distance(distance)
            segment.entry_progress = progress

            if distance <= 0.0:
                segment.status = "pending"
                segment.status_age = 0.0
                continue

            started += 1
            before = segment.status
            segment.status = "alive" if progress >= 1.0 else "entered"
            segment.status_age = progress * self.config.segment_entry_duration

            x, _ = self._snake_path_position(distance)
            segment.lane = max(
                0, min(self.config.lanes - 1, int(x * self.config.lanes))
            )
            if before == "pending":
                segment.status_age = 0.0

        self.next_entry_index = min(len(self.snake_segments), started)

    def _smoothstep(self, value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _segment_entry_progress(self, segment: SnakeSegment) -> float:
        if segment.status == "pending":
            return 0.0
        if segment.status == "alive":
            return 1.0
        if segment.status == "cleared":
            return 1.0
        return max(0.0, min(1.0, segment.entry_progress))

    def _segment_render_geometry(self, segment: SnakeSegment) -> dict[str, float]:
        distance = self._segment_path_distance(segment)
        progress = self._segment_entry_progress(segment)
        x, y = self._snake_path_position(distance)
        return {
            "x": x,
            "y": y,
            "w": 82.0 / 390.0,
            "h": 42.0 / 693.0,
            "entry_progress": progress,
        }

    def _lane_screen_x(self, lane: int) -> float:
        lane = max(0, min(self.config.lanes - 1, lane))
        if self.config.lanes == 3:
            return (82.0 + lane * 112.0) / 390.0
        return (lane + 0.5) / max(1, self.config.lanes)

    def _bullet_path_for_action(
        self, action: int, lane: int, gate: GateState | None
    ) -> tuple[float, float]:
        if action == 6:
            chest = self._best_chest_segment()
            if chest is not None:
                return self._segment_render_geometry(chest)["x"], self.config.bullet_end_y
        if gate is not None:
            return self._gate_x(gate), gate.config.y
        return self._lane_screen_x(lane), self.config.bullet_end_y

    def _first_colliding_segment(
        self, bullet_x: float
    ) -> tuple[SnakeSegment | None, float | None]:
        best: tuple[float, SnakeSegment] | None = None
        start_y = self.config.bullet_start_y
        end_y = self.config.bullet_end_y
        for segment in self.snake_segments:
            if segment.status not in {"entered", "alive"}:
                continue
            geo = self._segment_render_geometry(segment)
            if geo["entry_progress"] <= 0.0:
                continue
            half_w = geo["w"] * 0.5
            half_h = geo["h"] * 0.5
            if not (geo["x"] - half_w <= bullet_x <= geo["x"] + half_w):
                continue
            top = geo["y"] - half_h
            bottom = geo["y"] + half_h
            if bottom < end_y or top > start_y:
                continue
            hit_y = min(start_y, bottom)
            if best is None or hit_y > best[0]:
                best = (hit_y, segment)
        if best is None:
            return None, None
        return best[1], best[0]

    def _count_status(self, status: str) -> int:
        return sum(1 for s in self.snake_segments if s.status == status)

    def _active_segment_count(self) -> int:
        return sum(1 for s in self.snake_segments if s.status in {"entered", "alive"})

    def _snake_defeated(self) -> bool:
        return all(s.status in {"dead", "cleared"} for s in self.snake_segments)

    def _fire(self, action: int):
        lane = self._lane_for_action(action)
        damage = self.player.dps * self.config.dt
        shaped = 0.05 * damage
        event = "snake_hit"
        gate = self.gates[action] if 0 <= action < len(self.gates) else None
        bullet_x, aim_y = self._bullet_path_for_action(action, lane, gate)
        target, hit_y = self._target_segment(bullet_x)

        # 即使不是瞄准门的子弹，也要检查弹道路径上是否有门
        if gate is None:
            gate = self._gate_in_path(bullet_x)
        gate_blocked = (
            gate is not None and hit_y is not None and hit_y > gate.config.y
        )

        bullet_consumed = False

        if gate is not None and not gate_blocked:
            gate.remaining_cost -= 1.0
            event = "gate_hit"
            shaped += 0.1
            bullet_consumed = True
            hit_y = gate.config.y
            self.last_floaters.append(
                {"text": "-1", "x": self._gate_x(gate), "y": gate.config.y,
                 "color": "#fff176", "size": "small"}
            )
            if gate.remaining_cost <= 0 and gate.unlocked:
                before = self.player.dps
                self._resolve_gate(gate)
                damage *= 1.0 + min(
                    2.0, max(0.0, self.player.dps - before) / max(1.0, before)
                )
                event = "gate_upgraded"
                shaped += 18.0 + 0.02 * max(0.0, self.player.dps - before)

        if target is not None and not bullet_consumed:
            target.hp -= damage
            self.last_target_id = target.id
            target_geo = self._segment_render_geometry(target)
            self.last_floaters.append(
                {"text": f"-{damage:.0f}", "x": target_geo["x"], "y": target_geo["y"],
                 "color": "#ffe082", "size": "small"}
            )
            if target.hp <= 0:
                shred = self._resolve_segment(target)
                shaped += shred
                self.last_floaters.append(
                    {"text": f"+{shred:.0f}💰", "x": target_geo["x"], "y": target_geo["y"],
                     "color": "#ffd54f", "size": "large"}
                )
                event = "chest_opened" if target.chest else "segment_destroyed"
        else:
            event = "empty_lane" if gate is None else event
            shaped -= 0.4

        self.last_bullets.append(
            {
                "lane": lane,
                "x": round(bullet_x, 3),
                "aimY": round(aim_y, 3),
                "hitY": round(hit_y, 3) if hit_y is not None else None,
                "gate": gate.id if gate else None,
                "gateBlocked": gate_blocked,
                "target": self.last_target_id,
            }
        )
        return event, shaped

    def _resolve_gate(self, gate: GateState) -> None:
        reward = gate.current_reward
        before_attack = self.player.attack
        before_fire_rate = self.player.fire_rate
        if gate.config.gate_type == GateType.ATTACK_ADD:
            self.player.attack += reward
            self.last_floaters.append(
                {"text": f"+{reward:.0f} 攻击", "x": gate.config.x, "y": gate.config.y,
                 "color": "#ff9f4b", "size": "large"}
            )
        elif gate.config.gate_type == GateType.ATTACK_MULT:
            self.player.attack *= reward
            self.last_floaters.append(
                {"text": f"x{reward:.2f} 攻击", "x": gate.config.x, "y": gate.config.y,
                 "color": "#ff7043", "size": "large"}
            )
        elif gate.config.gate_type == GateType.FIRE_RATE_MULT:
            self.player.fire_rate *= reward
            self.last_floaters.append(
                {"text": f"x{reward:.2f} 攻速", "x": gate.config.x, "y": gate.config.y,
                 "color": "#4fc3f7", "size": "large"}
            )
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
            geo = self._segment_render_geometry(segment)
            atk_bonus = 35.0 + self.kills * 4.5
            self.last_floaters.append(
                {"text": f"宝箱 +{atk_bonus:.0f}攻击 x1.09速",
                 "x": geo["x"], "y": geo["y"],
                 "color": "#ce93d8", "size": "huge"}
            )
        return reward

    def _target_segment(self, bullet_x: float):
        return self._first_colliding_segment(bullet_x)

    def _lane_for_action(self, action: int) -> int:
        if 0 <= action < len(self.gates):
            return self.gates[action].config.lane
        if 3 <= action <= 5:
            return action - 3
        chest_lane = self._best_chest_lane()
        if chest_lane is not None:
            return chest_lane
        return int(np.argmax([self._lane_threat(i) for i in range(3)]))

    def _best_chest_lane(self):
        chest = self._best_chest_segment()
        return chest.lane if chest is not None else None

    def _best_chest_segment(self):
        chests = [
            s
            for s in self.snake_segments
            if s.chest and s.status in {"entered", "alive"}
        ]
        if not chests:
            return None
        return max(chests, key=lambda s: self._segment_render_geometry(s)["y"])

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

    def _gate_in_path(self, bullet_x: float) -> GateState | None:
        half_w = (82.0 / 390.0) * 0.5
        for g in self.gates:
            gx = self._gate_x(g)
            if abs(bullet_x - gx) > half_w:
                continue
            if not (self.config.bullet_end_y < g.config.y < self.config.bullet_start_y):
                continue
            return g
        return None

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
                        "renderX": round(self._segment_render_geometry(s)["x"], 3),
                        "renderY": round(self._segment_render_geometry(s)["y"], 3),
                        "entryProgress": round(
                            self._segment_render_geometry(s)["entry_progress"], 3
                        ),
                        "collisionW": round(self._segment_render_geometry(s)["w"], 3),
                        "collisionH": round(self._segment_render_geometry(s)["h"], 3),
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
                        "cost": round(max(0.0, g.remaining_cost)),
                        "reward": round(g.current_reward, 2),
                        "roi": round(self._gate_roi(g), 3),
                        "unlocked": g.unlocked,
                    }
                    for g in self.gates
                ],
                "bullets": self.last_bullets,
                "floaters": [
                    {"text": fl["text"], "x": fl["x"], "y": fl["y"],
                     "color": fl["color"], "size": fl["size"],
                     "age": fl.get("age", 0.0), "maxAge": fl.get("max_age", 1.5)}
                    for fl in self.last_floaters
                ],
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
