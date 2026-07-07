# Snake Gate 算法建模

## 1. 变量提取

- `coverage`: 大蛇覆盖屏幕比例，达到 `fail_coverage` 判负。
- `player.attack`: 单发攻击强度。
- `player.fire_rate`: 发射频率。
- `player.dps`: `attack * fire_rate`，每秒有效输出。
- `snake_segments`: 大蛇完整身体分段，每段包含 `lane / row / path_x / path_y / renderX / renderY / entryProgress / hp / max_hp / depth / status / chest`。
- `gates`: 门状态，每个门包含 `type / lane / x / y / moving / level / cost / reward`。
- `action`: 智能体动作，选择打哪个门、哪条路，或宝箱优先。
- `kills`: 已打死蛇身段数；只用于进度和奖励，不再单独判胜。
- `pending / entered / alive / dead / cleared`: 蛇身生命周期。所有段是同一条连续折返蛇身上的采样点，死亡后短暂保留，再清除并释放身体槽位。

## 2. 变量关系

- 大蛇持续下压：`coverage += snake_speed * dt`。
- 大蛇按连续距离游入：`entry_meter` 每帧只增加一小段，蛇身块按固定间距挂在横向折返轨迹上，从屏幕上方连续露出。
- 被击破的身体块清除后不再占位，蛇头一侧的身体会往后缩来补空位，尾巴一侧不会主动前顶。
- 蛇身连续游入会增加屏幕压力：`coverage += snake_push_per_segment * delta_entry * segment_pressure`。
- 后段蛇身 `hp` 和 `segment_pressure` 更高，入场节奏也随进度和覆盖压力提升。
- 击杀蛇身会减轻压力：`coverage -= snake_retreat_on_kill`。
- 子弹穿门会消耗门成本：`gate.cost -= player.dps * dt`。
- 门被打穿后升级，下一次成本和奖励同时增长。
- 子弹使用物理碰撞命中蛇身：沿子弹射线从玩家侧向上检测，先碰到哪块身体就只扣哪块血，不能越过前面的身体块打后面的块。
- 宝箱蛇身被击破后额外提升攻击和攻速。

## 3. 数学建模

状态向量 `s_t` 由玩家属性、大蛇压力、门状态、三条 lane 的威胁值组成：

```text
s_t = [log(attack), log(fire_rate), log(dps), coverage, progress,
       gate_0..gate_n, lane_0_threat..lane_n_threat]
```

动作空间固定为 7 个离散动作：

```text
0: 左门
1: 移动门
2: 右门
3: 左路射击
4: 中路射击
5: 右路射击
6: 宝箱优先
```

奖励函数以“阻止覆盖 + 消灭蛇身 + 正确投资门”为核心：

```text
r = -0.02 - 0.35 * coverage
    + damage_reward
    + gate_upgrade_reward
    + segment_kill_reward
    + chest_reward
```

终止条件：

```text
success: no pending segments and every body segment is dead or cleared
failure: coverage >= fail_coverage
truncate: steps >= max_steps
```

## 4. RL 算法

当前实现保持项目统一接口，可直接使用既有 DQN/PPO：

```bash
./start.sh --env snake_gate --algo dqn --restart
./start.sh --env snake_gate --algo ppo --restart
```

建议优先使用 DQN：动作空间固定且离散，状态为低维结构化变量，适合快速验证门升级策略和 lane 选择策略。
