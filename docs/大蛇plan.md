下面给你一版**开发文档 v0.1**。目标是两个：

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
