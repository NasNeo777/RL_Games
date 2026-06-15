<div align="center">

# 🕹️ 强化学习实验室

**挑一个游戏 → 敲一行命令 → 看 AI 从乱玩到精通**

训练全程可视化:浏览器里看演示动画、训练曲线实时爬升

<img src="https://img.shields.io/badge/Python-3.13-3776ab?logo=python&logoColor=white" alt="Python">&nbsp;<img src="https://img.shields.io/badge/PyTorch-DQN%20·%20PPO-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">&nbsp;<img src="https://img.shields.io/badge/游戏-8_款-66bb6a" alt="games">&nbsp;<img src="https://img.shields.io/badge/算法-4_种-4fc3f7" alt="algos">&nbsp;<img src="https://img.shields.io/badge/前端依赖-0-8b98a5" alt="zero-deps">

</div>

---

## 🚀 三十秒开局

```bash
./start.sh --env snake --algo dqn
```

一行命令同时启动 **训练** + **演示网页**(自动打开浏览器,
[localhost:8000](http://localhost:8000))。训练边跑,网页边播 AI 的最新表现,
每 5 秒刷新曲线。<kbd>Ctrl C</kbd> 一键全停;再次运行**自动断点续练**。

---

## 🎮 游戏厅:今天想看 AI 学什么?

<table>
<tr>
<td width="50%" valign="top">

<h3 align="center">🐍 贪吃蛇</h3>
<p align="center"><i>食物随机刷新,追着吃、别咬自己</i></p>
<p align="center"><code>./start.sh --env snake --algo dqn</code></p>
<p align="center">🏆 吃满 30 个食物 &nbsp;·&nbsp; ⏱️ <b>约 2 分钟</b>见通关局<br>🎲 随机性:食物位置 &nbsp;·&nbsp; 🔰 新手首选,进步肉眼可见</p>

</td>
<td width="50%" valign="top">

<h3 align="center">🔢 2048</h3>
<p align="center"><i>推格子合数字,AI 学会蛇形压角</i></p>
<p align="center"><code>./start.sh --env 2048 --algo td2048</code></p>
<p align="center">🏆 合出 2048 &nbsp;·&nbsp; ⏱️ <b>约 7 分钟</b>开始合出 2048<br>🎲 随机性:刷新的数字与位置 &nbsp;·&nbsp; ⭐ 专用算法,镇馆之宝</p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3 align="center">🧱 俄罗斯方块</h3>
<p align="center"><i>7-bag 随机发牌,学会铺平少留洞</i></p>
<p align="center"><code>./start.sh --env tetris --algo dqn</code></p>
<p align="center">🏆 消满 40 行(经典竞速) &nbsp;·&nbsp; ⏱️ <b>约 8 分钟练成</b>(实测 57 万步)<br>🎲 随机性:方块序列 &nbsp;·&nbsp; 🧠 动作 = 选旋转 × 落点,一步落一块</p>

</td>
<td width="50%" valign="top">

<h3 align="center">🐦 Flappy Bird</h3>
<p align="center"><i>就是那个让全世界摔手机的游戏</i></p>
<p align="center"><code>./start.sh --env flappy_bird --algo dqn</code></p>
<p align="center">🏆 连过 777 根管道(≈永生) &nbsp;·&nbsp; ⏱️ 几分钟像样,练成要挂机(PPO 实测 2300 万步)<br>🎲 随机性:管道高度 &nbsp;·&nbsp; 📦 需装:<code>.venv/bin/pip install flappy-bird-gymnasium</code></p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3 align="center">⛰️ 小车爬山</h3>
<p align="center"><i>引擎不够劲,得先左右荡秋千攒动能</i></p>
<p align="center"><code>./start.sh --env mountain_car --algo dqn</code></p>
<p align="center">🏆 登顶(位置 ≥ 0.5) &nbsp;·&nbsp; ⏱️ <b>约 1 分钟练成</b>(实测 1.6 万步)<br>📐 经典控制问题,MountainCar-v0 同款物理</p>

</td>
<td width="50%" valign="top">

<h3 align="center">🎯 二阶摆(默认)</h3>
<p align="center"><i>甩起来,倒立,稳住 5 秒——欠驱动控制的硬骨头</i></p>
<p align="center"><code>./start.sh</code></p>
<p align="center">🏆 倒立区连续稳定 5 秒 &nbsp;·&nbsp; ⏱️ 硬骨头:挂机数小时(实测 1.5 亿步)<br>📐 Acrobot 同款动力学(RK4 积分),本项目的元老环境</p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3 align="center">🍄 超级马里奥 1-1</h3>
<p align="center"><i>AI 直接看像素玩马里奥,硬核玩家区</i></p>
<p align="center"><code>./start.sh --env mario</code></p>
<p align="center">🏆 拿到关底旗子 &nbsp;·&nbsp; ⏱️ <b>约一夜</b>(实测 427 万步),建议挂机<br>🖼️ 图像观测,只支持 PPO(自动切 CNN)<br>📦 需装:<code>.venv/bin/pip install gym-super-mario-bros gym</code></p>

</td>
<td width="50%" valign="top">

<h3 align="center">🕹️ 跳一跳</h3>
<p align="center"><i>蓄力一跳,稳稳落在下一块台子上,差一点就摔进缝里</i></p>
<p align="center"><code>./start.sh --env jump --algo dqn</code></p>
<p align="center">🏆 连续踩中 25 块台子 &nbsp;·&nbsp; ⏱️ <b>约 1 分钟练成</b>(实测约 4000 步)<br>🎲 随机性:台距与台宽 &nbsp;·&nbsp; 🎯 看缺口调力度,带容差的回归题</p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3 align="center">🕹️ 跳一跳 PPO</h3>
<p align="center"><i>先识别缺口和台宽,再让 PPO 预测合适的蓄力档位</i></p>
<p align="center"><code>./start.sh --env jump --algo ppo</code></p>
<p align="center">🏆 连续踩中 25 块台子 &nbsp;·&nbsp; ⏱️ 作为对照算法可直接开练<br>📐 结构化观测 &nbsp;·&nbsp; 🤖 和 ADB 真机脚本的“检测后决策”链路一致</p>

</td>
<td width="50%" valign="top">

<h3 align="center">➕ 你的游戏?</h3>
<p align="center"><i>三五行接入任何 Gymnasium 游戏</i></p>
<p align="center"><code>详见下方「接入新游戏」</code></p>
<p align="center">🛠️ 继承 <code>GymEnv</code> 填个 env_id 就能跑<br>前端零代码——演示画面自动录制播放</p>

</td>
</tr>
</table>

> 💡 **算法怎么选?** 向量观测的游戏 `dqn` / `ppo` 都行;`2048` 务必用专属的
> `td2048`(比通用算法快几个数量级);`mario` 这类图像观测只支持 `ppo`。

---

## 🖥️ 演示网页都能干什么

| 功能 | 说明 |
|---|---|
| 🎬 模型演示 | 循环播放当前模型的完整对局,可调 0.5×~4× 速度 |
| 🥇 最优 / 最新切换 | `best.pt`(历史最高分)vs `latest.pt`(刚出炉的) |
| ♻️ 跑完刷新模型 | 每播完一局自动加载新检查点——训练进步实时可见 |
| 📈 训练曲线 | 评估回报 + 成功率,每 5 秒刷新 |
| 🎯 练成进度条 | 按本仓库实测的步数预算估算进度和剩余时间 |
| 🔀 多 run 切换 | 同时训练多个游戏/算法,下拉框随意切换围观 |

---

## ⚙️ 训练命令进阶

```bash
# 学会后也不停,继续优化
./start.sh --env snake --algo dqn --forever

# 不要断点续练,从零重训(旧目录自动备份为 *_old)
./start.sh --env 2048 --algo td2048 --restart

# 换端口 / 不自动开浏览器
PORT=8888 ./start.sh
NO_OPEN=1 ./start.sh

# 训练和演示也可以分开跑
.venv/bin/python -m rl_lab.train --env tetris --algo dqn
.venv/bin/python -m rl_lab.server --port 8000
```

### 🤖 附加件:跳一跳真机部署(单个环境的额外能力)

> 这是 **`jump` 这一个环境** 的真机落地附加件,**不是项目主体**。把训练好的
> 跳一跳 agent 部署到真手机:截图 → YOLO 检测棋子/台子 → PPO 决定力度 → adb 长按。
> 完整步骤、输入输出、排错见 **[docs/jump_yolo_pipeline.md](docs/jump_yolo_pipeline.md)**,
> 工具索引见 **[tools/README.md](tools/README.md)**。

```bash
.venv/bin/pip install ultralytics

# ① 生成合成训练数据(推荐,标签完美;另有真图标注路线见文档)
.venv/bin/python tools/gen_synthetic_jump.py --n 2000 --out datasets/jump_synth

# ② 训练 YOLO 检测器(Mac 用 mps,无 GPU 用 cpu)
.venv/bin/python tools/train_jump_yolo.py --data datasets/jump_synth/dataset.yaml \
    --device mps --name jump_yolo_synth --project runs/detect/runs

# ③ 真机运行(--dry-run 只测检测不真按)
.venv/bin/python adb_jump_ppo.py --serial <adb-serial> \
    --yolo-model runs/detect/runs/jump_yolo_synth/weights/best.pt
```

训练产物写进 `runs/<env>_<algo>/`:

| 文件 | 含义 |
|---|---|
| `latest.pt` | 最新模型(每次评估覆盖) |
| `best.pt` | 历史最优模型(评估回报创新高时覆盖) |
| `metrics.jsonl` | 评估记录,网页曲线的数据源 |
| `meta.json` | 当前状态摘要 |

> ⚠️ 改了 `envs/` 或 `algos/` 的代码后,演示服务器要重启才会加载新模块。

---

## 🧠 算法

| 名字 | 实现 | 适用 |
|---|---|---|
| `dqn` | 手写 Double DQN(经验回放 + ε 贪心) | 所有向量观测游戏 |
| `ppo` | Stable-Baselines3 PPO(`algos/sb3_ppo.py` 适配层) | 全部游戏,图像观测唯一选择 |
| `td2048` | **2048 专用** afterstate TD(0) + N-tuple 查表 | 仅 `2048` |
| `ppo_custom` | 旧手写 PPO,保留以兼容历史检查点 | — |

<details>
<summary><b>🔍 为什么 2048 要专用算法?(点开看今天最有意思的结论)</b></summary>

2048 的一步天然分两段:**确定性的推合** + **随机的数字刷新**。
通用 Q(s,a) 必须连"骰子"一起预测,目标噪声大,DQN 练 410 万步平均才
4000 多分;**afterstate 学习**只评估"推完之后、刷新之前"的棋盘价值
V(ŝ),决策时对四个方向做一步前瞻 `argmax [合并得分 + V(ŝ)]`,随机性
被 TD 平均掉。价值函数不用神经网络,用 17 条 4-tuple 查表(行、列、
2×2 方块)在 8 个对称视角下共享——查表更新一条经验立刻生效。

实测:**训练 7 分钟(约 100 万步)平均 15000+ 分,20% 的局合出 2048**,
是 DQN 同步数成绩的近 20 倍(Szubert & Jaśkowski 2014 配方,
实现见 `algos/td2048.py`)。瓶颈从来不在观测编码或奖励塑形,
在算法结构是否绕开了随机性。

</details>

<details>
<summary><b>🍄 mario 的图像管线(本项目第一个像素观测环境)</b></summary>

agent 不直接看 240×256 RGB 原始画面:跳帧 4(相邻两帧取 max 去闪烁)
→ 灰度化 → 缩到 84×84 → 叠 4 帧,最终 (4, 84, 84) uint8(经典 Atari
式预处理,见 `envs/mario.py`)。训练用图像专属超参(`algos/sb3_ppo.py`
的 `HP_IMAGE`:lr 2.5e-4、gamma 0.95、clip 0.1、熵系数 0.1 线性衰减),
默认 8 个并行环境采样(`--n-envs` 可调)。网页演示左右分屏:左边原始
彩色画面,右边是 agent 网络真正看到的 84×84 灰度观测。

</details>

<details>
<summary><b>🎲 自制随机环境的设计(snake / 2048 / tetris / jump)</b></summary>

全程带随机性的纯 Python 环境(零依赖),agent 背不下动作序列,
只能学泛化策略:

- **snake**(`envs/snake.py`):食物随机刷新。观测 10 维,以蛇头朝向
  为参考系(三方向障碍距离、食物相对偏移、朝向、长度);动作是相对
  转向(直行/左转/右转),天然无"瞬间掉头"废动作。
- **2048**(`envs/game_2048.py`):每步后在随机空格刷 2(90%)或 4(10%)。
  观测 260 维(每格 16 档 one-hot + 4 维有效动作掩码);选了无效方向
  会被罚分并随机替走有效方向——否则观测不变、确定性策略原地死循环。
  另有"行列单调 + 最大数压角"的势函数差分塑形。
- **tetris**(`envs/tetris.py`):7-bag 随机发牌。动作空间用经典的
  "落点选择"方案(旋转 4 × 列 10 = 40 个动作,一步落一块,逐帧操作
  对 MLP 太难);观测 34 维(每列高度、每列洞数、当前块 + 下一块
  one-hot);奖励鼓励连消(1/2/3/4 行 = 1/3/5/8 分)、惩罚造洞。
- **jump**(`envs/jump.py`):跳一跳。每步台距、台宽、方向(左前/右前
  两条等距斜轴随机选)、底座样式都重新随机。观测只有 2 维(到下一块
  台子中心的距离、台子半宽),动作是 41 档蓄力力度 → 跳跃距离;方向与
  样式纯属画面表现(小人自动朝向目标台),不进观测、不影响决策——本质
  是一道带容差的回归题:看缺口调力度,落在台面上即得分(越靠台心奖励
  越高),摔进缝里就结束。DQN 约 4000 步练成。前端是 2.5D 等距视角,
  连跳之间镜头连续平移(还原原版手感)。

</details>

<details>
<summary><b>🎯 二阶摆任务细则</b></summary>

两杆吊在固定支点,只有两杆之间的关节有电机(欠驱动),动作是 5 档
力矩 {-2, -1, 0, +1, +2},控制频率 20Hz。目标:甩到倒立位置
(末端高度大于 1.9 / 满高 2.0,且角速度小)并**连续稳定 5 秒**。
奖励 = 高度塑形(小)− 顶部超速惩罚 + 倒立区每步 +1 + 成功奖金
(用时越短奖金越高,同时优化"稳得住"和"摆得快")。动力学与
Gymnasium Acrobot-v1 相同(Sutton & Barto 方程,RK4 积分)。

</details>

---

## 🛠️ 接入新游戏

**推荐:用 Gymnasium 集成包**(不自己写游戏逻辑),继承
`envs/gym_adapter.py` 的 `GymEnv`,几行就够:

```python
# rl_lab/envs/my_game.py
from .gym_adapter import GymEnv

class MyGameEnv(GymEnv):
    env_id = "CartPole-v1"      # gym.make 的环境 id
    import_module = None         # 第三方包需 import 注册时填包名
    max_steps = 500

    def is_success(self, info):  # 按游戏定义"成功"
        return False
```

然后在 `rl_lab/envs/__init__.py` 的 `ENVS` 登记名字即可开练。
演示画面由适配器自动抓 `rgb_array` 压成 JPEG 帧,前端用通用 `video`
渲染器播放,**一行前端代码都不用写**(`flappy_bird` 就是这么接的)。

自己写物理/逻辑的环境(如 `snake`、`2048`)走另一条路:继承 `BaseEnv`
实现 `reset / step / render_spec` 并录制状态帧,再在 `web/index.html`
的 `RENDERERS` 加一个对应的 canvas 画法。

## 🧩 接入新算法

- **自己实现**:实现 `rl_lab/algos/base.py` 的 `BaseAgent` 接口
  (`act / observe / update / state_dict / load_state_dict`),在
  `rl_lab/algos/__init__.py` 的 `ALGOS` 登记。
- **接外部库**(如 SB3 的其他算法):参考 `algos/sb3_ppo.py`,设
  `trains_itself = True` 并实现 `train_loop()`,由外部库自跑训练循环,
  评估/检查点/曲线通过回调按框架格式落盘,界面无感知差异。环境侧用
  `envs/to_gym.py` 把 BaseEnv 包成标准 gymnasium.Env。

## 📁 目录结构

**主体——多环境 RL 实验室**(项目核心,环境/算法无关):

```
rl_lab/
  envs/        环境(注册表 + 8 款游戏)        ← 加游戏改这里
  algos/       算法(注册表 + DQN / PPO / td2048)  ← 加算法改这里
  train.py     持续训练入口
  server.py    演示服务器(纯标准库,零依赖)
  web/         网页界面(单文件,零依赖)
runs/          训练产物(检查点、曲线数据)
start.sh       一键训练 + 演示
```

**附加件——跳一跳真机部署**(只服务 `jump` 一个环境,可忽略):

```
adb_jump_ppo.py            真机主程序(截图→检测→决策→长按)
tools/                     YOLO 数据/训练工具集  → 见 tools/README.md
docs/jump_yolo_pipeline.md 真机部署完整步骤文档
```

<div align="center">

**🎓 选个游戏,开练!**

<sub>每一条训练曲线的爬升,都是 AI 从"乱按"到"开窍"的过程</sub>

</div>
