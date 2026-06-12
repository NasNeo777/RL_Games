# 强化学习实验室

二阶摆(Acrobot)甩起并**稳定倒立**任务 + 可扩展的环境/算法框架 + 网页演示界面。

## 快速开始

一键同时启动训练 + 演示界面(自动打开浏览器):

```bash
./start.sh                       # 默认 PPO,练到学会为止
./start.sh --algo dqn --forever  # 换 DQN,学会后也不停
PORT=8888 ./start.sh --resume    # 换端口 + 断点续练
```

Ctrl+C 同时停掉两者;训练自然结束后服务器会留着供演示。也可以分开跑:

```bash
# 训练(默认 PPO,练到评估中每次都能摆上去为止)
.venv/bin/python -m rl_lab.train --env double_pendulum --algo ppo

# 换 DQN
.venv/bin/python -m rl_lab.train --env double_pendulum --algo dqn

# 解决后也不停,持续优化;可随时 Ctrl+C,加 --resume 接着练
.venv/bin/python -m rl_lab.train --algo ppo --forever
.venv/bin/python -m rl_lab.train --algo ppo --resume

# 演示界面(另开一个终端,与训练同时跑)
.venv/bin/python -m rl_lab.server        # 打开 http://localhost:8000
```

训练过程把检查点写进 `runs/<env>_<algo>/`:

| 文件 | 含义 |
|---|---|
| `latest.pt` | 最新模型(每次评估覆盖) |
| `best.pt` | 历史最优模型(评估回报创新高时覆盖) |
| `metrics.jsonl` | 评估记录,网页训练曲线的数据源 |
| `meta.json` | 当前状态摘要 |

网页端可以切换不同 run、切换最优/最新模型、循环播放演示动画,并每 5 秒刷新训练曲线。

## 内置环境

| 名字 | 任务 | 成功条件 |
|---|---|---|
| `double_pendulum` | 二阶摆甩起 + 稳定倒立 | 倒立区连续保持 5 秒 |
| `mountain_car` | 小车爬山(MountainCar-v0 物理) | 登顶(位置 ≥ 0.5) |
| `flappy_bird` | Flappy Bird([flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) 集成包) | 连过 20 根管道 |
| `mario` | 超级马里奥世界 1-1([gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) 集成包,图像观测) | 拿到关底旗子 |
| `snake` | 贪吃蛇(纯 Python 自制,食物位置随机刷新) | 吃到 30 个食物 |
| `2048` | 2048(纯 Python 自制,新数字随机刷新) | 合出 2048 |
| `tetris` | 俄罗斯方块(纯 Python 自制,7-bag 随机发牌) | 消满 40 行 |

```bash
./start.sh --env flappy_bird --algo dqn
./start.sh --env mario              # 图像观测,只支持 ppo(自动用 CNN 策略)
./start.sh --env snake --algo dqn   # 带随机性:每局食物序列都不同
./start.sh --env 2048 --algo dqn
./start.sh --env 2048 --algo td2048   # 2048 专用 afterstate TD,快得多
./start.sh --env tetris --algo dqn
```

`snake` / `2048` / `tetris` 是三个**全程带随机性**的自制环境(零依赖,
实现见 `envs/snake.py`、`envs/game_2048.py`、`envs/tetris.py`),agent
无法靠背动作序列通关,只能学出泛化策略:

- `snake`:食物每次被吃掉后随机刷新位置。观测 10 维(以蛇头朝向为
  参考系:三方向障碍距离、食物相对偏移、朝向、长度),动作是相对转向
  (直行/左转/右转)。
- `2048`:每次有效移动后在随机空格刷新数字(90% 是 2、10% 是 4)。
  观测 260 维(每格 16 档 one-hot + 4 个方向的有效动作掩码),动作 4 个
  方向;另有"行列单调 + 最大数压角"的势函数差分塑形,把维持棋盘结构
  的长线价值折现成每步小奖励。**推荐用 `--algo td2048` 训练**:2048
  专用的 afterstate TD + N-tuple 查表(`algos/td2048.py`,Szubert &
  Jaśkowski 2014 配方),只评估"推完之后、刷新之前"的棋盘价值,随机
  刷新的噪声被 TD 平均掉,比通用 DQN/PPO 快几个数量级。
- `tetris`:方块按 7-bag 随机发牌。为了让 MLP 学得动,动作空间用经典
  的"落点选择"方案(旋转 4 x 落点列 10 = 40 个动作,一步落一块),
  观测 34 维(每列高度、每列洞数、当前块和下一块的 one-hot)。

`mario` 是本项目第一个**图像观测**环境:agent 不直接看 240x256 RGB
原始画面,观测经过跳帧 4(相邻两帧取 max 去闪烁)→ 灰度化 →
缩小栅格化到 84x84 → 叠 4 帧,最终是 (4, 84, 84) uint8
(经典 Atari 式预处理,实现见 `envs/mario.py`,PPO 自动切 CnnPolicy)。
训练用图像专属超参(`algos/sb3_ppo.py` 的 `HP_IMAGE`,参考已验证的
马里奥 PPO 配方:lr 2.5e-4、gamma 0.95、clip 0.1、target_kl 0.03、
熵系数 0.1 线性衰减到 0.005),并默认开 8 个并行环境采样
(`--n-envs` 可调)。
网页演示左右分屏:左边原始彩色画面(仅供观看),右边同步显示
agent 网络真正看到的 84x84 灰度观测。
依赖(首次使用前装一次):

```bash
.venv/bin/pip install gym-super-mario-bros gym   # gym 仅为包内注册所需
```

## 任务说明:二阶摆甩起 + 稳定倒立

两杆吊在固定支点,只有两杆之间的关节有电机(欠驱动),
动作是 5 档力矩 {-2, -1, 0, +1, +2},控制频率 20Hz。
目标:尽快甩到倒立位置(末端高度 > 1.9 / 满高 2.0,且角速度小)
并**连续稳定保持 5 秒**才算成功。

奖励 = 高度塑形(小)- 顶部超速惩罚 + 倒立区每步 +1 + 成功奖金;
成功奖金随用时减少而增加,所以训练会同时优化"稳得住"和"摆得快"。
动力学与 Gymnasium Acrobot-v1 相同(Sutton & Barto 方程,RK4 积分)。

## 扩展:接入新小游戏

**推荐方式:用 Gymnasium 集成包**(不自己写游戏逻辑)。
继承 `envs/gym_adapter.py` 的 `GymEnv`,几行就够:

```python
# rl_lab/envs/my_game.py
from .gym_adapter import GymEnv

class MyGameEnv(GymEnv):
    env_id = "CartPole-v1"          # gym.make 的环境 id
    import_module = None             # 第三方包需要 import 注册时填包名
    max_steps = 500

    def is_success(self, info):     # 按游戏定义"成功"
        return False
```

然后在 `rl_lab/envs/__init__.py` 的 `ENVS` 登记名字即可。
演示画面由适配器自动抓 `rgb_array` 压成 JPEG 帧,前端用通用
`video` 渲染器播放,**不需要写任何前端代码**。`flappy_bird` 就是这么接的。

自己写物理的环境(如本项目的 `double_pendulum`)走另一条路:
继承 `BaseEnv` 实现 `reset / step / render_spec` 并录制状态帧,
再在 `web/index.html` 的 `RENDERERS` 加一个对应 canvas 画法。

接好后直接 `python -m rl_lab.train --env 新名字 --algo ppo` 开练。

## 算法

| 名字 | 实现 |
|---|---|
| `ppo` | **Stable-Baselines3 PPO**(`algos/sb3_ppo.py` 适配层) |
| `dqn` | 手写 Double DQN |
| `ppo_custom` | 旧的手写 PPO,保留以兼容历史检查点 |

## 扩展:接入新算法

两种方式:

- **自己实现**:实现 `rl_lab/algos/base.py` 的 `BaseAgent` 接口
  (`act / observe / update / state_dict / load_state_dict`),在
  `rl_lab/algos/__init__.py` 的 `ALGOS` 登记。
- **接外部库(如 SB3 的其他算法)**:参考 `algos/sb3_ppo.py`,
  设 `trains_itself = True` 并实现 `train_loop()`,由外部库自跑训练循环,
  评估/检查点/曲线数据通过回调按框架格式落盘,界面无感知差异。
  环境侧用 `envs/to_gym.py` 把 BaseEnv 包成标准 gymnasium.Env 即可。

## 目录结构

```
rl_lab/
  envs/        环境(注册表 + 二阶摆)
  algos/       算法(注册表 + DQN + PPO)
  train.py     持续训练入口
  server.py    演示服务器(纯标准库)
  web/         网页界面
runs/          训练产物(检查点、曲线数据)
```
