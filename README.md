# 强化学习实验室

二阶摆(Acrobot)甩摆任务 + 可扩展的环境/算法框架 + 网页演示界面。

## 快速开始

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

## 任务说明:二阶摆甩摆

两杆吊在固定支点,只有两杆之间的关节有电机(欠驱动),动作是力矩 {-1, 0, +1}。
奖励为末端高度(稠密塑形),末端甩过高度 1.5(满高 2.0)记为成功,+50 并结束回合。
动力学与 Gymnasium Acrobot-v1 相同(Sutton & Barto 方程,RK4 积分)。

## 扩展:接入新小游戏

1. 在 `rl_lab/envs/` 新建文件,继承 `BaseEnv`(见 `envs/base.py` 的接口说明),
   实现 `reset / step / render_spec`,并在 `record=True` 时往 `self.frames` 记录渲染帧;
2. 在 `rl_lab/envs/__init__.py` 的 `ENVS` 注册表登记名字;
3. 在 `rl_lab/web/index.html` 的 `RENDERERS` 里按 `render_spec()["type"]`
   加一个 canvas 画法(输入是你记录的 frames);
4. `python -m rl_lab.train --env 新名字 --algo ppo` 直接开练。

## 扩展:接入新算法

实现 `rl_lab/algos/base.py` 的 `BaseAgent` 接口
(`act / observe / update / state_dict / load_state_dict`),
在 `rl_lab/algos/__init__.py` 的 `ALGOS` 登记即可。

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
