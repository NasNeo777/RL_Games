"""环境基类。

所有环境实现 gym 风格接口(reset / step 五元组),并额外提供:

- ``render_spec()``: 告诉前端用哪个渲染器、画布参数等(静态信息)。
- ``record`` 开关 + ``frames``: 开启后 step 会把(细分的)物理状态记录下来,
  服务器跑完一局直接把 frames 发给网页做动画。

新增小游戏:在 envs/ 下新建文件实现本接口,然后在 envs/__init__.py 的
ENVS 注册表里登记,再在 web/index.html 的 RENDERERS 里加一个对应的画法。
"""


class BaseEnv:
    # 子类必须设置
    obs_dim: int = 0
    n_actions: int = 0
    max_steps: int = 500
    # 图像观测的环境设置成 (通道, 高, 宽),观测为 uint8;
    # 向量观测的环境保持 None(观测为 obs_dim 维 float32)
    obs_shape: tuple = None

    def __init__(self):
        self.record = False
        self.frames = []

    def reset(self, seed=None):
        """返回初始观测 obs (np.float32 数组)。"""
        raise NotImplementedError

    def step(self, action):
        """返回 (obs, reward, terminated, truncated, info)。

        terminated: 任务本身结束(成功/失败),价值不再 bootstrap。
        truncated: 仅因步数上限截断,价值仍 bootstrap。
        """
        raise NotImplementedError

    def render_spec(self):
        """前端渲染所需的静态描述,至少包含 {"type": <渲染器名>}。"""
        raise NotImplementedError
