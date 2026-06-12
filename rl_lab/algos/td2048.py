"""2048 专用:afterstate TD(0) + N-tuple 查表(Szubert & Jaśkowski 2014)。

为什么值得单开一个算法
----------------------
2048 的一步天然分成两段:**确定性的推合**(棋盘怎么滑、怎么合并,
完全由动作决定)+ **随机的刷新**(新数字落在哪、是 2 还是 4)。
通用的 Q(s, a) 必须把这两段一起学,随机刷新让目标值噪声很大;
afterstate 学习只评估"推完之后、刷新之前"的棋盘 ŝ 的价值 V(ŝ):

    决策:  a* = argmax_a [ 合并得分(a) + V(ŝ_a) ]
    更新:  V(ŝ) ← V(ŝ) + α·[ max_a'(r' + V(ŝ'_a')) − V(ŝ) ]

随机性被 TD 的样本平均自然吸收,网络再也不用预测"骰子"。

价值函数不用神经网络,用 **N-tuple 查表**:17 条 4-tuple(4 行 +
4 列 + 9 个 2x2 方块),每条是一张 16^4 的小表,V = 所有表项之和;
同一组表在棋盘的 8 个对称视角下共享(旋转/镜像后局面价值相同),
样本效率 x8。查表没有梯度下降的迭代损耗,一条经验立刻生效——
这是文献里真正能稳定合出 2048 的配方。

与框架的关系
------------
- 只适用于 ``--env 2048``(从 one-hot 观测解码棋盘,借用环境的
  _move 模拟推合),构造时校验观测维度。
- TD 更新用**自己算的原始合并得分**,不用环境的塑形奖励——塑形是
  给"看不清几百步外"的神经网络準备的拐杖,查表 TD 不需要。
  训练曲线里的评估回报仍按环境奖励统计,与其他算法可比。
- 无效方向在 act 里直接屏蔽,永远只输出有效动作。
"""
import numpy as np

from ..envs.game_2048 import SIZE, _move
from .base import BaseAgent

ALPHA = 0.25          # 单次 TD 把 V(ŝ) 朝目标拉近的总比例(摊到所有表项)
EXP_MAX = 15          # 指数封顶:2^15 = 32768,4x4 棋盘实际到不了更高


def _build_feature_maps():
    """返回 (CELLS, TID, POW):
    CELLS[k] = 第 k 个 (对称视角 x tuple) 要读的 4 个棋盘扁平下标,
    TID[k]   = 它写哪张表,POW = 4 个格子的进制权重。"""
    tuples = []
    for y in range(SIZE):                              # 4 行
        tuples.append([y * SIZE + x for x in range(SIZE)])
    for x in range(SIZE):                              # 4 列
        tuples.append([y * SIZE + x for y in range(SIZE)])
    for y in range(SIZE - 1):                          # 9 个 2x2 方块
        for x in range(SIZE - 1):
            tuples.append([y * SIZE + x, y * SIZE + x + 1,
                           (y + 1) * SIZE + x, (y + 1) * SIZE + x + 1])

    base = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
    views = []
    for b in (base, np.fliplr(base)):
        for k in range(4):
            views.append(np.rot90(b, k).ravel())       # 8 个对称视角

    cells, tid = [], []
    for view in views:
        for t, tup in enumerate(tuples):
            cells.append(view[tup])
            tid.append(t)
    pow_ = np.array([(EXP_MAX + 1) ** i for i in range(3, -1, -1)],
                    dtype=np.int64)
    return (np.array(cells), np.array(tid), pow_, len(tuples))


CELLS, TID, POW, N_TABLES = _build_feature_maps()
N_FEATURES = len(CELLS)                                # 8 视角 x 17 tuple = 136


class TD2048Agent(BaseAgent):
    name = "td2048"

    def __init__(self, obs_dim, n_actions, device="cpu", seed=None):
        super().__init__(obs_dim, n_actions, device)
        expected = SIZE * SIZE * (EXP_MAX + 1) + 4
        if obs_dim != expected or n_actions != 4:
            raise SystemExit(
                f"td2048 是 2048 专用算法(期望观测 {expected} 维/动作 4 个,"
                f"拿到 {obs_dim}/{n_actions}),请配合 --env 2048 使用")
        self.tables = np.zeros((N_TABLES, (EXP_MAX + 1) ** 4),
                               dtype=np.float32)
        self._pending = None
        self.updates = 0
        self._last_err = 0.0

    # ---- N-tuple 价值 ----
    def _value(self, board):
        idx = self._indices(board)
        return float(self.tables[TID, idx].sum())

    def _indices(self, board):
        exps = np.zeros(SIZE * SIZE, dtype=np.int64)
        flat = board.ravel()
        nz = flat > 0
        exps[nz] = np.minimum(np.log2(flat[nz]).astype(np.int64), EXP_MAX)
        return (exps[CELLS] * POW).sum(axis=1)

    def _best_move(self, board):
        """返回 (动作, 合并得分 + V(afterstate));无有效动作时 (None, 0)。"""
        best_a, best_v = None, 0.0
        for d in range(4):
            after, gained = _move(board, d)
            if np.array_equal(after, board):
                continue
            v = gained + self._value(after)
            if best_a is None or v > best_v:
                best_a, best_v = d, v
        return best_a, best_v

    @staticmethod
    def _board_from_obs(obs):
        exps = np.asarray(obs)[:SIZE * SIZE * (EXP_MAX + 1)] \
            .reshape(SIZE * SIZE, EXP_MAX + 1).argmax(axis=1)
        return np.where(exps > 0, 2 ** exps, 0) \
            .astype(np.int32).reshape(SIZE, SIZE)

    # ---- 框架接口 ----
    def act(self, obs, deterministic=False):
        # 纯贪心:刷新本身的随机性已提供足够探索(文献配方,无需 ε)
        a, _ = self._best_move(self._board_from_obs(obs))
        return a if a is not None else 0

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        self._pending = (obs, int(action), next_obs, terminated)

    def update(self):
        if self._pending is None:
            return None
        obs, action, next_obs, terminated = self._pending
        self._pending = None

        board = self._board_from_obs(obs)
        after, _ = _move(board, action)
        if np.array_equal(after, board):    # 防御:无效动作(本算法不会选)
            return None
        if terminated:
            target = 0.0                    # 终局(死局或合出 2048)后无后续价值
        else:
            _, target = self._best_move(self._board_from_obs(next_obs))

        idx = self._indices(after)
        err = target - float(self.tables[TID, idx].sum())
        # 同一表项可能被多个特征命中(对称重合),用 np.add.at 保证累加
        np.add.at(self.tables, (TID, idx), ALPHA * err / N_FEATURES)

        self.updates += 1
        self._last_err = err
        if self.updates % 2000 == 0:
            return {"td_err": round(err, 2),
                    "value_scale": round(float(np.abs(self.tables).max()), 1)}
        return None

    def state_dict(self):
        return {"tables": self.tables}

    def load_state_dict(self, sd):
        self.tables = np.asarray(sd["tables"], dtype=np.float32)
