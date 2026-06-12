"""2048 —— 纯 Python 自制,带随机性的经典数字合成游戏。

4x4 网格,每步向一个方向推,相同数字相撞合并翻倍;每次有效移动后
在随机空格刷一个新数字(90% 是 2,10% 是 4)—— 这是随机性来源:
同一套动作每局结果都不同,agent 只能学"怎么把大数往角落攒"的策略。

观测 260 维:
- 256 维:每格 one-hot(16 档:空、2、4、…、32768)。比 log 标量
  好学得多——"两格相等"、"这行递增"这类模式判断对 MLP 来说在
  one-hot 上是线性可分的,在连续标量上则要自己学相等性检测。
- 4 维:四个方向当前是否可推(无效动作掩码,免得网络自己猜)

动作 4 个:上 / 右 / 下 / 左。选了不改变棋盘的无效方向,罚 0.1 分后
**随机替走一个有效方向**——不能让棋盘原样不动:观测不变的话,
确定性策略的 argmax 也不变,会原地死循环把局憋死(实测过,贪心
评估时局局如此)。替走机制让死循环在机制上不可能发生,小罚仍在
教网络利用观测里的掩码避开无效方向。

奖励:每次合并按合并值 / 100 给分,无效移动 -0.1,无路可走 -2;
合出 SUCCESS_TILE 视为成功,+50 并结束回合。
另加**势函数塑形** ΔΦ:Φ = 单调性 + 压角,把"维持棋盘结构"这种
几百步后才兑现的价值折现成每步小奖励——不塑形的话,贪心地见合就合
每步都有分拿,但会毁掉棋盘结构,agent 很难自己越过这个局部最优。
差分形式(只算 Φ 的变化量)基本不改变最优策略,只是加快收敛。
"""
import numpy as np

from .base import BaseEnv

SIZE = 4
SUCCESS_TILE = 2048
INVALID_PENALTY = -0.1
N_CHANNELS = 16     # one-hot 档位:空 + 2^1..2^15
MONO_W = 0.05       # 单调性塑形权重(行列按大小排开)
CORNER_W = 0.2      # 压角塑形权重(最大数待在角落)
# 方向编号: 0 上, 1 右, 2 下, 3 左


def _slide_row_left(row):
    """一行向左滑动+合并,返回 (新行, 本次合并得分)。"""
    tiles = [int(v) for v in row if v]   # 转纯 int,防 numpy 类型混进得分
    out, gained, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gained += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (SIZE - len(out)), gained


def _move(board, d):
    """向方向 d 推一次,返回 (新棋盘, 合并得分)。先把方向 d 变换成
    "向左",滑完再逆变换回来。"""
    b = board
    if d == 0:
        b = b.T
    elif d == 1:
        b = b[:, ::-1]
    elif d == 2:
        b = b.T[:, ::-1]
    rows, gained = [], 0
    for r in b:
        nr, g = _slide_row_left(list(r))
        rows.append(nr)
        gained += g
    b2 = np.array(rows, dtype=board.dtype)
    if d == 0:
        b2 = b2.T
    elif d == 1:
        b2 = b2[:, ::-1]
    elif d == 2:
        b2 = b2[:, ::-1].T
    return b2, gained


class Game2048Env(BaseEnv):
    obs_dim = SIZE * SIZE * N_CHANNELS + 4
    n_actions = 4
    max_steps = 1500

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((SIZE, SIZE), dtype=np.int32)
        self.score = 0
        self.t = 0
        self.new_tile = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((SIZE, SIZE), dtype=np.int32)
        self.score = 0
        self.t = 0
        self.new_tile = None
        self._spawn()
        self._spawn()
        self.frames = []
        if self.record:
            self._record_frame(dead=False)
        return self._obs()

    def step(self, action):
        d = int(action)
        self.t += 1
        reward = 0.0
        mask = self._valid_mask()
        if not mask[d]:
            reward = INVALID_PENALTY
            d = int(self.rng.choice(np.flatnonzero(mask)))

        phi_before = self._potential()
        self.board, gained = _move(self.board, d)
        self.score += gained
        self._spawn()
        reward += gained / 100.0 + self._potential() - phi_before

        success = False
        dead = False
        if int(self.board.max()) >= SUCCESS_TILE:
            success = True
            reward += 50.0
        elif not any(self._valid_mask()):   # 棋满且无可合并
            dead = True
            reward -= 2.0

        terminated = dead or success
        truncated = (not terminated) and self.t >= self.max_steps
        info = {"success": success, "score": self.score,
                "max_tile": int(self.board.max())}
        if self.record:
            self._record_frame(dead=dead)
        return self._obs(), reward, terminated, truncated, info

    def _spawn(self):
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            self.new_tile = None
            return
        y, x = empty[int(self.rng.integers(len(empty)))]
        self.board[y, x] = 2 if self.rng.random() < 0.9 else 4
        self.new_tile = (int(x), int(y))

    def _valid_mask(self):
        return [not np.array_equal(_move(self.board, d)[0], self.board)
                for d in range(4)]

    def _potential(self):
        """棋盘结构势函数:单调性 + 最大数压角,供差分塑形用。"""
        logs = np.where(self.board > 0,
                        np.log2(np.maximum(self.board, 1)), 0.0)
        # 单调性:每行每列,完全单调时 = 整条线的总变差,锯齿状趋近 0
        mono = 0.0
        for line in list(logs) + list(logs.T):
            d = np.diff(line)
            incr = d[d > 0].sum()
            decr = -d[d < 0].sum()
            mono += abs(incr - decr)
        # 压角:最大数待在四角之一才给,数越大给越多
        top = logs.max()
        corners = (logs[0, 0], logs[0, -1], logs[-1, 0], logs[-1, -1])
        corner = top if top > 0 and top in corners else 0.0
        return MONO_W * mono + CORNER_W * corner

    def _obs(self):
        cells = np.zeros((SIZE * SIZE, N_CHANNELS), dtype=np.float32)
        idx = np.where(self.board > 0,
                       np.log2(np.maximum(self.board, 1)), 0).astype(int)
        cells[np.arange(SIZE * SIZE), idx.ravel()] = 1.0
        mask = np.array(self._valid_mask(), dtype=np.float32)
        return np.concatenate([cells.ravel(), mask])

    def _record_frame(self, dead):
        self.frames.append({
            "b": self.board.tolist(),
            "n": list(self.new_tile) if self.new_tile else None,
            "score": self.score,
            **({"dead": True} if dead else {}),
        })

    def render_spec(self):
        return {
            "type": "2048",
            "size": SIZE,
            "goal": SUCCESS_TILE,
            "frame_dt": 0.18,
        }
