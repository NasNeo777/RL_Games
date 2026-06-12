"""2048 —— 纯 Python 自制,带随机性的经典数字合成游戏。

4x4 网格,每步向一个方向推,相同数字相撞合并翻倍;每次有效移动后
在随机空格刷一个新数字(90% 是 2,10% 是 4)—— 这是随机性来源:
同一套动作每局结果都不同,agent 只能学"怎么把大数往角落攒"的策略。

观测 20 维(均归一化):
- 16 维:每格数字的 log2 / 16(空格为 0)
- 4 维:四个方向当前是否可推(无效动作掩码,免得网络自己猜)

动作 4 个:上 / 右 / 下 / 左。推了不改变棋盘算无效移动,小罚不刷新;
连续 INVALID_LIMIT 次无效直接判负(防演示时死循环)。

奖励:每次合并按合并值 / 100 给分,无效移动 -0.1,无路可走 -2;
合出 SUCCESS_TILE 视为成功,+50 并结束回合。
"""
import numpy as np

from .base import BaseEnv

SIZE = 4
SUCCESS_TILE = 2048
INVALID_LIMIT = 10
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
    obs_dim = SIZE * SIZE + 4
    n_actions = 4
    max_steps = 1500

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((SIZE, SIZE), dtype=np.int32)
        self.score = 0
        self.t = 0
        self.invalid = 0
        self.new_tile = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((SIZE, SIZE), dtype=np.int32)
        self.score = 0
        self.t = 0
        self.invalid = 0
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
        new_board, gained = _move(self.board, d)
        moved = not np.array_equal(new_board, self.board)

        success = False
        dead = False
        if not moved:
            reward = -0.1
            self.invalid += 1
            self.new_tile = None
            dead = self.invalid >= INVALID_LIMIT
            if dead:
                reward = -2.0
        else:
            self.invalid = 0
            self.board = new_board
            self.score += gained
            reward = gained / 100.0
            self._spawn()
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

    def _obs(self):
        cells = np.where(self.board > 0, np.log2(np.maximum(self.board, 1)),
                         0.0) / 16.0
        mask = np.array(self._valid_mask(), dtype=np.float32)
        return np.concatenate([cells.ravel(), mask]).astype(np.float32)

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
