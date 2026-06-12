"""俄罗斯方块 —— 纯 Python 自制,随机性来自 7-bag 随机发牌的方块序列。

10x20 标准棋盘。为了让 MLP 算法学得动,采用经典的**落点选择**动作空间:
一步 = 选好旋转和落点列,方块直接落到底(不逐帧操作)。
动作 = 旋转(4) x 列(10) = 40 个;旋转数对不同方块取模,列越界则贴边。

观测 34 维(均归一化):
- 10 维:每列堆叠高度
- 10 维:每列洞数(顶部封死的空格)
- 7 维:当前方块 one-hot
- 7 维:下一个方块 one-hot(支持提前规划)

奖励(参考社区已验证的 Tetris DQN 配方):
- 每放一块 +0.05(活着)
- 消行 1/2/3/4 行分别 +1/+3/+5/+8(鼓励攒多行一起消)
- 每新增一个洞 -0.1(塑形,洞是 Tetris 的万恶之源)
- 堆到顶 -5
- 累计消满 SUCCESS_LINES 行视为成功(经典 40 行竞速),+50 结束。
"""
import numpy as np

from .base import BaseEnv

W, H = 10, 20
SUCCESS_LINES = 40
LINE_REWARDS = [0.0, 1.0, 3.0, 5.0, 8.0]
# 方块 id: 1=I 2=O 3=T 4=S 5=Z 6=J 7=L;每个朝向的格子偏移(x右, y下),
# 均已归一到 min x = min y = 0
SHAPES = {
    1: [[(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)]],
    2: [[(0, 0), (1, 0), (0, 1), (1, 1)]],
    3: [[(0, 0), (1, 0), (2, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (0, 1), (1, 1), (1, 2)]],
    4: [[(1, 0), (2, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2)]],
    5: [[(0, 0), (1, 0), (1, 1), (2, 1)],
        [(1, 0), (0, 1), (1, 1), (0, 2)]],
    6: [[(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(1, 0), (1, 1), (0, 2), (1, 2)]],
    7: [[(2, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2)]],
}


class TetrisEnv(BaseEnv):
    obs_dim = W + W + 7 + 7
    n_actions = 4 * W       # (旋转, 落点列)
    max_steps = 500

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((H, W), dtype=np.int8)
        self.bag = []
        self.cur = 1
        self.nxt = 1
        self.lines = 0
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((H, W), dtype=np.int8)
        self.bag = []
        self.cur = self._draw()
        self.nxt = self._draw()
        self.lines = 0
        self.t = 0
        self.frames = []
        if self.record:
            self._record_frame(placed=[], cleared=[], dead=False)
        return self._obs()

    def step(self, action):
        action = int(action)
        shapes = SHAPES[self.cur]
        cells = shapes[(action // W) % len(shapes)]
        width = max(x for x, _ in cells) + 1
        col = min(action % W, W - width)
        self.t += 1

        # 从顶落到底;顶上都放不进 = 堆满判负
        if not self._fits(cells, col, 0):
            if self.record:
                self._record_frame(placed=[], cleared=[], dead=True)
            info = {"success": False, "score": self.lines}
            return self._obs(), -5.0, True, False, info

        row = 0
        while self._fits(cells, col, row + 1):
            row += 1
        holes_before = self._holes()
        placed = [(col + x, row + y) for x, y in cells]
        for x, y in placed:
            self.board[y, x] = self.cur

        full = [y for y in range(H) if self.board[y].all()]
        n = len(full)
        self.lines += n
        if self.record:   # 录像帧存消行前的棋盘,前端做消行闪白动画
            self._record_frame(placed=placed, cleared=full, dead=False)
        if n:
            keep = self.board[[y for y in range(H) if y not in full]]
            self.board = np.vstack(
                [np.zeros((n, W), dtype=np.int8), keep])

        new_holes = max(0, self._holes() - holes_before)
        reward = 0.05 + LINE_REWARDS[n] - 0.1 * new_holes
        success = self.lines >= SUCCESS_LINES
        if success:
            reward += 50.0

        self.cur, self.nxt = self.nxt, self._draw()
        truncated = (not success) and self.t >= self.max_steps
        info = {"success": success, "score": self.lines}
        return self._obs(), reward, success, truncated, info

    def _draw(self):
        if not self.bag:
            self.bag = list(self.rng.permutation([1, 2, 3, 4, 5, 6, 7]))
        return int(self.bag.pop())

    def _fits(self, cells, col, row):
        for x, y in cells:
            bx, by = col + x, row + y
            if bx >= W or by >= H or self.board[by, bx]:
                return False
        return True

    def _holes(self):
        """顶部已封死的空格总数。"""
        holes = 0
        for x in range(W):
            seen = False
            for y in range(H):
                if self.board[y, x]:
                    seen = True
                elif seen:
                    holes += 1
        return holes

    def _obs(self):
        heights = np.zeros(W, dtype=np.float32)
        col_holes = np.zeros(W, dtype=np.float32)
        for x in range(W):
            filled = np.nonzero(self.board[:, x])[0]
            if len(filled):
                top = filled[0]
                heights[x] = (H - top) / H
                col_holes[x] = np.sum(self.board[top:, x] == 0) / 10.0
        cur = np.zeros(7, dtype=np.float32)
        cur[self.cur - 1] = 1.0
        nxt = np.zeros(7, dtype=np.float32)
        nxt[self.nxt - 1] = 1.0
        return np.concatenate([heights, col_holes, cur, nxt])

    def _record_frame(self, placed, cleared, dead):
        self.frames.append({
            "b": ["".join(str(v) for v in row) for row in self.board],
            "p": [[int(x), int(y)] for x, y in placed],
            "clr": cleared,
            "lines": self.lines,
            "next": self.nxt,
            **({"dead": True} if dead else {}),
        })

    def render_spec(self):
        return {
            "type": "tetris",
            "w": W, "h": H,
            "goal": SUCCESS_LINES,
            "frame_dt": 0.25,
            # 给前端画"下一个方块"预览用(取第一个朝向)
            "pieces": {pid: shapes[0] for pid, shapes in SHAPES.items()},
        }
