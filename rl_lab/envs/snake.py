"""贪吃蛇 —— 纯 Python 自制,本项目第一个全程带随机性的环境。

12x12 网格。蛇每步前进一格,吃到食物长一节、得 1 分;食物吃掉后
随机刷新在任意空格上 —— 这就是随机性来源:每局的食物序列都不同,
agent 背不下来动作序列,只能学出"看着食物走、别撞东西"的策略。
撞墙、撞自己、或太久没吃到食物(防无限绕圈)都算死。

动作是相对转向(3 个):直行 / 左转 / 右转,天然不存在"瞬间掉头"
这种无效动作。

观测 10 维(均归一化,前 5 维以蛇头朝向为参考系):
- 3 维:直行 / 左侧 / 右侧三个方向到最近障碍(墙或身体)的空格数
- 2 维:食物相对蛇头的(前方, 左方)偏移
- 4 维:当前朝向 one-hot(决策与绝对方位有关时需要,如贴墙走)
- 1 维:当前长度占格子总数的比例

奖励:吃到食物 +10,死亡 -10,每步按靠近/远离食物 ±0.1 塑形
(吃到食物刷新位置的那步不算)。吃到 SUCCESS_SCORE 个食物视为
成功,+50 并结束回合。
"""
import numpy as np

from .base import BaseEnv

GRID = 12
SUCCESS_SCORE = 30
HUNGER_LIMIT = 120   # 连续这么多步没吃到食物按死亡处理(防绕圈不死)
EAT_REWARD = 10.0
DEATH_PENALTY = -10.0
SHAPING = 0.1
# 方向编号: 0 上, 1 右, 2 下, 3 左;y 轴向下(和画布一致)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class SnakeEnv(BaseEnv):
    obs_dim = 10
    n_actions = 3        # 0 直行, 1 左转, 2 右转
    max_steps = 2000

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.snake = []      # [(x, y), ...],蛇头在最前
        self.dir = 0
        self.food = None
        self.score = 0
        self.t = 0
        self.hunger = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.dir = int(self.rng.integers(4))
        dx, dy = DIRS[self.dir]
        c = GRID // 2
        # 蛇头在中心,身体向反方向延伸 2 节
        self.snake = [(c, c), (c - dx, c - dy), (c - 2 * dx, c - 2 * dy)]
        self.score = 0
        self.t = 0
        self.hunger = 0
        self._place_food()
        self.frames = []
        if self.record:
            self._record_frame(dead=False)
        return self._obs()

    def step(self, action):
        action = int(action)
        if action == 1:
            self.dir = (self.dir - 1) % 4
        elif action == 2:
            self.dir = (self.dir + 1) % 4
        dx, dy = DIRS[self.dir]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy
        self.t += 1
        self.hunger += 1

        # 撞墙/撞身体即死;尾巴这步会让开,所以不含最后一节
        hit_wall = not (0 <= nx < GRID and 0 <= ny < GRID)
        dead = hit_wall or (nx, ny) in self.snake[:-1]

        reward = 0.0
        success = False
        if dead:
            reward = DEATH_PENALTY
        else:
            ate = (nx, ny) == self.food
            prev_dist = abs(hx - self.food[0]) + abs(hy - self.food[1])
            self.snake.insert(0, (nx, ny))
            if ate:
                self.score += 1
                self.hunger = 0
                reward = EAT_REWARD
                self._place_food()
            else:
                self.snake.pop()
                new_dist = abs(nx - self.food[0]) + abs(ny - self.food[1])
                reward = SHAPING * (prev_dist - new_dist)
                if self.hunger >= HUNGER_LIMIT:   # 饿死(防无限绕圈)
                    dead = True
                    reward = DEATH_PENALTY
            # 吃满目标数,或整个棋盘被占满,都算成功
            if self.score >= SUCCESS_SCORE or self.food is None:
                success = True
                reward += 50.0

        terminated = dead or success
        truncated = (not terminated) and self.t >= self.max_steps
        info = {"success": success, "score": self.score}
        if self.record:
            self._record_frame(dead=dead)
        return self._obs(), reward, terminated, truncated, info

    def _place_food(self):
        body = set(self.snake)
        free = [(x, y) for x in range(GRID) for y in range(GRID)
                if (x, y) not in body]
        self.food = free[int(self.rng.integers(len(free)))] if free else None

    def _free_dist(self, d):
        """蛇头沿方向 d 到最近障碍(墙或身体)之间的空格数,归一化。"""
        dx, dy = DIRS[d]
        x, y = self.snake[0]
        body = set(self.snake)
        n = 0
        while True:
            x, y = x + dx, y + dy
            if not (0 <= x < GRID and 0 <= y < GRID) or (x, y) in body:
                return n / GRID
            n += 1

    def _obs(self):
        d = self.dir
        left, right = (d - 1) % 4, (d + 1) % 4
        hx, hy = self.snake[0]
        if self.food is not None:
            fdx, fdy = self.food[0] - hx, self.food[1] - hy
        else:
            fdx = fdy = 0
        # 食物偏移旋转到蛇头参考系:前方分量、左方分量
        food_fwd = (fdx * DIRS[d][0] + fdy * DIRS[d][1]) / GRID
        food_left = (fdx * DIRS[left][0] + fdy * DIRS[left][1]) / GRID
        onehot = [0.0] * 4
        onehot[d] = 1.0
        return np.array([
            self._free_dist(d), self._free_dist(left), self._free_dist(right),
            food_fwd, food_left,
            *onehot,
            len(self.snake) / (GRID * GRID),
        ], dtype=np.float32)

    def _record_frame(self, dead):
        self.frames.append({
            "s": [[x, y] for x, y in self.snake],
            "f": list(self.food) if self.food else None,
            "score": self.score,
            **({"dead": True} if dead else {}),
        })

    def render_spec(self):
        return {
            "type": "snake",
            "grid": GRID,
            "goal": SUCCESS_SCORE,
            "frame_dt": 0.12,
        }
