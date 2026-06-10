#!/usr/bin/env bash
# 一键启动:演示服务器 + 训练同时跑。
#
# 用法(方括号里的参数原样透传给 rl_lab.train):
#   ./start.sh                          # 默认 PPO 练二阶摆,练到学会为止
#   ./start.sh --algo dqn               # 换 DQN
#   ./start.sh --algo ppo --forever     # 学会后也不停
#   ./start.sh --resume                 # 断点续练
#   PORT=8888 ./start.sh                # 换演示端口
#   NO_OPEN=1 ./start.sh                # 不自动打开浏览器
#
# Ctrl+C(或 kill 本脚本)同时停掉训练和服务器;
# 训练自然结束后服务器继续供演示,再按 Ctrl+C 才退出。
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
PORT="${PORT:-8000}"
SERVER_PID=""
TRAIN_PID=""

cleanup() {
    [ -n "$TRAIN_PID" ] && kill "$TRAIN_PID" 2>/dev/null || true
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if curl -sf "http://127.0.0.1:${PORT}/api/runs" >/dev/null 2>&1; then
    echo "端口 ${PORT} 已有演示服务器,直接复用: http://localhost:${PORT}"
else
    "$PY" -m rl_lab.server --host 127.0.0.1 --port "$PORT" &
    SERVER_PID=$!
    sleep 0.5
    echo "演示界面: http://localhost:${PORT}"
fi

# 自动打开浏览器(NO_OPEN=1 可跳过)
[ -z "${NO_OPEN:-}" ] && command -v open >/dev/null && open "http://localhost:${PORT}" || true

# 训练也放后台,脚本用可中断的 wait 等它,信号才能即时触发清理
"$PY" -m rl_lab.train "$@" &
TRAIN_PID=$!
wait "$TRAIN_PID" || true
TRAIN_PID=""

if [ -n "$SERVER_PID" ]; then
    echo "训练结束,演示服务器仍在运行,按 Ctrl+C 退出。"
    wait "$SERVER_PID" || true
fi
