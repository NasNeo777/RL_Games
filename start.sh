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
#   RL_CONDA_ENV=强化学习 ./start.sh     # 指定 conda 环境(默认:强化学习)
#
# Ctrl+C(或 kill 本脚本)同时停掉训练和服务器;
# 训练自然结束后服务器继续供演示,再按 Ctrl+C 才退出。
set -euo pipefail
cd "$(dirname "$0")"
RL_CONDA_ENV="${RL_CONDA_ENV:-强化学习}"
PY=python
PORT="${PORT:-8000}"
SERVER_PID=""
TRAIN_PID=""

activate_conda_env() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "$RL_CONDA_ENV" ]; then
        PY="${CONDA_PREFIX:-}/bin/python"
        if [ ! -x "$PY" ]; then
            PY=python
        fi
        echo "使用 conda 环境: ${CONDA_DEFAULT_ENV} (${CONDA_PREFIX:-已激活})"
        return
    fi

    local conda_exe=""
    if command -v conda >/dev/null 2>&1; then
        conda_exe="$(command -v conda)"
    elif [ -x /opt/miniconda3/bin/conda ]; then
        conda_exe=/opt/miniconda3/bin/conda
    elif [ -x "$HOME/miniconda3/bin/conda" ]; then
        conda_exe="$HOME/miniconda3/bin/conda"
    elif [ -x "$HOME/anaconda3/bin/conda" ]; then
        conda_exe="$HOME/anaconda3/bin/conda"
    fi

    if [ -z "$conda_exe" ]; then
        echo "找不到 conda，请先安装/初始化 conda，或把 conda 加到 PATH。" >&2
        exit 1
    fi

    local conda_base
    conda_base="$("$conda_exe" info --base)"
    # shellcheck disable=SC1091
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$RL_CONDA_ENV"
    PY="$CONDA_PREFIX/bin/python"
    if [ ! -x "$PY" ]; then
        echo "conda 环境 ${RL_CONDA_ENV} 中找不到可执行 Python: $PY" >&2
        exit 1
    fi
    echo "使用 conda 环境: ${CONDA_DEFAULT_ENV} (${CONDA_PREFIX})"
}

check_python_deps() {
    if "$PY" - <<'PY' >/dev/null 2>&1
import importlib.util

missing = [name for name in ("numpy", "torch")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(",".join(missing))
PY
    then
        return
    fi

    echo "conda 环境 ${RL_CONDA_ENV} 缺少核心依赖，请先运行：" >&2
    echo "  $PY -m pip install -r requirements.txt" >&2
    exit 1
}

cleanup() {
    [ -n "$TRAIN_PID" ] && kill "$TRAIN_PID" 2>/dev/null || true
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

activate_conda_env
check_python_deps

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
