"""演示服务器(纯标准库,无额外依赖)。

用法:
    python -m rl_lab.server [--port 8000]

打开 http://localhost:8000 即可看到最新/最优模型的演示动画和训练曲线。
训练进程只管往 runs/ 写检查点,本服务器按需读取,两者互不干扰。

接口:
    GET /api/runs                       所有训练运行及其状态
    GET /api/metrics?run=<名>&limit=N   训练曲线数据
    GET /api/demo?run=<名>&which=best|latest   跑一局并返回动画帧
"""
import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .base_agent_loader import load_agent_for_demo

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
WEB_DIR = Path(__file__).resolve().parent / "web"


def list_runs():
    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            meta_path = d / "meta.json"
            if not d.is_dir() or not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            meta["run"] = d.name
            meta["has_best"] = (d / "best.pt").exists()
            meta["has_latest"] = (d / "latest.pt").exists()
            runs.append(meta)
    return runs


def read_metrics(run, limit):
    path = RUNS_DIR / run / "metrics.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records = [json.loads(x) for x in lines]
    if limit and len(records) > limit:
        # 均匀抽样,保证曲线覆盖全程且点数有限
        step = len(records) / limit
        records = [records[int(i * step)] for i in range(limit - 1)] \
            + [records[-1]]
    return records


def run_demo(run, which):
    ckpt_path = RUNS_DIR / run / f"{which}.pt"
    if not ckpt_path.exists():
        return {"error": f"找不到检查点 {ckpt_path.name}"}
    env, agent, ckpt = load_agent_for_demo(ckpt_path)
    env.record = True
    obs = env.reset()
    total, steps, done, info = 0.0, 0, False, {}
    while not done:
        obs, r, terminated, truncated, info = env.step(
            agent.act(obs, deterministic=True))
        total += r
        steps += 1
        done = terminated or truncated
    return {
        "spec": env.render_spec(),
        "frames": env.frames,
        "return": round(total, 1),
        "steps": steps,
        "success": bool(info.get("success", False)),
        "which": which,
        "algo": ckpt["algo"],
        "env": ckpt["env"],
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                body = (WEB_DIR / "index.html").read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self._json(200, list_runs())
            elif url.path == "/api/metrics":
                self._json(200, read_metrics(q.get("run", ""),
                                             int(q.get("limit", "300"))))
            elif url.path == "/api/demo":
                self._json(200, run_demo(q.get("run", ""),
                                         q.get("which", "best")))
            else:
                self._json(404, {"error": "not found"})
        except Exception:
            self._json(500, {"error": traceback.format_exc()})

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(),
                   "application/json; charset=utf-8")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # 安静模式


def main():
    p = argparse.ArgumentParser(description="RL 演示服务器")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"演示界面: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
