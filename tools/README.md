# tools/ — 跳一跳真机部署工具集

这些脚本都是**单个环境(`jump`)的真机落地附加件**,服务于把训练好的跳一跳
agent 部署到真手机。完整步骤见 [docs/jump_yolo_pipeline.md](../docs/jump_yolo_pipeline.md)。
项目主体(多环境 RL)在 `rl_lab/`,与这里无关。

| 脚本 | 作用 | 输入 → 输出 |
|---|---|---|
| `gen_synthetic_jump.py` | **①推荐** 生成合成数据集(多形状台子+装饰+真实棋子抠图,标签完美) | `tools/sprites/pawn_*.png` → `datasets/jump_synth/{images,labels}` + `dataset.yaml` |
| `extract_pawn_sprites.py` | 从真图抠棋子贴图候选(供人工挑) | 真图 → `tools/sprites/candidates/*` |
| `capture_jump_yolo_dataset.py` | **②** 真机抓截图 + 导出初始标签 | adb 屏幕 → `datasets/jump_yolo/images` |
| `auto_label_jump.py` | 半自动预标注(piece 准,台子尽力);`--eval` 评估质量 | 真图 → labelme JSON + 预览图 |
| `labelme_to_yolo.py` | labelme 标注 → YOLO 数据集 | `*.json` → `datasets/jump_yolo_manual/...` + `dataset.yaml` |
| `train_jump_yolo.py` | **③** 训练 YOLO 检测器 | `dataset.yaml` + `yolo11n.pt` → `runs/detect/runs/<name>/weights/best.pt` |

快速跑通(合成路线):

```bash
.venv/bin/python tools/gen_synthetic_jump.py --n 2000 --out datasets/jump_synth
.venv/bin/python tools/train_jump_yolo.py --data datasets/jump_synth/dataset.yaml --device mps --name jump_yolo_synth --project runs/detect/runs
.venv/bin/python adb_jump_ppo.py --yolo-model runs/detect/runs/jump_yolo_synth/weights/best.pt --dry-run
```
