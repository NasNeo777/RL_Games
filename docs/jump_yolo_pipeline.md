# 跳一跳真机部署:YOLO 检测 + 线性蓄力流水线

> 这是 **单个环境(`jump`)的真机落地附加件**,不是项目主体。
> 项目主体是 `rl_lab/` 多环境强化学习实验室(见 [根 README](../README.md))。
> 本文档只讲:怎么把训练好的跳一跳 agent 部署到真手机上自动玩。

---

## 1. 这条流水线在做什么

真机上没有现成的"台距/台宽"数值,只有一张屏幕截图。所以需要两段:

```
 手机截图 ──▶ ① YOLO 检测 ──▶ piece(棋子) + landing(台子) 的像素框
                                   │
                                   ▼
                  ② 线性公式  press_ms = coef × gap_px  ──▶ 蓄力时长(ms)
                                   │
                                   ▼
                            ③ adb 长按 ──▶ 跳!
```

- **① 检测**:`adb_jump_ppo.py` 里的 `YoloJumpDetector`,识别棋子和台子。
  多个台子时**选最上面那个**作为目标;棋子位置取**落脚点中心**(底边中心)。
- **② 决策**:微信跳一跳经验公式——蓄力毫秒 = 棋子落脚点到目标台中心的
  像素距离 `gap_px` × 系数 `coef`(默认 **1.35**)。纯线性,**不需要 PPO/检查点**。
- **③ 执行**:`adb shell input swipe`(同点长按)蓄力。

> 早期版本绕道"world 量纲换算 + PPO 预测力度 + scale"和卡尔曼/在线校准,
> 反而不准(PPO 只学会"落在台面",力度在容差内晃)。现已全部移除,回归
> 这条干净的线性公式。

检测器训不好,整条链就废(模型会一直 `detect → None, None`)。所以
**本文档的重点是怎么得到一个靠谱的 YOLO 检测器**。

---

## 2. 需要准备什么

| 前置 | 说明 | 怎么装 |
|---|---|---|
| Python venv | 项目自带 `.venv` | 见根 README |
| ultralytics | YOLO 训练/推理 | `.venv/bin/pip install ultralytics` |
| `yolo11n.pt` | YOLO11-nano 预训练权重(迁移起点) | 已在仓库根目录 |
| adb + 安卓手机 | 真机截图与长按(仅真机运行时需要) | 系统装 `adb`,手机开 USB 调试 |
| labelme(可选) | 仅"真图标注"路线需要 | `.venv/bin/pip install labelme` |

---

## 3. 两条造数据的路线

检测器要训练数据(图 + 标签)。有两条路,**推荐合成路线**:

| | A. 合成数据(推荐) | B. 真图标注 |
|---|---|---|
| 标签来源 | 程序生成,**完美且完整** | 手工/半自动,有噪声 |
| 工作量 | 全自动,零标注 | 需要人工 review |
| 域差距 | 有(合成 vs 真机),靠拟真+微调缩小 | 无 |
| 适用 | 先跑通、快速验证 | 合成不够时补真实样本微调 |

> ⚠️ **历史教训**:本仓库最早的 23 个手工标注(`datasets/jump_labelme/raw/*.json`)
> 与图片**对不上**(框落在空背景上),把第一版 YOLO 喂废了(置信度恒 ~0.01,
> 永远 `None, None`)。它们已被移到 `datasets/jump_labelme/raw_json_backup_corrupt/`。
> **任何真图标注都要先在预览图上核对再用。**

---

## 4. 路线 A:合成数据(推荐)

### 步骤 A1 — 抠棋子贴图

合成场景里贴的是**真实棋子的抠图**,不是手画的。

```bash
# 已有一张可用 sprite:tools/sprites/pawn_00.png
# 想加更多:从真图自动抠候选(find_piece 不可靠,所以输出到 candidates/ 供人工挑)
.venv/bin/python tools/extract_pawn_sprites.py
# 然后人工把好的复制成 tools/sprites/pawn_NN.png
```

| 输入 | 输出 |
|---|---|
| `datasets/jump_labelme/raw/*.png`(真图) | `tools/sprites/candidates/*`(候选,需人工挑) |
| — | `tools/sprites/pawn_*.png`(**已 curate,生成器直接用**) |

### 步骤 A2 — 生成合成数据集

```bash
.venv/bin/python tools/gen_synthetic_jump.py --n 2000 --out datasets/jump_synth
# 想看效果加 --preview 12,会在 datasets/jump_synth/preview/ 画框预览
```

生成内容:渐变背景 + 多形状台子(**方形/圆柱/椭圆/桌子**)+ 蛋糕分层 +
表面装饰(萌脸/电子钟/快递盒标/同心环盖/侧面方块旋钮)+ 形状匹配的干净阴影
+ 真实棋子抠图。**landing 框只框顶面**(不含侧壁/桌腿)。

| 输入 | 输出(YOLO 格式) |
|---|---|
| `tools/sprites/pawn_*.png` | `datasets/jump_synth/images/{train,val}/*.png` |
| | `datasets/jump_synth/labels/{train,val}/*.txt` |
| | `datasets/jump_synth/dataset.yaml` |

→ 跳到 **步骤 5 训练**。

---

## 5. 路线 B:真图标注(补充/微调用)

### 步骤 B1 — 抓真机截图

```bash
.venv/bin/python tools/capture_jump_yolo_dataset.py --serial <adb-serial> --count 120 --preview
```

| 输入 | 输出 |
|---|---|
| 手机屏幕(adb) | `datasets/jump_yolo/images/...` + 规则检测导出的初始标签 |

### 步骤 B2 — 半自动预标注(可选,省事)

```bash
.venv/bin/python tools/auto_label_jump.py --src datasets/jump_labelme/raw
# 评估自动标注质量(对比已有 GT):
.venv/bin/python tools/auto_label_jump.py --eval
```

棋子检测较准,台子尽力而为(**灰色台子常漏**)。输出 labelme JSON + 预览图
到 `datasets/jump_labelme/preview/`,供下一步人工修正。

### 步骤 B3 — labelme 人工核对/标注

```bash
./run_labelme.sh        # 打开 datasets/jump_labelme/raw
```

两类标签:`piece`(棋子,框整体)、`landing`(**只框台子顶面**)。

### 步骤 B4 — labelme JSON → YOLO 数据集

```bash
.venv/bin/python tools/labelme_to_yolo.py --src datasets/jump_labelme/raw --out datasets/jump_yolo_manual
```

| 输入 | 输出 |
|---|---|
| `datasets/jump_labelme/raw/*.json` + `*.png` | `datasets/jump_yolo_manual/{images,labels}/{train,val}/` + `dataset.yaml` |

---

## 6. 训练检测器

```bash
.venv/bin/python tools/train_jump_yolo.py \
  --data datasets/jump_synth/dataset.yaml \
  --device mps --name jump_yolo_synth --project runs/detect/runs
# 默认:--model yolo11n.pt --epochs 60 --imgsz 512 --batch 16
```

| 输入 | 输出 |
|---|---|
| `<dataset>/dataset.yaml` + `yolo11n.pt` | `runs/detect/runs/<name>/weights/best.pt`(历史最优) |
| | `runs/detect/runs/<name>/weights/last.pt` + `results.csv`(逐 epoch 曲线) |

**判断训得好不好**:看 `results.csv`,`cls_loss` 要往下、`mAP50` 要往上。
(坏标签的症状:`cls_loss` 卡在 ~2.8 不动,精度 ~0.01。)

> `--device`:Mac 用 `mps`,无 GPU 用 `cpu`。竖图(1080×2400)在 `imgsz 512`
> 下棋子会缩到很小;若**棋子漏检**,提到 `--imgsz 640` 重训。

---

## 7. 真机运行

```bash
.venv/bin/python adb_jump_ppo.py \
  --serial <adb-serial> \
  --yolo-model runs/detect/runs/jump_yolo_synth/weights/best.pt

# 只测一帧检测、不真按(验证检测器):加 --dry-run
```

每跳把标注调试图写到 `debug_jump_ppo/<序号>.png`(红框=棋子,绿框=目标台,
黄点=台心),便于核对检测对不对。

常用参数:

| 参数 | 默认 | 说明 |
|---|---|---|
| `--detector` | `yolo` | `yolo`/`heuristic`/`auto`(auto 缺模型时回退规则检测) |
| `--yolo-conf` | 0.25 | 检测置信度阈值 |
| `--coef` | 1.35 | 蓄力系数 ms/像素(`press_ms = coef × gap_px`) |
| `--dry-run` | — | 只检测+存调试图,不真按 |
| `--max-jumps` | 0 | 0 = 不限 |

> 注:卡尔曼多帧融合、在线系数校准、PPO/world 换算均已**移除**。现在是
> 单帧检测 + 纯线性公式 `press_ms = coef × gap_px`,跳得太远/太近就调 `--coef`。

---

## 8. 文件地图(放哪、生成啥)

```
adb_jump_ppo.py                 真机主程序:截图→检测→决策→长按
run_labelme.sh                  启动 labelme 标注
tools/
  gen_synthetic_jump.py         ① 生成合成数据集(推荐)
  extract_pawn_sprites.py       └ 抠棋子贴图(候选供人工挑)
  sprites/pawn_*.png            └ 已 curate 的棋子贴图(签入)
  capture_jump_yolo_dataset.py  ② 真机抓图+导出初始标签
  auto_label_jump.py            └ 半自动预标注(piece 准/台子尽力)
  labelme_to_yolo.py            └ labelme JSON → YOLO 数据集
  train_jump_yolo.py            ③ 训练 YOLO 检测器
datasets/        (gitignore)    数据集:jump_synth / jump_yolo* / jump_labelme
runs/            (gitignore)    训练产物:runs/detect/runs/<name>/weights/best.pt
debug_jump_ppo/  (gitignore)    真机运行的逐跳标注调试图
yolo11n.pt       (gitignore)    YOLO 预训练权重
```

> `datasets/`、`runs/`、`debug_jump_ppo/`、`*.pt` 都在 `.gitignore` 里
> (数据和模型不进版本库);`tools/sprites/pawn_*.png` 会签入。

---

## 9. 排错速查

| 现象 | 原因 / 处理 |
|---|---|
| 一直 `detect → None, None` | 检测器没训好(看 `results.csv` 的 cls_loss / mAP50);或标签是坏的 |
| 训练 `cls_loss` 不降、精度 ~0.01 | 标签和图对不上 → 核对 labelme 标注 / 改用合成数据 |
| 棋子漏检 | `imgsz` 提到 640 重训;或补真实样本微调 |
| 灰色台子漏检 | 自动标注的已知短板,labelme 手动补;或靠合成数据覆盖 |
| 跳过头/不够 | 调 `--coef`(大=跳更远) |
| 选错目标台 | 检测多框时选最上面的;确认棋子框/落脚点正确 |
```
