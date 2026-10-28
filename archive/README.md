# archive/ — 历史产物归档区

本目录用于收纳**已被主流程淘汰、但仍有追溯价值**的中间产物（模型权重快照、历史运行日志、
一次性实验的中间结果）。

> **重要**：本目录已写入 `.gitignore`，**不会进入版本库**。它只存在于本地磁盘，
> 目的是让工作区保持干净可读，而不是删除证据。清理前请先确认没有正在跑的实验在读这里的文件。

## 目录内容

```
archive/
├── README.md                     # 本文件：归档清单与恢复方式（入库）
├── MANIFEST_ADAPT.csv            # 25 份权重的文件字节数与 sha256（入库）
├── problem_run_log.txt           # 原根目录 problem.txt：一次远程训练运行的终端记录（不入库）
├── checkpoints/
│   └── ADAPT/                    # ADAPT 基线的 25 份历史权重（85 MB，不入库）
│       ├── adaptive_epoch_{50,100,150,200,250}.pt   # 训练中途快照（6 份）
│       ├── adaptive_curriculum_k{1,2,3,5,10}_*.pt   # 课程学习超参扫描（17 份）
│       └── adaptive_final.pt / adaptive_imitation_k5.pt  # 终版与模仿学习版（2 份）
└── logs/                         # 4 份 2026-07 前后的历史运行日志（不入库）
```

（另有 4 个 0 字节日志——`ablation_basic_topology.log`、`deep_gat_training.log`、`training.log`、
`training_improved.log`——已于 2026-10-28 清理，它们只是训练进程未写出内容就中断的残留。）

## 为什么这些被归档

`network_dismantling/ADAPT/` 是作为**对比基线**接入 `network_dismantling.unified_interface`
的（`unified_interface.py` 第 253 行 `import network_dismantling.ADAPT`）。
但主流程 `main.py` 的对比实验并未调用 ADAPT，ADAPT 自身默认只加载一个权重：

```python
# network_dismantling/ADAPT/__init__.py:16
model_path: str = "network_dismantling/ADAPT/checkpoints/adaptive_best.pt"
```

因此 **26 份权重里只有 `adaptive_best.pt` 是"活"的**，其余 25 份是训练 epoch 快照与
课程学习 `k` 参数扫描留下的中间产物，合计 85 MB，却全部被 git 跟踪，导致仓库
94 MB 中 90 MB 是二进制权重。归档后，仓库体积降到约 5 MB，且 `adaptive_best.pt`
仍保留原位，`adapt_dismantle()` 的默认路径不受影响。

## 怎么恢复

```bash
# 恢复全部（回到归档前的状态）
mv archive/checkpoints/ADAPT/*.pt network_dismantling/ADAPT/checkpoints/

# 只恢复某一份
mv archive/checkpoints/ADAPT/adaptive_curriculum_k5_enhanced_final.pt \
   network_dismantling/ADAPT/checkpoints/

# 校验归档文件是否完整（对比 archive/MANIFEST_ADAPT.csv 里的 sha256）
sha256sum archive/checkpoints/ADAPT/*.pt
```

`archive/README.md` 与 `archive/MANIFEST_ADAPT.csv` 是**唯一会被 git 跟踪**的文件——
归档事实本身需要留痕，85 MB 的二进制不需要。

## 注意

- 这些文件仍是 **git 历史中的旧提交**的一部分。归档只清理了工作区，历史里那 85 MB 依然存在。
  如果目标是彻底给仓库瘦身，需要 `git filter-repo` / BFG 重写历史（会改变所有提交 hash，
  协作分支需要重新 clone），这一步请单独决策，不要顺手做。
- `archive/logs/` 原有 4 个 0 字节文件，已于 2026-10-28 删除；归档不做无谓保留。
