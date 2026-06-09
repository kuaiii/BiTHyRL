# 增量训练说明（GNN + PPO）

本文说明如何在 **data/raw/train** 中的 20–100 节点 BA/WS/RA 数据集上进行**增量训练**（在已有 checkpoint 上继续训练）。

---

## 1. 数据准备：在 raw-train 中生成 20–100 节点 BA / WS / RA

在 `data/raw/train` 下按类型各生成 200 个图（共 600 个），节点数 20–100：

```bash
python scripts/generate_training_data.py ^
  --min_nodes 20 ^
  --max_nodes 100 ^
  --count_per_type 200 ^
  --types ba ws er ^
  --output_dir data/raw/train
```

- **BA**：无标度（Barabási–Albert）
- **WS**：小世界（Watts–Strogatz）
- **ER**：随机图（Erdős–Rényi，即 RA）

生成后目录中会有约 600 个 `.gml` 文件（如 `BA_gen_50n_0.gml`、`WS_gen_80n_1.gml`、`ER_gen_30n_2.gml` 等）。

---

## 2. 首次训练（从头训练）

仅用上述 20–100 节点数据训练一个新模型（不加载旧 checkpoint）：

```bash
python scripts/train_deep_gat.py ^
  --train_dir data/raw/train ^
  --min_nodes 20 ^
  --max_nodes 100 ^
  --epochs 200 ^
  --mode gcc ^
  --output models/deep_gat_gcc_L5_H8_D128.pth
```

- `--min_nodes 20 --max_nodes 100`：只使用 20–100 节点的图，与生成数据一致。
- 不指定 `--resume_from` 即为**从头训练**。

---

## 3. 增量训练（在已有模型上继续训练）

在**已有 GNN checkpoint** 上，用同一批或新数据继续训练：

```bash
python scripts/train_deep_gat.py ^
  --train_dir data/raw/train ^
  --min_nodes 20 ^
  --max_nodes 100 ^
  --epochs 100 ^
  --mode gcc ^
  --resume_from models/deep_gat_gcc_L5_H8_D128.pth ^
  --output models/deep_gat_gcc_L5_H8_D128_continued.pth
```

要点：

- **`--resume_from`**：要加载的 checkpoint 路径（如之前保存的 `deep_gat_gcc_L5_H8_D128.pth`）。
- **架构一致**：checkpoint 的 GAT 层数、heads、hidden 等需与当前脚本默认一致（如 L5_H8_D128），否则会提示架构不一致并**重新初始化**（相当于从头训练）。
- **`--output`**：可另存为新文件（如 `_continued.pth`），避免覆盖原模型；不指定则按默认命名覆盖。
- **`--epochs`**：增量训练再跑的轮数（例如再训 100 轮）。

增量训练会：

- 加载 `resume_from` 的模型权重与历史奖励；
- 在此基础上继续训练 `epochs` 轮；
- 将最佳模型保存到 `--output`。

---

## 4. 建议流程小结

| 步骤 | 命令/操作 |
|------|------------|
| 1. 生成数据 | `generate_training_data.py --min_nodes 20 --max_nodes 100 --count_per_type 200 --types ba ws er --output_dir data/raw/train` |
| 2. 首次训练 | `train_deep_gat.py --train_dir data/raw/train --min_nodes 20 --max_nodes 100 --epochs 200 --mode gcc` |
| 3. 增量训练 | `train_deep_gat.py --train_dir data/raw/train --min_nodes 20 --max_nodes 100 --epochs 100 --resume_from models/deep_gat_gcc_L5_H8_D128.pth --output models/deep_gat_gcc_L5_H8_D128_continued.pth` |

在 Linux / macOS 下将 `^` 换成 `\` 即可。

---

## 5. 可选参数

- **`--max_graphs N`**：每轮最多使用 N 个图（用于快速调试）。
- **`--lr`**：学习率（默认 3e-4）；增量训练时可略调小（如 1e-4）以微调。
- **`--collect_per_epoch`**：每轮采样的轨迹数（默认 30）。
- **`--evaluate`**：训练结束后在部分图上评估一次。

示例（增量训练 + 较小学习率 + 训练后评估）：

```bash
python scripts/train_deep_gat.py ^
  --train_dir data/raw/train ^
  --min_nodes 20 --max_nodes 100 ^
  --epochs 100 ^
  --resume_from models/deep_gat_gcc_L5_H8_D128.pth ^
  --lr 1e-4 ^
  --evaluate
```
