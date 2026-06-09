# BiT-HyRL 训练指南

本文档说明如何运行 BiT-HyRL 的两种训练模式：**大规模重新训练 (Scaled)** 和 **优化研究训练 (Optimized)**。

---

## 快速开始

### 1. 生成扩展数据集

```bash
python dataset/generate_datasets.py --split all --category syn --count 300 \
    --sizes 20 50 100 150 200 300 500 800 --seed 42
```

这会生成约 **19,200** 个合成网络（8 种拓扑 × 8 个尺寸 × 300 个实例），保存在 `dataset/all/syn/`。

支持的拓扑类型：
- `BA` / `BAdense` — Barabási-Albert (m=4 / m=6)
- `ER` — Erdős-Rényi
- `WS` / `WSlow` / `WShigh` — Watts-Strogatz (p=0.3 / 0.1 / 0.5)
- `HK` — Holme-Kim Powerlaw Cluster
- `RG` — Random Geometric

### 2. 运行训练

#### A) 大规模重新训练 (Scaled-Up)

增大模型规模 + 完整数据集 + 完整攻击池的基线重新训练。

```bash
python scripts/train_bit_hyrl_scaled.py \
    --epochs 200 \
    --embed-dim 256 \
    --hidden-channels 256 \
    --heads 12 \
    --num-layers 6 \
    --collect-per-epoch 40 \
    --sample-methods 5 \
    --aggregation mean \
    --attack-ratio 0.15
```

**模型配置**：
| 参数 | 值 |
|------|-----|
| 输入维度 (Node2Vec) | 256 |
| 隐藏层维度 | 256 |
| 注意力头数 | 12 |
| GAT 层数 | 6 |
| 聚合方式 | mean |
| 每次采样攻击方法 | 5 |

模型保存至：`src/train/v1/checkpoints/gnn_ppo_agent_scaled.pth`

#### B) 优化研究训练 (Optimized)

在同样模型规模下，引入课程式攻击采样和自适应聚合策略。

```bash
python scripts/train_bit_hyrl_optimized.py \
    --epochs 200 \
    --embed-dim 256 \
    --hidden-channels 256 \
    --heads 12 \
    --num-layers 6 \
    --collect-per-epoch 60 \
    --sample-methods 0 \
    --aggregation min
```

**核心改进**：
- **课程式攻击采样**：前 25% epochs 只用简单攻击（degree/random/pagerank），中期加入 medium 攻击，后期加入 hard 攻击（GND/GDM/EGND）
- **自适应聚合**：早期用 `mean` 鼓励泛化，后期逐步过渡到 `min` 强化最坏情况鲁棒性
- **完整攻击池**：每次使用全部可用攻击方法（不采样）

模型保存至：`src/train/v1/checkpoints/gnn_ppo_agent_optimized.pth`

### 3. 断点续训

两个脚本均支持 `--resume`：

```bash
python scripts/train_bit_hyrl_scaled.py \
    --resume src/train/v1/checkpoints/gnn_ppo_agent_scaled.pth \
    --epochs 100
```

### 4. 评估与对比

#### 单模型评估

```bash
python scripts/evaluate_bit_hyrl_adversarial.py \
    --model src/train/v1/checkpoints/gnn_ppo_agent_scaled.pth \
    --output results/eval_scaled.csv
```

#### 多模型横向对比

```bash
python scripts/evaluate_bit_hyrl_adversarial.py \
    --models \
        src/train/v1/checkpoints/deep_gat_gcc_L5_H8_D128.pth \
        src/train/v1/checkpoints/gnn_ppo_agent_adversarial.pth \
        src/train/v1/checkpoints/gnn_ppo_agent_scaled.pth \
        src/train/v1/checkpoints/gnn_ppo_agent_optimized.pth \
    --model-names Baseline Adversarial Scaled Optimized \
    --output results/eval_comparison.csv
```

这会输出跨攻击方法的对比表格和各指标最佳模型。

---

## 训练参数速查

### 通用参数

| 参数 | 说明 | Scaled 默认 | Optimized 默认 |
|------|------|------------|----------------|
| `--epochs` | 训练轮数 | 200 | 200 |
| `--lr` | 学习率 | 3e-4 | 3e-4 |
| `--k-ratio` | 控制器比例 | 0.1 | 0.1 |
| `--collect-per-epoch` | 每轮采样图数 | 40 | 60 |
| `--attack-ratio` | 奖励函数攻击比例 | 0.15 | 0.15 |
| `--max-graphs` | 最大使用图数 (0=全部) | None | None |
| `--max-nodes` | 最大节点数过滤 | 1000 | 1000 |

### 模型参数

| 参数 | 说明 | Scaled | Optimized |
|------|------|--------|-----------|
| `--embed-dim` | Node2Vec 维度 | 256 | 256 |
| `--hidden-channels` | GAT 隐藏层 | 256 | 256 |
| `--heads` | 注意力头数 | 12 | 12 |
| `--num-layers` | GAT 层数 | 6 | 6 |

### 对抗训练特有参数

| 参数 | 说明 | Scaled | Optimized |
|------|------|--------|-----------|
| `--aggregation` | 奖励聚合方式 | mean | min |
| `--sample-methods` | 每次采样攻击数 | 5 | 0 (全部) |
| `--no-curriculum` | 禁用课程采样 | N/A | 可指定 |
| `--no-adaptive-aggregation` | 禁用自适应聚合 | N/A | 可指定 |

---

## 攻击方法池

训练时自动从以下方法中筛选可用的：

```python
['degree', 'betweenness', 'pagerank', 'eigenvector', 'random',
 'CI_L1', 'CI_L2', 'CoreHD', 'GND', 'EGND', 'EI_s1', 'GDM']
```

部分方法（如 `GDM`, `FINDER`）依赖 TensorFlow，若未安装则自动跳过。

---

## 预期训练时间

| 硬件 | Scaled (200 epochs) | Optimized (200 epochs) |
|------|---------------------|------------------------|
| RTX 3060 12GB | ~8-12 小时 | ~12-18 小时 |
| RTX 4090 / A100 | ~2-4 小时 | ~3-6 小时 |

> Optimized 训练更慢是因为 `sample_methods=0`（使用全部攻击方法）且 `collect_per_epoch=60`。

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `scripts/train_bit_hyrl_scaled.py` | 大规模重新训练入口 |
| `scripts/train_bit_hyrl_optimized.py` | 优化研究训练入口 |
| `scripts/evaluate_bit_hyrl_adversarial.py` | 多模型对比评估 |
| `dataset/generate_datasets.py` | 扩展数据集生成 |
| `src/bit_hyrl/reward.py` | 对抗奖励函数（含课程/自适应聚合） |
| `src/bit_hyrl/ppo_trainer.py` | PPO 训练器（含 epoch 跟踪） |
