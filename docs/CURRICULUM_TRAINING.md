# BiT-HyRL 课程学习训练指南

本文档详细说明三阶段课程学习策略的使用方法，包括动态覆盖特征和 Graph Transformer 的使用。

## 目录

1. [改进概述](#改进概述)
2. [三阶段课程学习](#三阶段课程学习)
3. [模型架构选择](#模型架构选择)
4. [训练命令详解](#训练命令详解)
5. [最佳实践](#最佳实践)

---

## 改进概述

### 1. 动态覆盖特征 (Dynamic State Representation)

**问题**：原模型使用 `selected_mask` 和 `selected_embedding` 来告诉模型哪些节点被选了，模型必须通过 GAT 隐式推理哪些节点已被覆盖，增加了学习难度。

**改进**：在输入特征中增加 6 维动态特征：

| 特征 | 维度 | 说明 |
|------|------|------|
| `is_covered` | 1 | 节点是否被覆盖（2跳邻域内有控制器） |
| `distance_to_nearest` | 1 | 到最近控制器的距离（归一化） |
| `coverage_density` | 1 | 局部覆盖密度（邻域内控制器比例） |
| `is_controller` | 1 | 该节点是否已被选为控制器 |
| `distance_to_farthest` | 1 | 到最远控制器的距离 |
| `potential_dispersion_gain` | 1 | 选择该节点的潜在分散度增益 |

**效果**：让模型直接"看到"哪些区域是盲区，显著提升覆盖率指标的收敛速度。

### 2. Graph Transformer 编码器

**问题**：GAT 在层数加深时容易出现"过平滑" (Over-smoothing) 问题，难以捕捉长距离依赖。

**改进**：引入 Graph Transformer，使用全图注意力机制：

- **全图自注意力**：每个节点可以直接关注所有其他节点
- **拉普拉斯位置编码**：使用图拉普拉斯矩阵的特征向量作为位置编码
- **GELU 激活函数**：替代 ELU，训练更稳定

**效果**：对控制器分散度 (dispersion) 指标特别有效，因为能捕捉远距离节点间的关系。

### 3. 课程学习 (Curriculum Learning)

**问题**：直接在复杂大图上学习多目标优化很难，模型容易陷入局部最优。

**改进**：分阶段训练，从简单到复杂：

| 阶段 | 图规模 | 优化目标 | Epochs 占比 |
|------|--------|----------|-------------|
| Phase 1 | 20-50 节点 | GCC（连通性） | 20% |
| Phase 2 | 20-150 节点 | GCC + 覆盖率 | 30% |
| Phase 3 | 20-500 节点 | GCC + 覆盖率 + 分散度 + 抗攻击 | 50% |

---

## 三阶段课程学习

### 阶段 1：连通性优化 (Phase 1)

**目标**：让模型首先学会保持网络的最大连通性

**奖励函数**：
```
R = GCC_ratio (攻击后的GCC / 原始节点数)
```

**攻击设置**：单一攻击比例 15%，简化学习

**数据要求**：小规模图 (20-50 节点)

**训练命令**：
```bash
python scripts/train_curriculum.py \
    --phase 1 \
    --epochs 100 \
    --min-nodes 20 \
    --max-nodes 50 \
    --model-type dynamic_gat \
    --lr 3e-4 \
    --save-path models/curriculum_phase1.pth
```

**期望效果**：
- 模型学会选择能保持网络连通性的节点
- 避免选择容易被攻击的高度数节点
- 训练约 1-2 小时（取决于数据量）

### 阶段 2：覆盖率优化 (Phase 2)

**目标**：在保持连通性的基础上，提高控制器的覆盖范围

**奖励函数**：
```
R = 0.7 * GCC_ratio + 0.3 * coverage_ratio
```

**覆盖率计算**：2跳邻域内被控制器覆盖的节点比例

**数据要求**：中等规模图 (20-150 节点)

**训练命令**：
```bash
python scripts/train_curriculum.py \
    --phase 2 \
    --epochs 150 \
    --min-nodes 20 \
    --max-nodes 150 \
    --model-type dynamic_gat \
    --lr 2e-4 \
    --resume models/curriculum_phase1.pth \
    --save-path models/curriculum_phase2.pth
```

**期望效果**：
- 模型学会分散部署控制器
- 覆盖率指标显著提升
- 训练约 2-3 小时

### 阶段 3：完整多目标优化 (Phase 3)

**目标**：加入分散度和抗攻击能力的优化

**奖励函数**：
```
R = 0.4 * GCC_multi_attack + 0.2 * coverage + 0.2 * dispersion + 0.2 * protection
```

**多轮攻击**：10%, 20%, 30% 三个攻击比例

**分散度**：控制器间平均距离和最小距离

**保护奖励**：控制器不在高度数节点

**数据要求**：全规模图 (20-500 节点)

**训练命令**：
```bash
python scripts/train_curriculum.py \
    --phase 3 \
    --epochs 200 \
    --min-nodes 20 \
    --max-nodes 500 \
    --model-type dynamic_gat \
    --lr 1e-4 \
    --resume models/curriculum_phase2.pth \
    --save-path models/curriculum_phase3.pth
```

**期望效果**：
- 模型在所有指标上达到平衡
- 抗攻击能力显著提升
- 训练约 4-6 小时

---

## 模型架构选择

### 1. DynamicGAT（推荐）

带动态覆盖特征的 GAT 模型，在大多数情况下表现最好。

**特点**：
- 动态特征更新：每一步都重新计算覆盖特征
- GAT 编码器：3层，4头注意力
- 参数量适中：约 50K 参数

**适用场景**：
- 通用场景
- 中等规模网络 (50-200 节点)
- 需要快速训练

**训练命令**：
```bash
python scripts/train_curriculum.py \
    --full-curriculum \
    --total-epochs 450 \
    --model-type dynamic_gat
```

### 2. GraphTransformer

全图注意力机制，适合需要捕捉长距离依赖的场景。

**特点**：
- 全图自注意力：O(n²) 复杂度
- 拉普拉斯位置编码
- 更强的表达能力

**适用场景**：
- 需要优化分散度指标
- 较小规模网络 (< 200 节点，避免内存问题)
- 网络结构复杂

**训练命令**：
```bash
python scripts/train_curriculum.py \
    --full-curriculum \
    --total-epochs 450 \
    --model-type graph_transformer
```

**注意**：GraphTransformer 的全图注意力复杂度为 O(n²)，对于大图可能导致内存问题。建议：
- 对于 n > 200 的图，使用 DynamicGAT
- 或者在训练时限制 `--max-nodes 200`

### 3. UnifiedGAT（原有模型）

原有的统一 GAT 模型，不使用动态特征。

**特点**：
- 规模自适应 Node2Vec
- 规模编码
- 无动态特征更新

**适用场景**：
- 作为基线对比
- 需要与之前的结果比较

**训练命令**：
```bash
python scripts/train_curriculum.py \
    --full-curriculum \
    --total-epochs 450 \
    --model-type unified_gat
```

---

## 训练命令详解

### 一键运行完整课程学习

```bash
# 使用 DynamicGAT（推荐）
python scripts/train_curriculum.py \
    --full-curriculum \
    --total-epochs 450 \
    --model-type dynamic_gat \
    --data-dir data/traindata \
    --save-dir models

# 使用 GraphTransformer
python scripts/train_curriculum.py \
    --full-curriculum \
    --total-epochs 450 \
    --model-type graph_transformer \
    --data-dir data/traindata \
    --save-dir models
```

### 分阶段训练（可中断恢复）

```bash
# 阶段 1
python scripts/train_curriculum.py \
    --phase 1 \
    --epochs 100 \
    --min-nodes 20 \
    --max-nodes 50 \
    --model-type dynamic_gat \
    --lr 3e-4 \
    --collect-per-epoch 30 \
    --save-path models/curriculum_phase1_dynamic_gat.pth

# 阶段 2（从阶段1继续）
python scripts/train_curriculum.py \
    --phase 2 \
    --epochs 150 \
    --min-nodes 20 \
    --max-nodes 150 \
    --model-type dynamic_gat \
    --lr 2e-4 \
    --resume models/curriculum_phase1_dynamic_gat.pth \
    --save-path models/curriculum_phase2_dynamic_gat.pth

# 阶段 3（从阶段2继续）
python scripts/train_curriculum.py \
    --phase 3 \
    --epochs 200 \
    --min-nodes 20 \
    --max-nodes 500 \
    --model-type dynamic_gat \
    --lr 1e-4 \
    --resume models/curriculum_phase2_dynamic_gat.pth \
    --save-path models/curriculum_phase3_dynamic_gat.pth
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--phase` | None | 训练阶段 (1, 2, 3) |
| `--epochs` | 100 | 单阶段 epoch 数 |
| `--full-curriculum` | False | 运行完整课程学习 |
| `--total-epochs` | 450 | 完整课程学习总 epoch 数 |
| `--model-type` | dynamic_gat | 模型架构 |
| `--data-dir` | data/traindata | 训练数据目录 |
| `--min-nodes` | 20 | 最小节点数 |
| `--max-nodes` | 500 | 最大节点数 |
| `--lr` | 3e-4 | 学习率 |
| `--k-ratio` | 0.1 | 控制器比例 |
| `--collect-per-epoch` | 30 | 每 epoch 采样图数 |
| `--resume` | None | 继续训练的模型路径 |
| `--save-path` | auto | 模型保存路径 |
| `--save-dir` | models | 保存目录 |
| `--seed` | 42 | 随机种子 |

---

## 最佳实践

### 数据准备

1. **数据量**：建议至少准备 500+ 张不同规模的图
2. **规模分布**：确保 20-500 节点范围内都有足够的数据
3. **网络类型**：包含 BA、WS、Random、Bimodal 等多种类型

### 训练技巧

1. **学习率衰减**：
   - 阶段 1: 3e-4
   - 阶段 2: 2e-4
   - 阶段 3: 1e-4

2. **监控指标**：
   - 阶段 1：关注 GCC 奖励是否稳定上升
   - 阶段 2：关注覆盖率奖励
   - 阶段 3：关注所有指标的平衡

3. **早停策略**：
   - 如果连续 30 个 epoch 没有改善，可以提前进入下一阶段

### 评估方法

```python
from src.bit_hyrl import (
    DynamicGATPolicy,
    compute_coverage_features,
    compute_dispersion_features,
)
import torch

# 加载模型
checkpoint = torch.load('models/curriculum_phase3_dynamic_gat.pth')
model = DynamicGATPolicy(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 在测试图上评估
# ... (参考 training.py 中的 evaluate_gnn_model)
```

### 常见问题

**Q: GraphTransformer 训练时内存不足？**
A: 限制最大节点数 `--max-nodes 200`，或使用 DynamicGAT。

**Q: 训练过程中奖励下降？**
A: 这在阶段切换时可能发生，因为奖励函数改变了。给模型一些时间适应。

**Q: 如何选择模型架构？**
A: 一般情况下使用 DynamicGAT。如果网络规模较小且需要优化分散度，尝试 GraphTransformer。

---

## 输出文件说明

训练完成后，会在 `--save-dir` 目录下生成：

```
models/
├── curriculum_phase1_dynamic_gat.pth   # 阶段1模型
├── curriculum_phase2_dynamic_gat.pth   # 阶段2模型
├── curriculum_phase3_dynamic_gat.pth   # 阶段3模型
└── curriculum_final_dynamic_gat.pth    # 最终模型（=阶段3）
```

每个 checkpoint 包含：
- `model_state_dict`: 模型权重
- `model_type`: 模型类型
- `model_config`: 模型配置
- `phase`: 训练阶段
- `best_reward`: 最佳奖励
- `history`: 训练历史
