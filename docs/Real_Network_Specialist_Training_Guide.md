# BiT-HyRL 真实网络专家模型训练指南

专门针对真实网络（Colt, Chinanet, UsCarrier 等）设计的多阶段训练流程，目标是在真实数据集上超越其它对比方法。

## 快速开始

### 一键运行完整流程（推荐）

```bash
# Windows
scripts\run_real_network_specialist_pipeline.bat

# Linux/Mac
bash scripts/run_real_network_specialist_pipeline.sh
```

将依次执行：数据增强 → 多阶段训练 → 评估。

### 分步执行

```bash
# 1. 数据增强：根据 data/raw/true 的网络生成合成网络
python scripts/generate_real_network_augmentation.py \
    --output data/raw/true_augmented \
    --per-graph 2 \
    --types BA,ER,Bimodal

# 2. 多阶段训练（多种子）
python scripts/train_real_network_specialist.py \
    --use-augmentation \
    --epochs 400 \
    --seeds 42,123,456,789,2024 \
    --k-ratios 0.08,0.10,0.12

# 3. 评估并生成对比结果
python scripts/evaluate_real_network_specialist.py \
    -m models/real_network_specialist_seed42.pth \
    --run-main \
    --datasets Colt,Chinanet,UsCarrier
```

## 设计说明

### 1. 多阶段课程学习

| 阶段 | 规模 | 奖励 | 目标 |
|------|------|------|------|
| Stage 1 | 10-80 节点 | GCC + R_GCC | 学习基础控制器选择 |
| Stage 2 | 20-150 节点 | GCC + R_GCC | 中等规模泛化 |
| Stage 3 | 10-500 节点 | GCC + R_GCC | 全规模训练 |
| Stage 4 | 全量数据 | Phase4 Enhanced | 真实网络微调 |

### 2. 奖励函数（GCC + R_GCC）

- **GCC**：关键攻击点（10%, 20%, 30%）的连通性保持率
- **R_GCC**：0-50% 攻击范围的曲线下面积（R 值）
- 组合：`R = 0.5 * R_GCC + 0.5 * 关键点GCC`

### 3. 多控制器占比 (k_ratio)

训练时随机采样 `k_ratio ∈ {0.08, 0.10, 0.12}`，增强对不同控制器比例的适应能力。

### 4. 多种子训练

使用 seeds `[42, 123, 456, 789, 2024]` 训练多个模型，取最佳或集成使用。

### 5. 数据增强

根据真实网络的 (节点数, 边数) 生成合成网络：
- **BA**：无标度网络
- **ER**：随机网络
- **Bimodal**：双峰度分布
- 真实:合成 ≈ 7:3

## 输出文件

- **模型**：`models/real_network_specialist_seed{N}.pth`
- **训练曲线**：`results/training_real_specialist/training_curves_seed{N}.png`
- **训练历史**：`results/training_real_specialist/training_history_seed{N}.json`
- **评估结果**：`results/<Colt|Chinanet|UsCarrier>/degree/<id>/`

## 评估与对比

运行 `main.py` 时会与以下方法对比：
- Baseline, GA+RCP, GA+RL
- Onion+RL, ROMEN+RL, UNITY+RL
- BiT-HyRL（使用专家模型）
- FRED-ABL, QDLM

BiT-HyRL 使用 `-m models/real_network_specialist_seed42.pth` 指定的专家模型。

## 自定义参数

```bash
# 仅使用真实网络（不用数据增强）
python scripts/train_real_network_specialist.py --no-augmentation --epochs 300

# 单种子快速测试
python scripts/train_real_network_specialist.py --seed 42 --epochs 200

# 自定义控制器比例
python scripts/train_real_network_specialist.py --k-ratios 0.05,0.10,0.15,0.20

# 更多数据增强
python scripts/generate_real_network_augmentation.py --per-graph 4 --types BA,ER,WS,Bimodal
```

## 故障排除

**问题**：数据增强后图数量很少  
- 检查 `data/raw/true` 是否有 .gml/.graphml 文件  
- 尝试 `--per-graph 3` 增加每种类型的生成数量  

**问题**：训练奖励不上升  
- 降低学习率：`--lr 1e-4`  
- 增加 epochs：`--epochs 600`  

**问题**：评估时 BiT-HyRL 仍落后  
- 使用多种子中表现最好的模型  
- 增加 Stage 4 的 epochs  
- 检查测试集（Colt, Chinanet, UsCarrier）是否在训练数据分布内  
