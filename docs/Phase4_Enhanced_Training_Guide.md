# BiT-HyRL 阶段4增强训练指南

## 问题诊断

真实网络上BiT-HyRL效果差的可能原因：
1. **训练数据分布不匹配**：合成网络与真实网络拓扑特性差异大
2. **奖励函数不聚焦**：阶段3-4的多目标奖励可能导致优化方向分散
3. **攻击范围有限**：只考虑0-40%的攻击，未能充分学习全范围鲁棒性
4. **学习率过高**：无法对真实网络进行精细微调

## 解决方案：阶段4增强版 (Phase 4 Enhanced)

### 核心改进

#### 1. 极致聚焦的奖励函数 (Phase 5)

```python
calculate_phase4_enhanced_reward(G, centers):
    奖励 = 70% * 完整R值(0-100%) + 20% * 关键点GCC + 10% * 控制器存活
```

**特点**：
- **完整R值积分**：覆盖0%-100%攻击范围，使用50个采样点
- **关键点强化**：重点优化10%, 20%, 30%攻击点的GCC
- **控制器存活**：避免控制器在高度数节点（前10%-20%）

#### 2. 训练策略优化

- **更低学习率**：3e-5（vs 5e-5），精细微调
- **更多采样**：每epoch 40个图（vs 30个）
- **早停机制**：patience=30，避免过拟合
- **自适应调度**：ReduceLROnPlateau，factor=0.5

#### 3. 数据增强（可选）

混合训练数据：
- 70% 真实网络（data/raw/true）
- 30% 相似规模合成网络（data/raw/train）

## 使用方法

### 方法1：基础增强训练

```bash
python scripts/train_phase4_enhanced.py --epochs 150
```

**说明**：
- 自动从phase 4或phase 3模型继续
- 使用真实网络数据
- 学习率3e-5
- 早停patience=30

### 方法2：使用数据增强

```bash
python scripts/train_phase4_enhanced.py \
    --epochs 200 \
    --use-synthetic \
    --mix-ratio 0.7 \
    --collect-per-epoch 50
```

**说明**：
- 混合70%真实 + 30%合成网络
- 每epoch采样50个图
- 200 epochs（有早停保护）

### 方法3：从特定模型继续

```bash
python scripts/train_phase4_enhanced.py \
    --resume models/curriculum_phase4_dynamic_gat.pth \
    --epochs 100 \
    --lr 2e-5
```

### 方法4：极致微调（推荐）

```bash
python scripts/train_phase4_enhanced.py \
    --epochs 200 \
    --lr 2e-5 \
    --collect-per-epoch 50 \
    --patience 40 \
    --use-synthetic
```

## 评估效果

训练完成后，使用评估脚本对比：

```bash
python scripts/evaluate_phase_models.py \
    --test-dir data/testdata \
    --phase3-model models/curriculum_phase3_dynamic_gat.pth \
    --phase4-model models/curriculum_phase4_dynamic_gat.pth \
    --phase4-enhanced-model models/curriculum_phase4_enhanced_dynamic_gat.pth
```

**输出示例**：
```
Phase 3:
  R-value:      0.6234 ± 0.0892
  GCC @ 10%:    0.8234 ± 0.1234
  GCC @ 20%:    0.6789 ± 0.1456

Phase 4 Enhanced:
  R-value:      0.7456 ± 0.0678
  GCC @ 10%:    0.9123 ± 0.0876
  GCC @ 20%:    0.8345 ± 0.0987

Improvement: R-value: +0.1222 (+19.6%)
```

## 其他改进方法

### 1. 课程学习改进

如果效果仍不理想，考虑重新设计课程：

```bash
# 重新训练phases 1-3，使用真实网络的子集
python scripts/train_curriculum.py --full-curriculum \
    --data-dir data/raw/true \
    --total-epochs 300 \
    --include-phase4
```

### 2. 超参数调优

关键超参数：
- `lr`: 学习率（推荐范围：1e-5 到 5e-5）
- `collect_per_epoch`: 采样量（推荐30-60）
- `k_ratio`: 控制器比例（尝试0.08, 0.10, 0.12）

### 3. 模型架构调整

如果DynamicGAT不够强，尝试：

```bash
python scripts/train_phase4_enhanced.py \
    --model-type graph_transformer \
    --epochs 200
```

### 4. 多次训练集成

训练多个模型，使用投票或平均：

```bash
for seed in 42 123 456; do
    python scripts/train_phase4_enhanced.py \
        --seed $seed \
        --save-path models/phase4_enhanced_seed${seed}.pth
done
```

## 预期效果

**Phase 4 Enhanced 应该带来**：
- R值提升：15-25%
- GCC@10%提升：10-20%
- GCC@20%提升：15-30%
- GCC@30%提升：20-40%

**如果仍然效果不佳**，可能需要：
1. 重新审视网络重构策略（双峰拓扑是否适合真实网络）
2. 考虑使用其他拓扑优化方法（ONION, ROMEN等）
3. 调整控制器选择策略（非RL方法）

## 训练监控

训练过程中关注：
- `reward`：应该稳定上升
- `best`：最佳奖励应该持续刷新
- `lr`：学习率应该逐渐降低
- `no_improve`：不改进轮数，达到patience会触发早停

**正常训练曲线**：
```
Epoch 10:  reward=0.4523, best=0.4621, lr=3.0e-05, no_improve=2
Epoch 20:  reward=0.5234, best=0.5301, lr=3.0e-05, no_improve=1
Epoch 50:  reward=0.6123, best=0.6234, lr=1.5e-05, no_improve=5
Epoch 80:  reward=0.6456, best=0.6512, lr=7.5e-06, no_improve=3
...
Early stopping at epoch 112
```

## 故障排除

**问题1：奖励不上升**
- 降低学习率：`--lr 1e-5`
- 增加采样：`--collect-per-epoch 60`
- 检查数据质量

**问题2：过拟合**
- 减少epochs：`--epochs 100`
- 增加dropout：修改模型配置
- 使用数据增强：`--use-synthetic`

**问题3：训练太慢**
- 减少采样点：修改reward函数中的`num_samples`
- 减少每epoch采样：`--collect-per-epoch 30`
- 使用GPU加速

## 总结

阶段4增强版通过：
1. **极致聚焦的奖励**（完整R值 + 关键点GCC）
2. **精细微调策略**（低学习率 + 早停）
3. **数据增强**（真实 + 合成网络）

应该能显著提升BiT-HyRL在真实网络上的表现。如果效果仍不理想，可能需要重新审视整体架构设计。
