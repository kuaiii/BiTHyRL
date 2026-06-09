# BiT-HyRL 模块结构说明

## 目录结构

```
src/bit_hyrl/
├── __init__.py          # 公共API导出
├── config.py            # 配置（设备、路径、Node2Vec可用性）
├── topology.py          # 双峰拓扑重构
├── features.py          # 节点特征（Node2Vec + 统计特征）
├── model.py             # MLP策略网络
├── reward.py            # 奖励函数（Step-wise + 端到端）
├── selection.py         # 控制器选择（CI算法、K-Median、混合RL选择）
├── training.py          # 训练（在线微调、离线批量训练）
├── attack_plot.py       # 攻击仿真与绘图
└── reconstruct.py       # 重构流程编排
```

## 模块依赖关系

```
config (无依赖)
  ↓
topology (无依赖)
  ↓
features (依赖: config)
  ↓
model (无依赖)
  ↓
reward (依赖: src.metrics)
  ↓
selection (依赖: model, features)
  ↓
training (依赖: features, model, reward, selection)
  ↓
attack_plot (依赖: src.simulation, src.metrics)
  ↓
reconstruct (依赖: topology, training, reward)
```

## 主要函数

### 核心流程
- `bit_hyrl_reconstruct()`: 完整重构流程（双峰拓扑 + RL选择）

### 训练
- `train_offline_optimized()`: 离线批量训练
- `train_offline_single()`: 单图在线微调
- `train_and_select()`: 预训练 + 在线微调

### 选择
- `hybrid_rl_select()`: 混合RL选择
- `predict()`: RL预测
- `ci_select_subset()`: CI算法选择
- `k_median_vectorized_subset()`: K-Median选择

### 特征
- `get_node_features()`: 提取节点特征（Node2Vec + 统计）
- `get_node2vec_embeddings()`: Node2Vec嵌入

### 奖励
- `calculate_stepwise_reward()`: Step-wise奖励
- `calculate_reward()`: 综合奖励

### 其他
- `create_bimodal_network_exact()`: 双峰拓扑重构
- `simulate_attack_and_plot()`: 攻击仿真与绘图

## 使用方式

### 1. 作为模块导入
```python
from src.bit_hyrl import bit_hyrl_reconstruct, train_offline_optimized

G_bimodal, controllers = bit_hyrl_reconstruct(G, controller_rate=0.1)
```

### 2. 命令行入口
```bash
# 运行重构
python scripts/run_bit_hyrl.py --dataset GtsCe --rate 0.1

# 训练模型
python scripts/train_bit_hyrl.py --use_node2vec --use_stepwise --epochs 100

# 通过main.py运行（BiT-HyRL模式）
python main.py --mode bit_hyrl --dataset GtsCe --rate 0.1 --run_attack

# 向后兼容（旧入口）
python bit_hyrl_reconstruction.py --dataset GtsCe --rate 0.1
```

## 优化功能

1. **Node2Vec特征** (`use_node2vec=True`): 69维特征（64维Node2Vec + 5维统计）
2. **Step-wise Reward** (`use_stepwise=True`): 逐步奖励反馈
3. **CI算法** (`use_ci=True`): 替换K-Median，考虑网络连接性

## 模型保存位置

- 默认: `models/rl_agent.pth` (combined)
- 特定目标: `models/rl_agent_{mode}.pth` (robustness, csa, etc.)

## 结果保存位置

- 默认: `results/bit_hyrl/`
- 攻击仿真: `results/bit_hyrl/attack_{attack_mode}/`
