# BiT-HyRL GNN 强化学习架构详解

本文档详细描述 BiT-HyRL 中 GNN + PPO 强化学习框架的完整流程，包括各组件的输入输出规格。

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BiT-HyRL GNN+PPO 架构                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐       │
│   │   输入图 G   │───▶│  特征提取器   │───▶│   GAT 策略网络      │       │
│   │ (NetworkX)  │    │              │    │  (Actor-Critic)    │       │
│   └─────────────┘    └──────────────┘    └─────────────────────┘       │
│                              │                    │                     │
│                              ▼                    ▼                     │
│                      ┌──────────────┐    ┌─────────────────────┐       │
│                      │ 节点特征 X   │    │  动作概率 + 状态价值  │       │
│                      │ [N, 5]      │    │                     │       │
│                      └──────────────┘    └─────────────────────┘       │
│                                                   │                     │
│                                                   ▼                     │
│                                          ┌─────────────────────┐       │
│                                          │    PPO 优化器       │       │
│                                          │  (策略梯度更新)      │       │
│                                          └─────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 特征提取模块

### 2.1 节点特征提取 (`get_simple_node_features`)

**功能**: 从 NetworkX 图中提取每个节点的结构特征

**输入**:
| 参数 | 类型 | 描述 |
|------|------|------|
| `G` | `networkx.Graph` | 输入图 |
| `device` | `torch.device` | 计算设备 (CPU/CUDA) |

**输出**:
| 返回值 | 形状 | 描述 |
|--------|------|------|
| `features` | `[N, 5]` | 节点特征矩阵 |
| `node_list` | `List[node]` | 节点 ID 列表 (保持顺序) |

**特征维度详解** (共 5 维):

| 维度 | 特征名称 | 计算方式 | 取值范围 |
|------|----------|----------|----------|
| 0 | 归一化度数 | `degree(v) / max_degree` | [0, 1] |
| 1 | 聚类系数 | `nx.clustering(G)[v]` | [0, 1] |
| 2 | PageRank | `nx.pagerank(G)[v]` | [0, 1] |
| 3 | 特征向量中心性 | `nx.eigenvector_centrality(G)[v]` | [0, 1] |
| 4 | 度数对数 | `log(1 + degree(v)) / max_log_deg` | [0, 1] |

**代码示例**:
```python
# 输入
G = nx.karate_club_graph()  # 34 nodes, 78 edges

# 处理
features, node_list = get_simple_node_features(G, device='cuda')

# 输出
# features.shape = torch.Size([34, 5])
# node_list = [0, 1, 2, ..., 33]
```

---

### 2.2 图结构转换 (`graph_to_pyg_data`)

**功能**: 将 NetworkX 图转换为 PyTorch Geometric 格式

**输入**:
| 参数 | 类型 | 描述 |
|------|------|------|
| `G` | `networkx.Graph` | 输入图 |
| `node_features` | `Tensor` (可选) | 预计算的节点特征 |
| `device` | `torch.device` | 计算设备 |

**输出**:
| 返回值 | 形状 | 描述 |
|--------|------|------|
| `x` | `[N, F]` | 节点特征矩阵 |
| `edge_index` | `[2, 2E]` | 边索引 (无向图双向) |
| `node_list` | `List[node]` | 节点 ID 列表 |

**边索引格式**:
```
原始边: (0,1), (0,2), (1,2)
         ↓
edge_index = [[0, 1, 0, 2, 1, 2],   # 源节点
              [1, 0, 2, 0, 2, 1]]   # 目标节点
```

---

## 3. GAT 策略网络 (`GATPolicy`)

### 3.1 网络架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GATPolicy 网络结构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入层                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  x: [N, 5]          节点特征                                     │   │
│  │  edge_index: [2, 2E] 边索引                                      │   │
│  │  selected_mask: [N]  已选节点掩码 (bool)                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  GAT Layer 1                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GATConv(in=5, out=64, heads=4, concat=True)                    │   │
│  │  输出: [N, 64×4] = [N, 256]                                      │   │
│  │  + LayerNorm + ELU + Dropout(0.1)                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  GAT Layer 2                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GATConv(in=256, out=64, heads=1, concat=False)                 │   │
│  │  输出: [N, 64]                                                   │   │
│  │  + LayerNorm + ELU                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  位置编码 (已选节点)                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  h = h + selected_mask × selected_embedding                     │   │
│  │  selected_embedding: [1, 64] (可学习参数)                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                    ┌───────────────┴───────────────┐                   │
│                    ▼                               ▼                    │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐    │
│  │      全局注意力池化          │   │        Actor Head           │    │
│  │  attention_weights: [N, 1]  │   │  MLP(128 → 64 → 1)         │    │
│  │  global_context: [1, 64]    │   │  输入: [h, global] = [N,128]│    │
│  └─────────────────────────────┘   │  输出: scores [N]           │    │
│                    │               └─────────────────────────────┘    │
│                    │                               │                    │
│                    ▼                               ▼                    │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐    │
│  │       Critic Head           │   │    温度缩放 + Masking        │    │
│  │  MLP(64 → 64 → 1)          │   │  scores = scores / temp      │    │
│  │  输入: mean_pool(h)         │   │  scores[selected] = -∞      │    │
│  │  输出: value [1]            │   │  probs = softmax(scores)    │    │
│  └─────────────────────────────┘   └─────────────────────────────┘    │
│                    │                               │                    │
│                    ▼                               ▼                    │
│  输出层                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  probs: [N]     节点选择概率分布                                  │   │
│  │  value: [1]     状态价值估计 V(s)                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 GAT 注意力机制详解

**Graph Attention Network (GAT)** 的核心是多头注意力机制：

```
对于节点 i，聚合邻居 j ∈ N(i) 的信息:

1. 线性变换:
   h'_i = W · h_i
   h'_j = W · h_j

2. 注意力系数计算:
   e_ij = LeakyReLU(a^T · [h'_i || h'_j])
   
3. 归一化 (Softmax over neighbors):
   α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k∈N(i) exp(e_ik)

4. 加权聚合:
   h''_i = σ(Σ_j∈N(i) α_ij · h'_j)

5. 多头注意力:
   h_final = Concat(head_1, head_2, ..., head_K)
```

**本项目配置**:
| 参数 | Layer 1 | Layer 2 |
|------|---------|---------|
| 输入维度 | 5 | 256 |
| 输出维度 | 64 | 64 |
| 注意力头数 | 4 | 1 |
| Concat | True | False |
| Dropout | 0.1 | 0.1 |

### 3.3 Actor-Critic 输出

**Actor (策略网络)**:
```python
# 输入: 节点嵌入 h [N, 64] + 全局上下文 [N, 64]
actor_input = torch.cat([h, global_context_expanded], dim=-1)  # [N, 128]

# MLP 处理
scores = Linear(128 → 64) → ReLU → Dropout → Linear(64 → 1)  # [N, 1]
scores = scores.squeeze(-1)  # [N]

# 温度缩放 (可学习参数)
scores = scores / temperature  # temperature ∈ [0.1, 2.0]

# 掩码已选节点
scores[selected_mask] = -1e9

# Softmax 得到概率
probs = softmax(scores)  # [N], 和为 1
```

**Critic (价值网络)**:
```python
# 输入: 节点嵌入的均值池化
pooled = h.mean(dim=0)  # [64]

# MLP 处理
value = Linear(64 → 64) → ReLU → Dropout → Linear(64 → 1)  # [1]
```

---

## 4. PPO 训练流程

### 4.1 整体训练循环

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PPO 训练主循环                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  for epoch in range(total_epochs):                                      │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 数据收集阶段 (Rollout)                                        │   │
│  │    for graph in sampled_graphs:                                  │   │
│  │        trajectory = collect_trajectory(graph, k, reward_fn)     │   │
│  │        buffer.add(trajectory)                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 2. 优势估计 (GAE)                                                │   │
│  │    advantages, returns = compute_gae(rewards, values, dones)    │   │
│  │    advantages = normalize(advantages)                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 3. PPO 更新 (多轮 Mini-batch)                                    │   │
│  │    for ppo_epoch in range(n_epochs):                            │   │
│  │        for batch in mini_batches:                               │   │
│  │            loss = policy_loss + value_loss - entropy_bonus      │   │
│  │            optimizer.step()                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 4. 清空缓冲区，记录日志                                          │   │
│  │    buffer.clear()                                               │   │
│  │    log(avg_reward, policy_loss, ...)                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 轨迹收集 (`collect_trajectory`)

**单图控制器选择过程**:

```
输入: 图 G, 控制器数量 k, 奖励函数 reward_fn

初始化:
  selected_mask = [False] × N
  centers = []
  experiences = []

for step in range(k):
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. 获取动作                                                         │
│    action, log_prob, value, probs = model.get_action(              │
│        x, edge_index, selected_mask, deterministic=False           │
│    )                                                                │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. 执行动作                                                         │
│    center = node_list[action]                                      │
│    centers.append(center)                                          │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. 计算即时奖励                                                     │
│    if step < k - 1:                                                │
│        reward = 0.01  # 存活奖励 (改进版使用增量奖励)                │
│    else:                                                           │
│        reward = reward_fn(G, centers)  # 最终奖励                  │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. 保存经验                                                         │
│    exp = Experience(x, edge_index, action, log_prob, value,        │
│                     reward, selected_mask, done)                   │
│    experiences.append(exp)                                         │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. 更新状态                                                         │
│    selected_mask[action] = True                                    │
└─────────────────────────────────────────────────────────────────────┘

输出: centers, final_reward, experiences → buffer
```

**经验数据结构**:
```python
Experience = namedtuple('Experience', [
    'x',              # 节点特征 [N, 5]
    'edge_index',     # 边索引 [2, 2E]
    'action',         # 选择的节点索引 (int)
    'log_prob',       # 动作对数概率 log π(a|s)
    'value',          # 状态价值估计 V(s)
    'reward',         # 即时奖励 r
    'selected_mask',  # 当前已选掩码 [N] (bool)
    'done',           # 是否为最后一步
])
```

### 4.3 GAE 优势估计

**Generalized Advantage Estimation (GAE)** 公式:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)           # TD 误差

A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}      # GAE 优势

简化递推形式:
A_t = δ_t + γλ · A_{t+1}    (从后向前计算)

回报值:
G_t = A_t + V(s_t)
```

**参数设置**:
| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 折扣因子 | γ | 0.99 | 未来奖励衰减 |
| GAE λ | λ | 0.95 | 偏差-方差平衡 |

**代码实现**:
```python
def compute_gae(rewards, values, dones):
    advantages = []
    returns = []
    gae = 0
    
    # 从后向前计算
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1] if not dones[t] else 0
        
        # TD 误差
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE 递推
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    
    return advantages, returns
```

### 4.4 PPO 损失函数

**PPO Clipped Objective**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PPO 损失函数                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 策略比率                                                             │
│     r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)                           │
│            = exp(log_prob_new - log_prob_old)                          │
│                                                                         │
│  2. Clipped Surrogate Loss                                              │
│     L^CLIP = min(                                                       │
│         r_t(θ) · A_t,                          # 原始目标              │
│         clip(r_t(θ), 1-ε, 1+ε) · A_t           # 裁剪目标              │
│     )                                                                   │
│     L_policy = -E[L^CLIP]                      # 取负号用于梯度下降     │
│                                                                         │
│  3. Value Loss (MSE)                                                    │
│     L_value = MSE(V(s_t), G_t)                                         │
│             = (V(s_t) - returns_t)²                                    │
│                                                                         │
│  4. Entropy Bonus (鼓励探索)                                            │
│     H(π) = -Σ π(a|s) · log π(a|s)                                      │
│     L_entropy = -entropy_coef · H(π)                                   │
│                                                                         │
│  5. 总损失                                                               │
│     L_total = L_policy + value_coef · L_value + L_entropy              │  
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**超参数设置**:
| 参数 | 符号 | 默认值 | 改进值 | 说明 |
|------|------|--------|--------|------|
| Clip ε | ε | 0.2 | 0.2 | 策略更新幅度限制 |
| Value 系数 | c_v | 0.5 | 0.5 | 价值损失权重 |
| Entropy 系数 | c_e | 0.01 | 0.02 | 熵奖励权重 |
| 梯度裁剪 | - | 0.5 | 0.5 | 最大梯度范数 |
| PPO Epochs | - | 4 | 10 | 每次更新的迭代数 |
| Batch Size | - | 64 | 32 | Mini-batch 大小 |

---

## 5. 奖励函数设计

### 5.1 GCC 专注奖励 (`calculate_gcc_focused_reward`)

```
R = w_gcc × GCC_retention + w_disp × dispersion_score

其中:
- GCC_retention: 攻击后最大受控连通分量 / 原始节点数
- dispersion_score: 控制器平均距离 / 理想距离

默认权重: w_gcc = 0.7, w_disp = 0.3
```

### 5.2 自适应奖励 (`calculate_adaptive_reward`)

```
根据网络规模自动调整参数:

小规模网络 (N < 50):
  - 攻击比例: [15%, 25%, 35%]
  - GCC 权重: 0.40
  - 分散性权重: 0.25
  - 保护权重: 0.20
  - 覆盖权重: 0.15

大规模网络 (N ≥ 200):
  - 攻击比例: [5%, 10%, 15%, 20%]
  - GCC 权重: 0.50
  - 分散性权重: 0.20
  - 保护权重: 0.15
  - 覆盖权重: 0.15
```

---

## 6. 完整数据流示例

以一个 34 节点的空手道俱乐部图为例:

```
输入:
  G = nx.karate_club_graph()
  N = 34 (节点数)
  E = 78 (边数)
  k = 3 (选择3个控制器)

Step 1: 特征提取
  x = get_simple_node_features(G)
  x.shape = [34, 5]
  
  edge_index = graph_to_pyg_data(G)
  edge_index.shape = [2, 156]  # 78×2 (双向边)

Step 2: 第一次选择 (step=0)
  selected_mask = [False] × 34
  
  # GAT 前向传播
  h1 = GAT_Layer1(x, edge_index)     # [34, 256]
  h2 = GAT_Layer2(h1, edge_index)    # [34, 64]
  
  # Actor 输出
  probs = Actor(h2, global_context)   # [34]
  probs = softmax(scores)             # 概率分布
  
  # 采样动作
  action = sample(Categorical(probs)) # e.g., 33
  log_prob = log(probs[33])           # e.g., -2.5
  
  # Critic 输出
  value = Critic(mean(h2))            # e.g., 0.3
  
  # 更新状态
  centers = [33]
  selected_mask[33] = True
  reward = 0.01  # 中间奖励

Step 3: 第二次选择 (step=1)
  # GAT 前向 (带位置编码)
  h2 = h2 + selected_mask × selected_embedding
  
  # Actor (已选节点被 mask)
  scores[33] = -∞
  probs = softmax(scores)             # 节点33概率为0
  
  action = sample(...)                # e.g., 0
  centers = [33, 0]
  reward = 0.01

Step 4: 第三次选择 (step=2, 最后一步)
  action = sample(...)                # e.g., 2
  centers = [33, 0, 2]
  
  # 计算最终奖励
  reward = reward_fn(G, [33, 0, 2])   # e.g., 0.65

Step 5: GAE 计算
  rewards = [0.01, 0.01, 0.65]
  values = [0.3, 0.35, 0.4]
  
  # 从后向前
  δ_2 = 0.65 + 0 - 0.4 = 0.25
  A_2 = 0.25
  
  δ_1 = 0.01 + 0.99×0.4 - 0.35 = 0.056
  A_1 = 0.056 + 0.99×0.95×0.25 = 0.291
  
  δ_0 = 0.01 + 0.99×0.35 - 0.3 = 0.057
  A_0 = 0.057 + 0.99×0.95×0.291 = 0.331

Step 6: PPO 更新
  # 归一化优势
  advantages = normalize([0.331, 0.291, 0.25])
  
  # 多轮更新
  for ppo_epoch in range(10):
      loss = policy_loss + 0.5×value_loss - 0.02×entropy
      optimizer.step()
```

---

## 7. 模型保存格式

```python
checkpoint = {
    'model_state_dict': model.state_dict(),  # 模型权重
    'model_type': 'GATPolicy',               # 模型类型
    'in_channels': 5,                        # 输入特征维度
    'hidden_channels': 64,                   # 隐藏层维度
    'heads': 4,                              # 注意力头数
    'best_reward': 0.65,                     # 最佳奖励
    'history': [[1, 0.45], [2, 0.48], ...],  # 训练历史
    'mode': 'robustness',                    # 训练模式
}

torch.save(checkpoint, 'models/gnn_ppo_agent_robustness.pth')
```

---

## 8. 关键代码文件

| 文件 | 功能 |
|------|------|
| `src/bit_hyrl/gnn_model.py` | GAT 策略网络定义 |
| `src/bit_hyrl/ppo_trainer.py` | PPO 训练器实现 |
| `src/bit_hyrl/reward.py` | 奖励函数定义 |
| `src/bit_hyrl/training.py` | 训练入口函数 |
| `scripts/train_bit_hyrl.py` | 训练脚本 |
| `scripts/train_bit_hyrl_improved.py` | 改进版训练脚本 |

---

## 9. 参考文献

1. **GAT**: Veličković et al., "Graph Attention Networks", ICLR 2018
2. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
3. **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016
