# BiT-HyRL 完整框架文档

> 生成时间：2026-06-10  
> 基于当前代码版本整理

---

## 一、总体架构

```
输入图 G(N, E)
    │
    ▼
┌─────────────────────────────────────┐
│  特征提取 (Feature Extraction)       │
│  - Node2Vec 嵌入 (256D)             │
│  - 统计特征 (5D)                     │
│    · 归一化度数 · 聚类系数 · PageRank │
│    · 特征向量中心性 · Katz 指数       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  GAT 策略网络 (GATPolicy)            │
│  6-layer GAT, 12 heads, 256 hidden  │
│  Actor-Critic 架构                  │
└─────────────────────────────────────┘
    │
    ├──► Actor Head ──► 节点选择概率分布
    │
    └──► Critic Head ──► 状态价值估计
    │
    ▼
PPO 训练器 (收集轨迹 → GAE → 策略更新)
    │
    ▼
对抗奖励评估 (Adversarial Reward)
    │
    ▼
模型输出：最优控制器节点集合
```

---

## 二、输入 (Input)

### 2.1 输入数据
- **图结构**：NetworkX `Graph` 对象，无向连通图
- **节点数范围**：20 ~ 1000（默认过滤参数）
- **训练数据集**：`dataset/all/syn/` 下的 `.gml` 文件

### 2.2 控制器比例
```python
k = max(1, int(G.number_of_nodes() * k_ratio))
```
默认 `k_ratio = 0.1`，即选择约 10% 的节点作为控制器。

---

## 三、嵌入层 (Embedding)

### 3.1 Node2Vec 嵌入

```python
node2vec = Node2Vec(
    G_work,
    dimensions=256,      # 嵌入维度
    walk_length=20,      # 随机游走长度
    num_walks=100,       # 每个节点游走次数
    p=1, q=1,            # 返回/进出参数 (DeepWalk)
    workers=1,
    quiet=True,
    seed=42              # 固定随机种子（已修复）
)
model = node2vec.fit(window=10, min_count=1, batch_words=4, seed=42)
```

**输出**：`[num_nodes, 256]` 的节点嵌入矩阵，L2 归一化后使用。

### 3.2 统计特征 (5D)

| 维度 | 特征 | 说明 |
|------|------|------|
| 1 | `deg_norm` | 归一化度数 (degree / (n-1)) |
| 2 | `clust` | 局部聚类系数 |
| 3 | `pr` | PageRank 值（迭代 10 轮） |
| 4 | `eig` | 特征向量中心性（幂迭代 10 轮） |
| 5 | `katz` | Katz 中心性近似（归一化） |

### 3.3 最终输入特征

- **大图 (>200 节点)**：`Node2Vec(256D) + 统计(5D)` → **261D**
- **中图 (50~200 节点)**：同大图，Node2Vec 失败时退化为 `统计(5D) + 增强(7D)` → **12D**
- **小图 (<50 节点)**：`统计(5D) + 增强(7D)` → **12D**

> 注：GATPolicy 的 `in_channels` 设置为 `embed_dim=256`，与 Node2Vec 维度一致。统计特征在当前代码中**未被拼接到输入**（`get_gnn_node_features` 只返回 Node2Vec），这是 `ppo_trainer.py` 中使用的特征提取路径。`features.py` 中的 `get_node_features` 返回拼接特征，但未被训练脚本直接调用。

---

## 四、模型架构 (Model Architecture)

### 4.1 GATPolicy

```python
class GATPolicy(nn.Module):
    def __init__(
        self,
        in_channels=128,      # 输入特征维度
        hidden_channels=256,  # 隐藏层维度
        heads=12,             # GAT 注意力头数
        dropout=0.1,
        num_layers=6          # GAT 层数
    )
```

**模型参数量**：约 **152 万**（默认配置下）

### 4.2 前向传播流程

```
输入 x [N, 256]
    │
    ▼
┌─────────────────┐
│ 输入投影层       │
│ Linear(256→256) │
│ LayerNorm(256)  │
│ ELU()           │
└─────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ GAT Layer 1 (12 heads, concat)         │
│ GATConv(256, 21, heads=12) → [N, 252]  │
│ Linear(252→256) 投影修复维度不匹配      │
│ LayerNorm(256)                         │
│ ELU() + 残差连接                       │
├────────────────────────────────────────┤
│ GAT Layer 2~5 (同上)                   │
├────────────────────────────────────────┤
│ GAT Layer 6 (1 head, no concat)        │
│ GATConv(256, 256, heads=1) → [N, 256]  │
│ LayerNorm(256)                         │
│ ELU() + 残差连接                       │
└────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 跳跃连接 (Jumping Knowledge)             │
│ concat(6层输出) → [N, 1536]              │
│ Linear(1536→256)                        │
│ LayerNorm(256) + ELU()                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 已选节点位置编码 (可学习参数)             │
│ 对 mask=True 的节点加 bias              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 全局注意力池化                            │
│ Linear(256→256) → Tanh → Linear(256→1)  │
│ Softmax 得到注意力权重                    │
│ 加权求和得全局上下文 [1, 256]              │
│ 扩展到每个节点 [N, 256]                   │
└─────────────────────────────────────────┘
    │
    ├──► Actor Head ─────────────────────────────┐
    │                                              │
    │    concat(节点特征[N,256], 全局上下文[N,256]) │
    │    → [N, 512]                                │
    │    Linear(512→256) + LayerNorm + ELU         │
    │    Dropout(0.1)                              │
    │    Linear(256→128) + ELU                     │
    │    Dropout(0.1)                              │
    │    Linear(128→64) + ELU                      │
    │    Linear(64→1)                              │
    │    → 得分 scores [N]                         │
    │    温度缩放: scores / clamp(temperature)     │
    │    Mask 已选节点 (score = -inf)              │
    │    Softmax → 概率分布 probs [N]              │
    │                                              │
    └──► Critic Head ────────────────────────────┘
         全局平均池化 [N,256] → [1,256]
         Linear(256→256) + LayerNorm + ELU
         Dropout(0.1)
         Linear(256→128) + ELU
         Dropout(0.1)
         Linear(128→64) + ELU
         Linear(64→1)
         → 状态价值 value [1]
```

### 4.3 关键设计

| 设计 | 说明 |
|------|------|
| 残差连接 | 每层 GAT 输出 + 输入，缓解深层梯度消失 |
| 跳跃连接 | 聚合全部 6 层输出，保留多尺度信息 |
| 可学习温度 | `temperature` 参数，自动调节策略确定性 |
| 位置编码 | 对已选控制器节点添加可学习偏置 |

---

## 五、训练流程 (Training)

### 5.1 外层训练循环

```python
for epoch in range(200):
    # 1. 随机采样 60 个图（默认 collect_per_epoch）
    sampled_graphs = random.sample(graphs, 60)
    
    for G in sampled_graphs:
        k = int(G.number_of_nodes() * 0.1)
        
        # 2. 收集单图轨迹
        centers, reward = trainer.collect_trajectory(G, k, reward_fn)
    
    # 3. PPO 策略更新
    stats = trainer.update()
    
    # 4. 保存最佳模型
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_model_state = model.state_dict().copy()
```

### 5.2 轨迹收集 (Collect Trajectory)

```python
def collect_trajectory(G, k, reward_fn):
    # 获取 Node2Vec 特征
    x, node_list = get_gnn_node_features(G, embed_dim=256, seed=42)
    
    selected_mask = [False] * N
    centers = []
    experiences = []
    
    for step in range(k):
        # 模型选择下一个控制器
        action, log_prob, value, probs = model.get_action(
            x, edge_index, selected_mask, deterministic=False
        )
        
        centers.append(node_list[action])
        
        # 中间步骤奖励 = 0.01（存活奖励）
        # 最后一步奖励 = 对抗奖励函数
        if step < k - 1:
            step_reward = 0.01
        else:
            step_reward = reward_fn(G, centers)
        
        experiences.append(Experience(
            x, edge_index, action, log_prob, value, step_reward, selected_mask, done
        ))
        
        selected_mask[action] = True
    
    # 用最终奖励覆盖最后一步的经验奖励
    final_reward = reward_fn(G, centers)
    experiences[-1].reward = final_reward
    
    buffer.add(experiences)
    return centers, final_reward
```

### 5.3 PPO 更新

```python
def update():
    experiences = buffer.experiences  # 本轮全部经验
    
    # 1. GAE 计算优势函数
    advantages, returns = compute_gae(rewards, values, dones)
    advantages = normalize(advantages)
    
    # 2. Mini-batch PPO 更新（4 轮 epoch）
    for epoch in range(4):
        shuffle(experiences)
        for batch in split(experiences, batch_size=64):
            # 重新计算动作概率和价值
            probs, value = model(x, edge_index, selected_mask)
            new_log_prob = log(probs[action])
            
            # 比率
            ratio = exp(new_log_prob - old_log_prob.detach())
            
            # Clipped Surrogate Loss
            surr1 = ratio * advantage
            surr2 = clamp(ratio, 0.8, 1.2) * advantage
            policy_loss = -min(surr1, surr2)
            
            # Value Loss (MSE)
            value_loss = MSE(value, return_target)
            
            # Entropy（鼓励探索）
            entropy = Categorical(probs).entropy()
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            loss.backward()
            clip_grad_norm_(0.5)
            optimizer.step()
    
    buffer.clear()
```

### 5.4 GAE (Generalized Advantage Estimation)

```python
for t in reversed(range(T)):
    delta = reward[t] + gamma * V(t+1) - V(t)
    gae = delta + gamma * lambda * gae
    advantage[t] = gae
    return[t] = gae + V(t)

# 默认参数
gamma = 0.99      # 折扣因子
gae_lambda = 0.95 # GAE 平滑参数
```

---

## 六、损失函数 (Loss Function)

### 6.1 总损失

```
L_total = L_policy + 0.5 * L_value - 0.01 * L_entropy
```

### 6.2 策略损失 (Policy Loss)

```
ratio = π_new(a|s) / π_old(a|s)
L_policy = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

ε = 0.2  (clip 范围 [0.8, 1.2])
```

### 6.3 价值损失 (Value Loss)

```
L_value = MSE(V(s), GAE_Return)
```

### 6.4 熵正则 (Entropy)

```
L_entropy = H(Categorical(probs))
```

鼓励策略探索，防止过早收敛到局部最优。

---

## 七、奖励函数 (Reward Function)

### 7.1 对抗奖励公式

```python
reward = 0.7 * gcc_reward + 0.3 * dispersion_score
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `w_gcc` | 0.7 | GCC 保持率权重 |
| `w_disp` | 0.3 | 控制器分散度权重 |
| `attack_ratio` | 0.15 | 攻击移除节点比例 |

### 7.2 GCC 保持率计算

对每种攻击方法：
```python
# 1. 获取攻击序列（优先缓存）
sequence = nd.dismantle(G, method=method)  # 节点移除顺序

# 2. 模拟攻击（移除 attack_ratio=15% 的节点）
num_remove = int(N * 0.15)
removed = sequence[:num_remove]
G_attacked = G.remove_nodes(removed)

# 3. 计算 GCC 保持率
if controllers_survive:
    gcc_retention = len(max_connected_component) / N
else:
    gcc_retention = 0.0
```

### 7.3 聚合方式

```python
if aggregation == 'min':
    gcc_reward = min(gcc_scores)        # 取最脆弱攻击下的表现
elif aggregation == 'mean':
    gcc_reward = mean(gcc_scores)       # 平均表现
elif aggregation == 'max':
    gcc_reward = max(gcc_scores)        # 取最强攻击下的表现
```

默认 `aggregation='min'`（optimized 脚本），强化**最坏情况鲁棒性**。

### 7.4 分散度奖励

```python
def _get_dispersion_score(G, centers):
    """控制器之间的平均最短路径距离的归一化值"""
    distances = []
    for i, c1 in enumerate(centers):
        for c2 in centers[i+1:]:
            try:
                d = nx.shortest_path_length(G, c1, c2)
                distances.append(d)
            except:
                pass
    
    if not distances:
        return 0.0
    
    avg_dist = sum(distances) / len(distances)
    diameter = nx.diameter(G) if nx.is_connected(G) else max(distances)
    return avg_dist / (diameter + 1e-6)
```

鼓励控制器分散在网络不同区域，避免集中部署。

### 7.5 攻击方法池

| 类别 | 方法 | 速度 | 强度 |
|------|------|------|------|
| 拓扑指标 | degree, betweenness, pagerank, eigenvector | 快 | 中 |
| 随机 | random | 快 | 弱 |
| 集体影响 | CI_L1, CI_L2, CI_L3, CoreHD | 快 | 中-强 |
| GNN/ML | FINDER, FINDER+R, GDM, GDM+R, EGND | 慢 | 强 |
| 纠缠 | entanglement_small/mid/large, vertex_entanglement | 中 | 中 |

当前可用方法共 **24 种**。

### 7.6 课程式攻击采样 (Curriculum)

```python
# 前期 (epoch < 30%)：只用简单攻击
easy_methods = ['random', 'degree']

# 中期 (30% ~ 60%)：加入中等攻击
medium_methods = ['pagerank', 'betweenness', 'CI_L1']

# 后期 (epoch > 60%)：使用全部攻击
all_methods = 全部可用方法
```

默认启用（optimized 脚本）。

### 7.7 自适应聚合 (Adaptive Aggregation)

```python
# 前期 (0~50%)：mean 聚合，鼓励泛化
# 中期 (50~75%)：mean + min 混合
# 后期 (75~100%)：min 聚合，强化鲁棒性
```

默认启用（optimized 脚本）。

---

## 八、训练脚本参数

### 8.1 `train_bit_hyrl_optimized.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 200 | 训练轮数 |
| `--lr` | 3e-4 | 学习率 |
| `--k-ratio` | 0.1 | 控制器比例 |
| `--hidden-channels` | 256 | GAT 隐藏层维度 |
| `--heads` | 12 | GAT 注意力头数 |
| `--num-layers` | 6 | GAT 层数 |
| `--embed-dim` | 256 | Node2Vec 嵌入维度 |
| `--collect-per-epoch` | 60 | 每轮采样图数 |
| `--max-graphs` | None | 最大使用图数 |
| `--min-nodes` | 20 | 最小节点过滤 |
| `--max-nodes` | 1000 | 最大节点过滤 |
| `--attack-ratio` | 0.15 | 攻击比例 |
| `--aggregation` | min | 奖励聚合方式 |
| `--sample-methods` | 0 | 每次采样攻击方法数 (0=全部) |
| `--attack-methods` | None | 指定攻击方法池 (逗号分隔) |
| `--no-curriculum` | False | 禁用课程学习 |
| `--no-adaptive-aggregation` | False | 禁用自适应聚合 |
| `--show-node-stats` | False | 显示节点统计 |

---

## 九、推理流程 (Inference)

```python
def inference(G, model, k_ratio=0.1):
    model.eval()
    
    # 1. 提取特征
    x, node_list = get_gnn_node_features(G, embed_dim=256, seed=42)
    _, edge_index, _ = graph_to_pyg_data(G)
    
    k = int(G.number_of_nodes() * k_ratio)
    selected_mask = [False] * N
    centers = []
    
    with torch.no_grad():
        for _ in range(k):
            # 确定性选择（贪婪）
            action, _, _, probs = model.get_action(
                x, edge_index, selected_mask, deterministic=True
            )
            centers.append(node_list[action])
            selected_mask[action] = True
    
    return centers
```

推理时使用 `deterministic=True`，直接选择概率最高的节点，不采样。

---

## 十、模型保存格式

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_type': 'GATPolicy',
    'in_channels': 256,
    'hidden_channels': 256,
    'heads': 12,
    'num_layers': 6,
    'best_reward': best_reward,
    'history': [[epoch, avg_reward], ...],
}

torch.save(checkpoint, save_path)
```

---

## 十一、关键问题与修复记录

| 问题 | 原因 | 修复 |
|------|------|------|
| GAT 维度不匹配 (256/12=21, 21×12=252≠256) | heads 不能整除 hidden_channels | GAT 输出后加 Linear 投影回 256D |
| Node2Vec 随机性导致 best 停滞 | `fit()` 未固定 seed | 传入 `seed=42` 到 Node2Vec 和 fit() |
| 训练极慢 (24 种攻击) | sample_methods=0 使用全部攻击 | 添加 `--attack-methods` 和 `--sample-methods` 参数 |
| TensorFlow 警告 | TF 初始化信息 | 不影响训练，可忽略 |

---

## 十二、训练命令参考

### 快速测试（10 图 10 epoch）
```bash
conda run -n kanResilience python scripts/train_bit_hyrl_optimized.py \
    --epochs 10 --max-graphs 10 --sample-methods 5 --show-node-stats
```

### 正式训练（200 epoch，degree+betweenness+random）
```bash
conda run -n kanResilience python scripts/train_bit_hyrl_optimized.py \
    --epochs 200 \
    --attack-methods "degree,betweenness,random" \
    --show-node-stats
```

### 全量方法训练（较慢但更全面）
```bash
conda run -n kanResilience python scripts/train_bit_hyrl_optimized.py \
    --epochs 200 \
    --sample-methods 5 \
    --show-node-stats
```
