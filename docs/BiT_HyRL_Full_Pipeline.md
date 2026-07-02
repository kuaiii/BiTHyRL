# BiT-HyRL 完整框架流程文档

> 生成时间：2026-06-11  
> 版本：基于 DRE(256) + Node2Vec(256) = 512D 输入的 GAT+PPO 优化版本

---

## 一、整体架构概览

```
原始图 G (gml)
    │
    ▼
[Step 1] 数据加载与双峰重构
    │  scripts/train_bit_hyrl_optimized.py::load_training_graphs()
    │  src/topology/reconstruction.py::create_bimodal_network_exact()
    ▼
双峰网络 G_bimodal (hub + leaf 结构)
    │
    ▼
[Step 2] 特征提取
    │  src/bit_hyrl/gnn_model.py::get_gnn_node_features()
    │  ├─ get_node2vec_features()  → Node2Vec(256D)
    │  └─ get_dre_embedding()       → DRE(256D)
    ▼
节点特征 X ∈ R^(N×512)
    │
    ▼
[Step 3] GAT 策略网络
    │  src/bit_hyrl/gnn_model.py::GATPolicy
    │  ├─ 输入投影: Linear(512→256)
    │  ├─ 6层 GAT + 残差 + Jumping Knowledge
    │  ├─ Actor Head → 节点选择概率
    │  └─ Critic Head → 状态价值
    ▼
控制器选择动作序列 (k 个节点)
    │
    ▼
[Step 4] PPO 训练
    │  src/bit_hyrl/ppo_trainer.py::PPOTrainer
    │  ├─ collect_trajectory() 收集经验
    │  ├─ compute_gae() 计算优势函数
    │  └─ update() PPO 策略更新
    ▼
[Step 5] 奖励评估
    │  src/bit_hyrl/reward.py::calculate_adversarial_reward()
    │  ├─ 攻击模拟 (degree/betweenness/CI/GND 等)
    │  ├─ GCC 保持率
    │  └─ 控制器分散度
    ▼
奖励信号 → 更新策略网络
```

---

## 二、数据准备阶段

### 2.1 数据集

**路径：** `dataset/all/syn/`

数据集包含约 19,200 个合成图（实际有效 19,173 个），覆盖以下拓扑类型：
- BA (Barabási-Albert)
- BA-dense
- ER (Erdős-Rényi)
- WS (Watts-Strogatz) 多种参数
- HK (Holme-Kim powerlaw cluster)
- RG (Random Geometric)

**规模分布：**
```
20, 50, 100, 150, 200, 300, 500, 800 节点
```

### 2.2 训练脚本入口

**文件：** `scripts/train_bit_hyrl_optimized.py`

**主函数：**
```python
def main():
    args = parse_args()
    log_path = init_training_logger()      # 初始化日志
    
    graphs, stats = load_training_graphs(
        data_dir=args.data_dir,
        max_graphs=args.max_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        use_bimodal=args.use_bimodal,
        hub_ratio=args.hub_ratio,
        seed=args.seed,
        num_workers=8,                     # 8 进程并行加载
    )
    
    result = train_gnn_ppo_optimized(
        graphs,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
        dre_dim=args.dre_dim,
        attack_methods=args.attack_methods,
        sample_methods=args.sample_methods,
        attack_ratio=args.attack_ratio,
        collect_per_epoch=args.collect_per_epoch,
        use_curriculum=args.use_curriculum,
        use_adaptive_aggregation=args.use_adaptive_aggregation,
        ...
    )
```

### 2.3 双峰网络重构

**文件：** `src/topology/reconstruction.py`

**函数：** `create_bimodal_network_exact(G, hub_num=None, hub_ratio=0.15, seed=42)`

**功能：** 将任意连通图转换为 **双峰网络 (Bimodal Network)**：
- 一部分节点作为 **hub**（高度连接）
- 其余节点作为 **leaf**（低度连接，主要连接 hub）

**核心流程：**

```python
def create_bimodal_network_exact(G, hub_num=None, hub_ratio=0.15, seed=42):
    if hub_num is None:
        hub_num = max(1, int(G.number_of_nodes() * hub_ratio))
    
    # 1. 按度数选择 hub 节点
    degrees = dict(G.degree())
    hub_nodes = sorted(degrees, key=degrees.get, reverse=True)[:hub_num]
    leaf_nodes = [n for n in G.nodes() if n not in hub_nodes]
    
    # 2. 构建新的边集合
    new_edges = set()
    
    # 3. hub 之间保留/添加连接，形成核心骨架
    for i, u in enumerate(hub_nodes):
        for v in hub_nodes[i+1:]:
            if G.has_edge(u, v) or random.random() < 0.5:
                new_edges.add((min(u,v), max(u,v)))
    
    # 4. leaf 节点连接到 hub（不直接连接其他 leaf）
    for leaf in leaf_nodes:
        connected_hubs = [h for h in hub_nodes if G.has_edge(leaf, h)]
        if not connected_hubs:
            connected_hubs = random.sample(hub_nodes, min(2, len(hub_nodes)))
        for h in connected_hubs:
            new_edges.add((min(leaf,h), max(leaf,h)))
    
    # 5. 构建新图
    G_new = nx.Graph()
    G_new.add_nodes_from(G.nodes())
    G_new.add_edges_from(new_edges)
    
    # 6. 确保连通性
    if not nx.is_connected(G_new):
        largest = max(nx.connected_components(G_new), key=len)
        G_new = G_new.subgraph(largest).copy()
        G_new = nx.convert_node_labels_to_integers(G_new)
    
    return G_new
```

**为什么用双峰网络？**
- 控制器部署问题中，双峰结构有理论优势
- hub 节点失效影响大，leaf 节点失效影响小
- 迫使策略学习"hub 鲁棒性"和"分散性"的平衡

**调用位置：**
```python
# scripts/train_bit_hyrl_optimized.py::_load_single_graph()
if use_bimodal:
    hub_num = max(1, int(G_original.number_of_nodes() * hub_ratio))
    G = create_bimodal_network_exact(G_original, hub_num=hub_num, seed=0)
```

---

## 三、特征工程阶段

### 3.1 入口函数

**文件：** `src/bit_hyrl/gnn_model.py`

**函数：** `get_gnn_node_features(G, device=None, embed_dim=256, dre_dim=256, seed=42, walk_length=20, num_walks=100, use_cache=True, cache_dir=None)`

**输出：** `features ∈ R^(N×512)`，`node_list`

**数据流：**
```
图 G
    │
    ├──► Node2Vec ──► 256D 结构嵌入
    │
    └──► DRE ───────► 256D 度排名嵌入
    │
    ▼
concat(256D, 256D) = 512D
```

### 3.2 Node2Vec 嵌入

**函数：** `get_node2vec_features(G, dimensions=256, walk_length=20, num_walks=100, p=1, q=1, seed=42)`

```python
node2vec = Node2Vec(
    G_work,                 # 移除孤立节点后的工作图
    dimensions=256,         # 嵌入维度
    walk_length=20,         # 随机游走长度
    num_walks=100,          # 每个节点游走次数
    p=1, q=1,               # DeepWalk 参数
    workers=1,
    quiet=True,
    seed=seed
)
model = node2vec.fit(window=10, min_count=1, batch_words=4, seed=seed)

# 提取嵌入并 L2 归一化
embeddings = [model.wv[str(node)] for node in G.nodes()]
features = embeddings / ||embeddings||_2
```

**确定性保证：**
- 向 `Node2Vec` 和 `fit()` 显式传入 `seed`
- 确保同一图在不同 epoch 生成完全相同的嵌入

**缓存机制：**
```python
# 缓存键：图结构 MD5 + 嵌入参数
param_key = f"d{embed_dim}_dr{dre_dim}_wl{walk_length}_nw{num_walks}_s{seed}"
cache_path = f"dataset/embedding_cache/{graph_hash}_{param_key}.npy"

if os.path.exists(cache_path):
    return torch.from_numpy(np.load(cache_path))
```

### 3.3 DRE (Degree Ranking Embedding)

**函数：** `get_dre_embedding(G, dim=256)`

```python
degrees = dict(G.degree())
sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
rank_map = {node: i for i, node in enumerate(sorted_nodes)}

# 正弦/余弦位置编码
for node in G.nodes():
    r = rank_map[node]
    for i in range(dim):
        freq = 1.0 / (10000.0 ** (i / dim))
        emb[i] = sin(r * freq) if i%2==0 else cos(r * freq)
```

**含义：** 度数排名越靠前（hub 节点），DRE 编码越能反映其枢纽地位。

---

## 四、模型架构阶段

### 4.1 GATPolicy

**文件：** `src/bit_hyrl/gnn_model.py`

**类：** `GATPolicy`

**初始化参数：**
```python
GATPolicy(
    in_channels=512,      # DRE(256) + Node2Vec(256)
    hidden_channels=256,  # GAT 隐藏层维度
    heads=12,             # 注意力头数
    num_layers=6,         # GAT 层数
    dropout=0.1
)
```

**参数量：** 约 159 万

### 4.2 前向传播流程

```
X ∈ R^(N×512)
    │
    ▼
[输入投影]
Linear(512→256)
LayerNorm(256)
ELU()
    │
    ▼
[GAT Layer 1~5]
对每层：
    GATConv(256, 21, heads=12, concat=True) → [N, 252]
    Linear(252→256)  # 修复维度不能整除
    LayerNorm(256)
    ELU()
    + 残差连接
    │
    ▼
[GAT Layer 6]
GATConv(256, 256, heads=1) → [N, 256]
LayerNorm(256)
ELU()
+ 残差连接
    │
    ▼
[Jumping Knowledge]
concat(6层输出) → [N, 1536]
Linear(1536→256)
LayerNorm(256)
ELU()
    │
    ▼
[已选节点位置编码]
对 selected_mask=True 的节点加可学习偏置
    │
    ▼
[全局注意力池化]
attention = Softmax(MLP(h)) ∈ R^N
global_context = Σ(attention_i * h_i) ∈ R^256
扩展到每个节点 → [N, 256]
    │
    ├──► [Actor Head]
    │    concat(h_local, h_global) ∈ R^(N×512)
    │    MLP(512→256→128→64→1) → scores[N]
    │    temperature scaling + mask + softmax → probs[N]
    │
    └──► [Critic Head]
         global_mean_pool(h) → [1, 256]
         MLP(256→256→128→64→1) → value
```

### 4.3 动作选择

**函数：** `GATPolicy.get_action(x, edge_index, selected_mask, deterministic=False)`

```python
probs, value = self.forward(x, edge_index, selected_mask=selected_mask)

if deterministic:
    action = argmax(probs)
else:
    action = Categorical(probs).sample()

log_prob = log(probs[action])
return action, log_prob, value, probs
```

---

## 五、训练阶段

### 5.1 训练入口与奖励函数选择

**文件：** `src/bit_hyrl/training.py`

**函数：** `train_gnn_ppo_optimized(...)`

```python
# 根据模式选择奖励函数
if mode == 'gcc':
    reward_fn = calculate_gcc_focused_reward
elif mode == 'multi_attack':
    reward_fn = calculate_multi_attack_reward
elif mode == 'adversarial':
    # 对抗式多攻击奖励：从多种 network_dismantling 方法中采样
    if reward_kwargs:
        reward_fn = functools.partial(calculate_adversarial_reward, **reward_kwargs)
    else:
        reward_fn = calculate_adversarial_reward
else:
    reward_fn = calculate_gcc_focused_reward

in_channels = embed_dim + dre_dim  # 256 + 256 = 512
```

**CLI 参数默认值：**
```python
--mode adversarial                    # 默认对抗式奖励
--attack-methods "degree"             # 默认单攻击
--attack-ratio 0.15                   # 攻击比例
--sample-methods None                 # 不采样，使用全部方法
--use-curriculum False                # 默认关闭课程学习
--use-adaptive-aggregation False      # 默认关闭自适应聚合
--collect-per-epoch 60                # 每轮采样的图数
```

### 5.2 外层训练循环

**文件：** `src/bit_hyrl/ppo_trainer.py`

**函数：** `train_gnn_ppo(...)`

```python
model = GATPolicy(in_channels=512, hidden_channels=256, heads=12, num_layers=6)
trainer = PPOTrainer(model, lr=3e-4)

for epoch in range(start_epoch, start_epoch + epochs):
    set_current_epoch(epoch, total_epochs)  # 供奖励函数课程/自适应使用
    epoch_rewards = []
    
    # 随机采样 collect_per_epoch 个图
    sampled_graphs = random.sample(graphs, min(collect_per_epoch, len(graphs)))
    
    for G in sampled_graphs:
        k = max(1, int(G.number_of_nodes() * k_ratio))
        
        centers, reward = trainer.collect_trajectory(
            G, k, reward_fn,
            embed_dim=embed_dim, dre_dim=dre_dim, seed=42,
            walk_length=walk_length, num_walks=num_walks,
            use_cache=use_embedding_cache
        )
        epoch_rewards.append(reward)
    
    # PPO 策略更新
    stats = trainer.update()
    
    # 保存最佳模型
    avg_reward = np.mean(epoch_rewards)
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_model_state = model.state_dict().copy()
```

### 5.3 轨迹收集

**函数：** `PPOTrainer.collect_trajectory(G, k, reward_fn, ...)`

```python
# 1. 获取 DRE + Node2Vec 特征
x, node_list = get_gnn_node_features(G, device=DEVICE, ...)
_, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)

# 2. 逐步选择 k 个控制器
selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
centers = []
experiences = []

with torch.no_grad():
    for step in range(k):
        action, log_prob, value, probs = self.model.get_action(
            x, edge_index, selected_mask=selected_mask, deterministic=False
        )
        action_idx = action.item()
        center = node_list[action_idx]
        centers.append(center)
        
        # 中间步骤使用小的存活奖励
        if step < k - 1:
            step_reward = 0.01
        else:
            step_reward = reward_fn(G, centers)
        
        done = (step == k - 1)
        exp = Experience(x=x.clone(), edge_index=edge_index.clone(),
                         action=action, log_prob=log_prob, value=value,
                         reward=step_reward, selected_mask=selected_mask.clone(),
                         done=done)
        experiences.append(exp)
        selected_mask[action_idx] = True

# 3. 最终奖励覆盖最后一步
final_reward = reward_fn(G, centers)
experiences[-1] = experiences[-1]._replace(reward=final_reward)

# 4. 加入缓冲区
for exp in experiences:
    self.buffer.add(exp)
self.buffer.add_episode_reward(final_reward)

return centers, final_reward
```

### 5.4 GAE 计算

**函数：** `PPOTrainer.compute_gae(rewards, values, dones)`

```python
for t in reversed(range(T)):
    if t == T - 1:
        next_value = 0
    else:
        next_value = values[t + 1] if not dones[t] else 0
    
    delta = rewards[t] + gamma * next_value - values[t]
    gae = delta + gamma * lambda * (1 - dones[t]) * gae
    
    advantages.insert(0, gae)
    returns.insert(0, gae + values[t])

# 归一化优势函数
advantages = (advantages - mean) / (std + 1e-8)
```

**默认参数：**
- `gamma = 0.99`
- `gae_lambda = 0.95`

### 5.5 PPO 更新

**函数：** `PPOTrainer.update()`

```python
for epoch in range(4):                    # 4 轮 mini-batch epoch
    shuffle(experiences)
    for batch in split(experiences, 64):  # batch_size=64
        for idx in batch_indices:
            probs, value = self.model(exp.x, exp.edge_index, selected_mask=exp.selected_mask)
            dist = Categorical(probs)
            new_log_prob = dist.log_prob(exp.action)
            entropy = dist.entropy()
            
            # 重要性采样比率
            ratio = exp(new_log_prob - old_log_probs[idx].detach())
            
            # Clipped surrogate loss
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = clamp(ratio, 0.8, 1.2) * adv
            policy_loss = -min(surr1, surr2)
            
            # Value loss
            value_loss = MSE(value.squeeze(), returns[idx])
            
            batch_policy_loss += policy_loss
            batch_value_loss += value_loss
            batch_entropy += entropy
        
        # 平均损失
        batch_policy_loss /= batch_size
        batch_value_loss /= batch_size
        batch_entropy /= batch_size
        
        # 总损失
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
```

### 5.6 Checkpoint 保存

```python
checkpoint = {
    'model_state_dict': best_model_state,
    'model_type': 'GATPolicy',
    'in_channels': in_channels,        # 512
    'embed_dim': embed_dim,            # 256
    'dre_dim': dre_dim,                # 256
    'hidden_channels': hidden_channels,# 256
    'heads': heads,                    # 12
    'num_layers': num_layers,          # 6
    'best_reward': best_reward,
    'history': history,
}
torch.save(checkpoint, save_path)
```

---

## 六、奖励设计阶段

### 6.1 对抗奖励函数入口

**文件：** `src/bit_hyrl/reward.py`

**函数：** `calculate_adversarial_reward(G, centers, attack_ratio=0.1, attack_methods=None, aggregation='min', w_gcc=0.7, w_disp=0.3, sample_methods=None, use_curriculum=False, use_adaptive_aggregation=False, ...)`

```python
reward = w_gcc * gcc_reward + w_disp * dispersion_score
```

### 6.2 攻击方法池

```python
ADVERSARIAL_ATTACK_METHODS = [
    'degree', 'betweenness', 'pagerank', 'eigenvector', 'random',
    'CI_L1', 'CI_L2', 'CoreHD', 'GND', 'EGND',
    'EI_s1', 'GDM'
]

_ATTACK_METHOD_GROUPS = {
    'easy':   ['degree', 'random', 'pagerank'],
    'medium': ['betweenness', 'CI_L1', 'CI_L2', 'CoreHD'],
    'hard':   ['GND', 'EGND', 'EI_s1', 'GDM', 'eigenvector'],
}
```

**可用性过滤：** 实际训练前用 `network_dismantling.list_methods()` 过滤掉不可用方法。

### 6.3 课程式攻击采样

**函数：** `_get_curriculum_methods(epoch, total_epochs, available_methods)`

```python
progress = epoch / total_epochs

if progress < 0.25:
    selected = easy                          # 只使用简单攻击
elif progress < 0.5:
    selected = easy + 50% medium             # 加入部分中等攻击
elif progress < 0.75:
    selected = easy + medium + 50% hard      # 加入部分困难攻击
else:
    selected = available_methods             # 使用全部攻击
```

### 6.4 自适应聚合策略

**函数：** `_get_adaptive_aggregation(epoch, total_epochs, base_aggregation='min')`

```python
progress = epoch / total_epochs

if progress < 0.25:
    return 'mean'                            # 早期鼓励泛化
elif progress < 0.5:
    return 'mean'                            # 继续泛化
elif progress < 0.75:
    return 'mean'                            # 混合阶段（后续取 mean 和 min 平均）
else:
    return base_aggregation                  # 后期强化最坏情况（min）
```

### 6.5 GCC 保持率计算

```python
# 为每种攻击方式计算 GCC 保持率
gcc_scores = []
for method in available_methods:
    sequence = _get_cached_sequence(G, method)
    if sequence is None:
        sequence = nd.dismantle(G, method=method)
        _set_cached_sequence(G, method, sequence)
    
    retention = _simulate_attack_sequence(G, centers, sequence, attack_ratio)
    gcc_scores.append(retention)

# 聚合
if aggregation == 'min' or aggregation == 'worst':
    gcc_reward = min(gcc_scores)
elif aggregation == 'mean':
    gcc_reward = mean(gcc_scores)
elif aggregation == 'max':
    gcc_reward = max(gcc_scores)

# 自适应混合阶段 (0.5-0.75 progress)
if use_adaptive_aggregation and 0.5 <= progress < 0.75:
    gcc_reward = 0.5 * mean(gcc_scores) + 0.5 * min(gcc_scores)
```

### 6.6 攻击模拟

**函数：** `_simulate_attack_sequence(G, centers, attack_sequence, attack_ratio)`

```python
num_remove = max(1, int(G.number_of_nodes() * attack_ratio))
G_attacked = G.copy()
remaining_centers = set(centers)

for node in attack_sequence[:num_remove]:
    if node in G_attacked:
        G_attacked.remove_node(node)
        remaining_centers.discard(node)

if G_attacked.number_of_nodes() == 0 or not remaining_centers:
    return 0.0

components = list(nx.connected_components(G_attacked))
controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
return max(controlled_sizes) / total_nodes
```

### 6.7 分散度奖励

**函数：** `_get_dispersion_score(G, centers)`

```python
# 计算控制器两两之间的最短路径
for c1, c2 in pairs(centers):
    if nx.has_path(G, c1, c2):
        distances.append(nx.shortest_path_length(G, c1, c2))

avg_dist = mean(distances)
diameter = nx.diameter(G) if nx.is_connected(G) else total_nodes // 2
ideal_dist = diameter / 3.0
dispersion_score = min(avg_dist / ideal_dist, 1.0)
```

**最终奖励：**
```python
reward = 0.7 * gcc_reward + 0.3 * dispersion_score
```

---

## 七、推理阶段

### 7.1 推理入口

**文件：** `src/bit_hyrl/selection.py`

**函数：** `gnn_predict(G, k, model_path, embed_dim=128, dre_dim=0, deterministic=True)`

```python
# 1. 加载 checkpoint
checkpoint = torch.load(model_path)
embed_dim = checkpoint.get('embed_dim', 128)
dre_dim = checkpoint.get('dre_dim', 0)
in_channels = embed_dim + dre_dim

model = GATPolicy(in_channels=in_channels, ...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 特征提取
x, node_list = get_gnn_node_features(G, device=DEVICE, embed_dim=embed_dim, dre_dim=dre_dim, seed=42)
_, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)

# 3. 贪婪选择 k 个控制器
selected_mask = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE)
centers = []

with torch.no_grad():
    for _ in range(k):
        action, _, _, _ = model.get_action(
            x, edge_index, selected_mask=selected_mask, deterministic=True
        )
        action_idx = action.item()
        centers.append(node_list[action_idx])
        selected_mask[action_idx] = True

return centers
```

### 7.2 评估函数

**文件：** `src/bit_hyrl/training.py::evaluate_gnn_model(...)`

评估时使用 `deterministic=True` 选择控制器，并计算 `calculate_gcc_focused_reward`。

---

## 八、代码文件与函数索引

| 阶段 | 文件 | 核心函数/类 |
|------|------|------------|
| 数据生成 | `dataset/generate_datasets.py` | `generate_ba`, `generate_er`, `generate_ws`, `save_graph` |
| 双峰重构 | `src/topology/reconstruction.py` | `create_bimodal_network_exact()` |
| 训练脚本 | `scripts/train_bit_hyrl_optimized.py` | `load_training_graphs()`, `main()`, `init_training_logger()` |
| 训练入口 | `src/bit_hyrl/training.py` | `train_gnn_ppo_optimized()` |
| PPO 训练 | `src/bit_hyrl/ppo_trainer.py` | `train_gnn_ppo()`, `PPOTrainer`, `collect_trajectory()`, `update()`, `compute_gae()` |
| 统一训练 | `src/bit_hyrl/ppo_trainer.py` | `UnifiedPPOTrainer`, `train_unified_gnn_ppo()` |
| 特征提取 | `src/bit_hyrl/gnn_model.py` | `get_gnn_node_features()`, `get_node2vec_features()`, `get_dre_embedding()` |
| 模型架构 | `src/bit_hyrl/gnn_model.py` | `GATPolicy`, `forward()`, `get_action()` |
| 奖励函数 | `src/bit_hyrl/reward.py` | `calculate_adversarial_reward()`, `_simulate_attack_sequence()`, `_get_dispersion_score()`, `_get_curriculum_methods()`, `_get_adaptive_aggregation()` |
| 推理 | `src/bit_hyrl/selection.py` | `gnn_predict()`, `hybrid_rl_select()` |
| 日志 | `src/utils/logger.py` | `setup_logger()`, `get_logger()` |

---

## 九、关键数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                     训练阶段 (Training)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  gml 文件                                                    │
│    │                                                        │
│    ▼                                                        │
│  load_training_graphs() ──► create_bimodal_network_exact()  │
│    │                         (hub/leaf 双峰重构)              │
│    ▼                                                        │
│  G_bimodal ──► get_gnn_node_features()                      │
│    │              ├─ Node2Vec(256D)                         │
│    │              └─ DRE(256D)                              │
│    ▼                                                        │
│  X ∈ R^(N×512)                                              │
│    │                                                        │
│    ▼                                                        │
│  GATPolicy.forward()                                        │
│    │              ├─ Actor: probs[N]                        │
│    │              └─ Critic: value                          │
│    ▼                                                        │
│  get_action() ──► 选择节点 action                            │
│    │                                                        │
│    ▼                                                        │
│  collect_trajectory()                                       │
│    │              ├─ 选 k 个节点                            │
│    │              └─ 最后一步调用 reward_fn                 │
│    ▼                                                        │
│  calculate_adversarial_reward()                             │
│    │              ├─ curriculum / adaptive agg              │
│    │              ├─ dismantle(G, method)                   │
│    │              ├─ simulate attack (15% 节点)             │
│    │              ├─ GCC retention                          │
│    │              └─ dispersion score                       │
│    ▼                                                        │
│  reward ──► buffer.add()                                    │
│    │                                                        │
│    ▼                                                        │
│  update() ──► GAE ──► PPO loss ──► optimizer.step()        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 十、当前配置默认值

```python
# 训练参数
epochs = 200
lr = 3e-4
k_ratio = 0.1
collect_per_epoch = 60

# 模型参数
embed_dim = 256
dre_dim = 256
hidden_channels = 256
heads = 12
num_layers = 6

# Node2Vec 参数
walk_length = 20
num_walks = 100

# 奖励参数
mode = 'adversarial'
attack_ratio = 0.15
attack_methods = ['degree']
aggregation = 'min'
w_gcc = 0.7
w_disp = 0.3
use_curriculum = False
use_adaptive_aggregation = False
sample_methods = None
```

---

## 十一、训练命令

### 11.1 默认单攻击训练

```bash
cd /home/u2024010340/codes/BiTHyRL
conda run -n kanResilience python scripts/train_bit_hyrl_optimized.py \
    --epochs 200 \
    --attack-methods "degree" \
    --show-node-stats
```

### 11.2 多攻击推荐配置

```bash
conda run -n kanResilience python scripts/train_bit_hyrl_optimized.py \
    --epochs 200 \
    --attack-methods "degree,betweenness,CI_L1" \
    --attack-ratio 0.25 \
    --collect-per-epoch 120 \
    --no-curriculum \
    --no-adaptive-aggregation \
    --show-node-stats
```

---

## 十二、已知问题与改进方向

1. **单攻击 plateau：** 仅使用 `degree` 攻击时，最佳奖励约 0.56 后停滞。随机策略已接近 0.5，优化空间有限。
2. **攻击多样性：** 建议启用 `degree,betweenness,CI_L1` 等多攻击方法，让 `min` 聚合、课程学习和自适应聚合真正发挥作用。
3. **DRE 稳定性：** Node2Vec 已添加确定性 seed，保证同一图多次调用结果一致。
4. **嵌入缓存：** `dataset/embedding_cache/` 缓存 Node2Vec+DRE 结果，显著加速多 epoch 训练。
5. **后续方向：** 在控制器选择基础上扩展联合拓扑调整（添加/删除边）能力。
