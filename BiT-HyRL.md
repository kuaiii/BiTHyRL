# BiT-HyRL: GAT + RL 框架文档

BiT-HyRL 核心模块：双峰拓扑构造 + GAT 策略网络 + 强化学习控制器选择。

---

## 1. 框架概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BiT-HyRL 主流程 (GAT + RL)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入: G(N, M) 或 (n, m)                                             │
│       ↓                                                              │
│  [1] 双峰拓扑构造  create_bimodal_theoretical(n, m, seed)             │
│       ↓                                                              │
│  G_BiT(N, M)  双峰度分布图                                            │
│       ↓                                                              │
│  [2] 节点特征提取  get_gnn_node_features(G_BiT)  → x (N, 128)         │
│  [3] 图结构转换    graph_to_pyg_data(G_BiT)      → edge_index (2, 2*M)│
│       ↓                                                              │
│  [4] GAT 策略网络  GATPolicy(x, edge_index)      → probs (N,), value  │
│       ↓                                                              │
│  [5] 贪婪选择 k 个控制器  gnn_predict / hybrid_gnn_select             │
│       ↓                                                              │
│  输出: centers (k,)  控制器节点 ID 列表                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块与数据逻辑

### 2.1 双峰拓扑构造

**函数**: `src.topology.reconstruction.create_bimodal_theoretical(n, m, seed)`

**输入**:
- `n`: 节点数
- `m`: 边数
- `seed`: 随机种子

**变换公式**:

$$\bar{k} = \frac{2m}{n}$$

$$A = \left( \frac{2 \bar{k}^2 (\bar{k}-1)^2}{2\bar{k} - 1} \right)^{1/3}$$

$$k_{\max} = \lfloor A \cdot n^{2/3} \rceil$$

度序列构造:
- Hub 节点: `n_high = 1` 个，度数 = `k_max`
- 低度节点: `n_low = n - 1` 个，度数分配为 `k_base` 或 `k_base + 1`，满足总度数 = `2m`

$$k_{\mathrm{base}} = \lfloor (2m - k_{\max}) / (n-1) \rfloor$$

$$\mathrm{remainder} = (2m - k_{\max}) \bmod (n-1)$$

$$\mathrm{degree\_seq} = [k_{\max}] \times 1 + [k_{\mathrm{base}}+1] \times \mathrm{remainder} + [k_{\mathrm{base}}] \times (n-1-\mathrm{remainder})$$

**输出**: `G_BiT` (nx.Graph)，节点数 n，边数 m，双峰度分布

---

### 2.2 节点特征提取 (Node2Vec)

**函数**: `get_gnn_node_features(G, embed_dim=128)` → `get_node2vec_features`

**输入**: `G` (nx.Graph)

**参数**:
- `dimensions` = 128
- `walk_length` = 20
- `num_walks` = 100
- `p` = 1, `q` = 1

**变换**:
1. Node2Vec 随机游走 + Skip-gram 训练 → 原始嵌入 `emb` (N, 128)
2. L2 归一化: `features = emb / ||emb||_2`

**输出**: `x` (N, 128), `node_list` (N,)

---

### 2.3 图结构转换

**函数**: `graph_to_pyg_data(G, node_features)`

**输入**: `G`, `x` (N, 128)

**变换**:
- `node_list = list(G.nodes())`
- `edge_index`: 无向图每条边 (u,v) 转为 `[u→v, v→u]`，形状 (2, 2*M)

**输出**: `x` (N, 128), `edge_index` (2, 2*M), `node_list` (N,)

---

### 2.4 GAT 策略网络

**类**: `GATPolicy(in_channels=128, hidden_channels=128, heads=8, num_layers=5)`

**输入**: `x` (N, 128), `edge_index` (2, 2*M), `selected_mask` (N,) bool

**前向变换**:

1. **输入投影**
   $$h^{(0)} = \mathrm{ELU}(\mathrm{LayerNorm}(\mathrm{Linear}_{128 \to 128}(x)))$$

2. **GAT 层** (5 层，带残差)
   - 层 1–4: 多头 GAT，`concat` 输出，维度 128
   - 层 5: 单头 GAT，维度 128
   $$h^{(l)} = h^{(l-1)} + \mathrm{ELU}(\mathrm{LayerNorm}(\mathrm{GAT}(h^{(l-1)}, \mathrm{edge\_index})))$$

3. **跳跃连接聚合**
   $$h = \mathrm{ELU}(\mathrm{LayerNorm}(\mathrm{Linear}([h^{(1)}; \ldots; h^{(5)}])))$$

4. **已选位置编码** (若 `selected_mask` 非空)
   $$h = h + \mathrm{selected\_mask} \odot \mathbf{e}_{\mathrm{sel}}$$

5. **全局上下文**
   $$\alpha = \mathrm{softmax}(\mathrm{MLP}(h)), \quad c = \sum_i \alpha_i h_i$$

6. **Actor 得分**
   $$\mathrm{scores} = \mathrm{MLP}([h; c]) / \tau$$
   $$\mathrm{probs} = \mathrm{softmax}(\mathrm{mask}(\mathrm{scores}))$$

7. **Critic 价值**
   $$v = \mathrm{MLP}(\mathrm{mean}(h))$$

**输出**: `probs` (N,), `value` (1,)

---

### 2.5 控制器选择

**函数**: `gnn_predict(G, k, model, deterministic=True)` 或 `hybrid_gnn_select`

**逻辑**:
1. 按连通分量分配控制器预算
2. 对每个分量: 调用 `gnn_predict` 贪婪选 k 个节点
3. 贪婪选择: `action = argmax(probs)`，选后 `selected_mask[action] = True`，重复 k 次

**输出**: `centers` (k,) 控制器节点 ID 列表

---

## 3. 数据流动

```
变量                    维度/类型              说明
────────────────────────────────────────────────────────────────
n, m                    int                   输入节点数、边数
G_BiT                   nx.Graph (N,M)        双峰拓扑
x                       Tensor (N, 128)       Node2Vec 特征
edge_index              Tensor (2, 2*M)       PyG 边索引
selected_mask           Tensor (N,) bool      已选节点掩码
h                       Tensor (N, 128)       GAT 隐层
probs                   Tensor (N,)           节点选择概率
value                   Tensor (1,)           状态价值
centers                 list (k,)             控制器节点 ID
```

**端到端数据流**:
```
(n, m, seed)
  → create_bimodal_theoretical     → G_BiT (N, M)
  → get_gnn_node_features          → x (N, 128), node_list
  → graph_to_pyg_data              → edge_index (2, 2*M)
  → GATPolicy.forward              → probs (N,), value
  → get_action (k 次贪婪)           → centers (k,)
```

---

## 4. 关键公式汇总

| 阶段       | 公式 |
|------------|------|
| 双峰 k_max | \(k_{\max} = A \cdot n^{2/3}\)，\(A = \bigl(\frac{2\bar{k}^2(\bar{k}-1)^2}{2\bar{k}-1}\bigr)^{1/3}\) |
| Node2Vec   | 随机游走 + Skip-gram，输出 L2 归一化 128 维 |
| GAT 概率   | \(\mathrm{probs} = \mathrm{softmax}(\mathrm{scores}/\tau)\) |
| 选择策略   | \(a_t = \arg\max_i \mathrm{probs}_i\)，\(\mathrm{mask}[a_t] \leftarrow \mathrm{True}\) |

---

## 5. 代码调用关系

```
main.py
  └─ ControllerManager
       └─ hybrid_gnn_select(G_BiT, k, model_path)
            └─ gnn_predict(sub_G, sub_k, model)
                 ├─ get_gnn_node_features(G)  → x, node_list
                 ├─ graph_to_pyg_data(G)      → edge_index
                 └─ GATPolicy.get_action(x, edge_index, selected_mask)
                      └─ forward() → probs, value
```

**拓扑生成** (main.py 中):
```
create_bimodal_theoretical(node_num, edge_num, seed) → G_BiT
```

---

**文档版本**: 1.0  
**更新**: 2026-01-31
