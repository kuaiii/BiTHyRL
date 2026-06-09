# Controller Placement Strategies & Algorithms

本目录包含 SDN 控制器部署的核心算法实现。以下是当前系统中集成的几种主要算法的详细流程说明，供优化参考。

## 1. 启发式算法 (Heuristic Algorithms)

代码位置: `strategies.py`

### 1.1 K-Means 变种 (K-Median / K-Center)
这是经典的聚类算法在控制器部署中的应用。

**核心流程:**
1.  **初始化 (Initialization)**:
    *   随机选择 `k` 个节点作为初始控制器位置。
2.  **分配 (Assignment)**:
    *   遍历网络中所有节点，将其分配给距离最近的控制器（形成簇）。
    *   *距离度量*: 可以是跳数 (Hop count) 或地理距离 (Latency/Distance)。
3.  **更新 (Update)**:
    *   在每个簇内，重新寻找一个“中心”节点。
    *   **K-Median**: 寻找簇内某节点，使其到簇内其他所有节点的距离之和最小。
    *   **K-Center**: 寻找簇内某节点，使其到簇内最远节点的距离最小（最小化最大延迟）。
4.  **迭代 (Iteration)**:
    *   重复“分配”和“更新”步骤，直到控制器位置不再变化或达到最大迭代次数。

### 1.2 E-RCP (Enhanced Resilient Controller Placement)
这是一种基于社区划分的分层部署策略，旨在提高网络的抗毁性。

**核心流程:**
1.  **社区划分 (Community Detection)**:
    *   使用 `Louvain` 或 `Girvan-Newman (GN)` 算法将网络划分为若干个高内聚的社区。
    *   如果社区过大，可能会进行二次划分。
2.  **控制器分配 (Controller Allocation)**:
    *   根据控制器的总预算（`rate * N`），按社区大小比例分配控制器数量。
    *   确保每个社区至少分配 1 个控制器。
3.  **社区内选址 (Intra-community Placement)**:
    *   在每个社区内部，视为一个独立的子图。
    *   使用 **K-Median** 算法（通常使用向量化加速版本）在社区内选择指定数量的最佳控制器。
    *   *目标*: 最小化社区内节点到控制器的平均延迟。

---

## 2. RL-RCP (Proposed Method)

代码位置: `manager.py` (集成), `reconstruction.py` (双峰重构), `rl_optimizer.py` (RL选择)

这是我们提出的**高韧性控制器部署方法 (Resilient Controller Placement)**，核心思想是结合**拓扑重构**与**智能选址**。

### 2.1 第一阶段：双峰拓扑重构 (Bimodal Topology Reconstruction)
我们首先将初始网络重构为一个具有**双峰度分布 (Bimodal Degree Distribution)** 的网络。研究表明，双峰网络在面对蓄意攻击时具有更强的鲁棒性。

**重构流程:**
1.  **参数计算**:
    *   计算原网络的平均度 $<k>$。
    *   根据公式 $A = (\frac{2<k>^2(<k>-1)^2}{2<k>-1})^{1/3}$ 计算基础参数。
    *   确定两个峰值的度数 $k_{max}$ 和 $k_{min}$。
        *   $k_{max} \approx A \cdot (\frac{1}{rate})^{2/3}$ (控制器节点的度数)
        *   $k_{min}$ 根据总度数守恒计算得出 (普通节点的度数)。
2.  **度序列生成**:
    *   生成一个度序列，其中 `rate * N` 个节点拥有 $k_{max}$ 度（作为潜在控制器候选）。
    *   剩余节点拥有 $k_{min}$ 度。
3.  **图生成**:
    *   使用配置模型 (Configuration Model) 或 `expected_degree_graph` 根据该双峰度序列生成新的拓扑 $G_{bimodal}$。

### 2.2 第二阶段：RL 控制器选择 (RL-based Controller Selection)
在重构后的双峰拓扑 $G_{bimodal}$ 上，使用强化学习模型选择最优控制器位置。

#### 2.2.1 状态/特征表示
对于图中的每个节点 $v$，提取 3 维特征向量 $x_v$：
1.  **度中心性 (Degree Centrality)**
2.  **聚类系数 (Clustering Coefficient)**
3.  **介数中心性 (Betweenness Centrality)**

#### 2.2.2 模型架构 (Deep MLP)
*   **输入**: 节点特征矩阵 $(N, 3)$
*   **网络**: 4层全连接网络 (64隐藏单元) + ReLU激活。
*   **输出**: 每个节点成为控制器的概率 $P(v)$。

#### 2.2.3 训练与决策
*   **训练**: 使用大量合成与真实拓扑进行离线训练 (Offline Training)，以最大化网络在蓄意攻击下的连通性 (GCC) 为奖励。
*   **决策**: 在 $G_{bimodal}$ 上前向传播，根据输出概率选择 Top-K 个节点作为控制器。

---

## 3. 优化思路 (Optimization Directions)

基于上述流程，针对提出的 RL 方法，可以考虑以下优化方向：

1.  **图神经网络 (GNN) 引入**:
    *   目前使用 MLP 处理节点特征，忽略了拓扑结构信息（邻接矩阵）。
    *   *改进*: 使用 GCN, GAT 或 GraphSAGE 替代 MLP，聚合邻居特征，捕捉局部结构信息。
2.  **奖励函数设计**:
    *   目前的奖励仅考虑了“蓄意攻击下的GCC”。
    *   *改进*: 引入多目标奖励，综合考虑 **Rconn (鲁棒性)**, **平均路径长度**, **负载均衡** 等指标。
3.  **训练机制**:
    *   目前是 One-shot selection (一次选k个)。
    *   *改进*: 改为 **序列决策 (Sequential Decision)**，即逐个选择控制器，每个选择后更新图状态（掩码），使用 PPO 或 DQN 算法。
4.  **特征增强**:
    *   目前的 3 个特征较为基础。
    *   *改进*: 加入特征向量中心性 (Eigenvector Centrality)、PageRank、K-Core 值等。
