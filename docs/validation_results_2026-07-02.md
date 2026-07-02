# BiT-HyRL 优化后性能验证报告

**生成时间**：2026-07-02  
**验证分支**：`feature/adaptive-bimodal-rl`  
**核心优化**：Adaptive Robust Bimodal Topology + RL 控制器部署  
**测试环境**：NVIDIA GeForce RTX 3060 (12GB)，Python 3.9，PyTorch CUDA

---

## 1. 验证目标

验证 `docs/revision.txt` 中提出的优化方案后，BiT-HyRL 在多种攻击方式下是否已超越基线方法，并定位剩余短板。

## 2. 实验设置

### 2.1 测试数据集

5 个真实网络拓扑：

- `GtsCe`
- `Chinanet`
- `Colt`
- `UsCarrier`
- `Cogentco`

### 2.2 攻击方式

| 攻击方式 | 说明 |
|---------|------|
| `degree` | 度攻击，优先移除高度节点 |
| `betweenness` | 介数攻击，优先移除高中介节点 |
| `random` | 随机攻击 |
| `wgcc` | degree 与 random 加权（0.5:0.5）|

### 2.3 运行命令

```bash
python main.py --dataset {DATASET} --attack {ATTACK} --batch 1 --rate 0.1
```

默认启用：`--bimodal-strategy adaptive_robust`。

### 2.4 对比方法

- Baseline
- GA+RL
- GA+RCP
- Onion+RL
- ROMEN+RL
- UNITY+RL
- FRED-ABL
- QDLM
- **BiT-HyRL**

---

## 3. 核心结论

### 3.1 一句话总结

> **degree 攻击下 BiT-HyRL 全面领先（5/5 第一），但 random 攻击明显落后（0/5 第一），wgcc 综合表现处于第一梯队但未形成统治力。**

### 3.2 各攻击方式下 BiT-HyRL 平均排名

| 攻击方式 | GCC | CSA | CCE | WCP | 综合 |
|---------|-----|-----|-----|-----|------|
| **degree** | 1.00 | 1.40 | 1.60 | 1.80 | **1.45** |
| **wgcc** | 3.00 | 2.60 | 3.20 | 4.60 | **3.35** |
| **betweenness** | 2.60 | 3.80 | 5.40 | 6.00 | **4.45** |
| **random** | 5.40 | 3.80 | 4.40 | 6.20 | **4.95** |

### 3.3 各攻击方式下 GCC Area 第一名统计

| 攻击方式 | 最频繁第一名 | 次数 | BiT-HyRL 排名 |
|---------|-------------|------|---------------|
| degree | **BiT-HyRL** | 5/5 | **#1** |
| wgcc | 多方法并列 | 各 1/5 | #3（平均） |
| betweenness | UNITY+RL | 3/5 | #2.6（平均） |
| random | FRED-ABL | 3/5 | #5.4（平均） |

---

## 4. 详细结果：GCC Area

### 4.1 degree 攻击

| Method | GtsCe | Chinanet | Colt | UsCarrier | Cogentco | Avg Rank |
|--------|-------|----------|------|-----------|----------|----------|
| **BiT-HyRL** | **0.2395(#1)** | **0.3108(#1)** | **0.1597(#1)** | **0.1951(#1)** | **0.2115(#1)** | **1.00** |
| QDLM | 0.1665(#2) | 0.2978(#2) | 0.1525(#2) | 0.1434(#3) | 0.1505(#4) | 2.60 |
| UNITY+RL | 0.1661(#3) | 0.2551(#4) | 0.1238(#4) | 0.1147(#7) | 0.1568(#2) | 4.00 |
| GA+RL | 0.1637(#4) | 0.2733(#3) | 0.1263(#3) | 0.1342(#5) | 0.1425(#6) | 4.20 |

**结论**：degree 攻击是 BiT-HyRL 的最强场景，5 个真实网络全部第一，彻底解决了此前真实网络垫底的短板。

### 4.2 wgcc 攻击（degree + random 加权）

| Method | GtsCe | Chinanet | Colt | UsCarrier | Cogentco | Avg Rank |
|--------|-------|----------|------|-----------|----------|----------|
| **BiT-HyRL** | 0.2444(#2) | 0.2929(#4) | 0.2012(#3) | 0.1928(#5) | **0.2325(#1)** | **3.00** |
| QDLM | 0.2084(#7) | 0.3080(#3) | **0.2136(#1)** | 0.1999(#2) | 0.2207(#3) | 3.20 |
| GA+RL | 0.2157(#5) | **0.3578(#1)** | 0.1955(#4) | 0.1976(#4) | 0.2084(#4) | 3.60 |
| FRED-ABL | 0.2200(#4) | 0.2916(#5) | 0.2104(#2) | **0.2256(#1)** | 0.2055(#6) | 3.60 |
| UNITY+RL | **0.2466(#1)** | 0.3246(#2) | 0.1927(#5) | 0.1760(#7) | 0.2083(#5) | 4.00 |

**结论**：wgcc 下各方法分散领先，BiT-HyRL 仅 Cogentco 第一，综合排名第三。random 成分的弱势拖累了 wgcc 表现。

### 4.3 betweenness 攻击

| Method | GtsCe | Chinanet | Colt | UsCarrier | Cogentco | Avg Rank |
|--------|-------|----------|------|-----------|----------|----------|
| UNITY+RL | **0.2212(#1)** | 0.2896(#2) | 0.1685(#2) | **0.1917(#1)** | **0.2333(#1)** | **1.40** |
| **BiT-HyRL** | 0.2193(#2) | **0.3242(#1)** | 0.1499(#4) | 0.1686(#3) | 0.1989(#3) | **2.60** |
| GA+RL | 0.1865(#3) | 0.2869(#3) | **0.1818(#1)** | 0.1558(#5) | 0.1640(#6) | 3.60 |

**结论**：betweenness 攻击下 UNITY+RL 最强，BiT-HyRL 处于第二梯队，Chinanet 上表现最佳。

### 4.4 random 攻击

| Method | GtsCe | Chinanet | Colt | UsCarrier | Cogentco | Avg Rank |
|--------|-------|----------|------|-----------|----------|----------|
| FRED-ABL | **0.3500(#1)** | 0.3608(#3) | **0.3297(#1)** | **0.3124(#1)** | 0.2635(#6) | **2.40** |
| GA+RL | 0.3230(#2) | 0.3583(#4) | 0.2686(#3) | 0.2614(#3) | 0.3054(#3) | 3.00 |
| QDLM | 0.2766(#5) | 0.3494(#5) | 0.2819(#2) | 0.2560(#4) | **0.3293(#1)** | 3.40 |
| UNITY+RL | 0.2843(#4) | 0.3792(#2) | 0.2642(#4) | 0.1790(#8) | 0.2947(#5) | 4.60 |
| **BiT-HyRL** | 0.2613(#6) | 0.3249(#6) | 0.2570(#5) | 0.2988(#2) | 0.2200(#8) | **5.40** |

**结论**：random 攻击是 BiT-HyRL 当前最大短板，平均排名 5.40，未在任何数据集上取得第一。

---

## 5. 优化前后对比（真实网络，GCC Area）

| 数据集 | 优化前（最早实验） | 优化后（当前） | 排名变化 |
|--------|-------------------|---------------|---------|
| Chinanet | 0.1107 | **0.3108** | 中游 → **#1** |
| Colt | 0.0371 | **0.1597** | 末位 → **#1** |
| Cogentco | 0.0466 | **0.2115** | 末位 → **#1** |
| UsCarrier | 0.0497 | **0.1951** | 末位 → **#1** |
| GtsCe | 0.2395 | **0.2395** | #1 → #1 |

*注：优化前数据取 `results/{dataset}/degree` 下最早完整实验；优化后为 2026-07-02 使用 `adaptive_robust` 的 batch=1 验证结果。*

---

## 6. 原因分析

1. **拓扑筛选过度偏向 degree 攻击**
   - `create_bimodal_adaptive_robust` 在候选拓扑评分时，degree AUC 权重过高，导致选出的拓扑对 random 攻击下的低度层碎片化缺乏抵抗力。

2. **控制器策略对 random 泛化不足**
   - `gnn_ppo_agent_optimized.pth` 训练奖励以 adversarial / degree 为主，控制器放置策略在 random 攻击场景下并非最优。

3. **双峰结构的固有 trade-off**
   - 分散 hub 能抗 degree，但过度分散可能削弱 random 攻击下低度节点的互助连通性。

---

## 7. 下一步建议

### 7.1 优先：重平衡拓扑筛选（预期收益最大）

在 `src/topology/reconstruction.py` 中：
- 将 `random` 攻击 AUC 权重提升至与 `degree` 相当；
- 增加低度层 2-core 比例、代数连通度、边介数均匀性等抗 random 指标；
- 目标：random 攻击下 GCC Area 平均排名进入前 3。

### 7.2 其次：重训控制器策略

修改 `scripts/train_bit_hyrl_optimized.py`：
- 奖励函数改为多攻击 min/avg 聚合：
  ```
  reward = min(R_degree, R_betweenness, R_random) + 0.5 * avg(R_degree, R_random)
  ```
- 或采用对抗式课程学习，逐步增加 random 攻击比例。

### 7.3 工程验证

1. 跑 batch=3 降低随机方差：
   ```bash
   python main.py --dataset Colt --attack random --batch 3 --rate 0.1
   python main.py --dataset Chinanet --attack wgcc --batch 3 --rate 0.1
   ```

2. 消融实验确认短板来源：
   ```bash
   python main.py --dataset Colt --attack random --bimodal-strategy theoretical
   python main.py --dataset Colt --attack random --bimodal-strategy adaptive_robust
   ```

3. 修复 `gnn_ppo_agent_optimized_training_history.csv` 仅 41 字节的问题，确认模型训练状态。

---

## 8. 附录：输出文件

- `results/metrics_tables/validation_degree_b1_summary.txt` — degree 攻击详细排名
- `results/metrics_tables/multi_attack_summary.txt` — 多攻击方式汇总
- `results/metrics_tables/per_attack_full_rankings.txt` — 各攻击四指标完整排名
- `results/metrics_tables/before_after_real_degree.txt` — 优化前后对比
- `results/metrics_tables/metrics_table_*.csv` — wgcc 口径指标表格（GCC/CSA/CCE/WCP）

---

**报告结论**：当前优化在 degree 攻击上取得决定性突破，但在 random 攻击上存在明显短板。若论文以 wgcc 或 random 为主要评价口径，需要进一步平衡拓扑筛选与控制器训练。
