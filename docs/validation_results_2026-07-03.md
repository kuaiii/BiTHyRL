# BiT-HyRL 多攻击完整验证报告

**生成时间**：2026-07-03  
**验证分支**：`feature/adaptive-bimodal-rl`  
**验证版本**：`8a74392 Add attack-profile adaptive bimodal scoring`  
**核心方案**：双峰网络 + Attack-profile Adaptive Scoring + RL 控制器部署  

---

## 1. 验证目标

继续验证当前优化方案在非 degree 场景下的泛化能力，覆盖：

- 攻击方式：`random` / `wgcc` / `betweenness`
- 真实网络：`GtsCe` / `Chinanet` / `Colt` / `UsCarrier` / `Cogentco`
- 主要评价指标：GCC Area、CSA、CCE、WCP

本轮重点不是只确认单一攻击表现，而是判断当前 adaptive scoring 是否真正解决“跨攻击方式鲁棒性”问题。

---

## 2. 实验设置

运行命令模板：

```bash
python main.py --dataset {DATASET} --attack {ATTACK} --batch 1 --rate 0.1 --bimodal-score-profile auto
```

说明：

- 默认拓扑策略：`adaptive_robust`
- `auto` profile 映射：`random/wgcc -> wgcc`，`degree/betweenness/target -> targeted`
- 本轮为 `batch=1` 快速完整覆盖验证，结果仍可能存在随机方差，后续关键结论建议用 `batch=3` 复核。

---

## 3. 核心结论

### 3.1 一句话总结

> 当前优化已经显著修复 `wgcc` 场景，BiT-HyRL 在 5 个真实网络上全部取得 GCC Area 第一；`betweenness` 有小幅改善但仍弱于 UNITY+RL；`random` 表现明显退化，是下一步最优先修复对象。

### 3.2 BiT-HyRL 在各攻击方式下的 GCC Area

| Attack | GtsCe | Chinanet | Colt | UsCarrier | Cogentco | Avg Rank |
|--------|-------|----------|------|-----------|----------|----------|
| `random` | 0.2983(#5) | **0.4206(#1)** | 0.2176(#8) | 0.1620(#9) | 0.2362(#8) | **6.20** |
| `wgcc` | **0.2507(#1)** | **0.3629(#1)** | **0.2358(#1)** | **0.2273(#1)** | **0.2371(#1)** | **1.00** |
| `betweenness` | 0.2193(#2) | **0.3242(#1)** | 0.1606(#2) | 0.1686(#3) | 0.1989(#3) | **2.20** |

### 3.3 BiT-HyRL 四指标平均排名

| Attack | GCC | CSA | CCE | WCP | Overall |
|--------|-----|-----|-----|-----|---------|
| `random` | 6.20 | 6.20 | 6.20 | 7.80 | **6.60** |
| `wgcc` | **1.00** | **1.00** | 1.80 | 3.60 | **1.85** |
| `betweenness` | 2.20 | 3.60 | 5.40 | 6.00 | **4.30** |

### 3.4 GCC Area 第一名统计

| Attack | 第一名分布 | BiT-HyRL 结果 |
|--------|------------|---------------|
| `random` | GA+RL 2/5，UNITY+RL 1/5，BiT-HyRL 1/5，Onion+RL 1/5 | 1/5 第一，平均 #6.20 |
| `wgcc` | BiT-HyRL 5/5 | 5/5 第一，平均 #1.00 |
| `betweenness` | UNITY+RL 4/5，BiT-HyRL 1/5 | 1/5 第一，平均 #2.20 |

---

## 4. 相比 2026-07-02 的变化

| Attack | 2026-07-02 GCC Avg Rank | 2026-07-03 GCC Avg Rank | 变化 | 判断 |
|--------|--------------------------|--------------------------|------|------|
| `wgcc` | 3.00 | **1.00** | +2.00 | 显著改善，已成为优势场景 |
| `betweenness` | 2.60 | **2.20** | +0.40 | 小幅改善，但尚未压过 UNITY+RL |
| `random` | 5.40 | **6.20** | -0.80 | 明显退化，成为当前最大短板 |

四指标综合排名变化：

| Attack | 2026-07-02 Overall | 2026-07-03 Overall | 判断 |
|--------|---------------------|---------------------|------|
| `wgcc` | 3.35 | **1.85** | 明显改善 |
| `betweenness` | 4.45 | **4.30** | 基本持平，略有改善 |
| `random` | 4.95 | **6.60** | 明显变差 |

---

## 5. 退化原因分析

### 5.1 random 退化的直接原因

当前 `auto` profile 将 `random` 与 `wgcc` 都映射到 `wgcc` profile。这个策略对混合攻击非常有效，但对纯随机攻击不够合适：

- `wgcc` 仍包含 degree/targeted 成分，拓扑评分会保留较强的 hub 防御倾向。
- 纯 random 攻击更依赖全局冗余、低度层互联、低桥边比例和局部团簇的均匀备份。
- 当前候选拓扑被筛到更适合“随机 + 定向混合”的结构，导致在 Colt、UsCarrier、Cogentco 等网络上纯 random GCC 排名跌到 #8/#9/#8。

### 5.2 wgcc 大幅改善的原因

新增 attack-profile adaptive scoring 后，候选拓扑不再单纯追求 degree 场景下的 hub 防御，而是兼顾随机删除后的 GCC AUC。因此在 `wgcc` 场景中，BiT-HyRL 同时利用了：

- 双峰网络对高影响节点攻击的分散承压能力；
- 随机 AUC 评分对全局连通冗余的约束；
- RL 控制器在重构后网络上的部署收益。

这说明“双峰网络 + RL + 控制器部署”的主线是有效的，但 profile 需要进一步细分。

### 5.3 betweenness 仍不够强的原因

`betweenness` 攻击针对的是桥接节点和跨社区通道。当前结构虽然改善了 hub 分散，但对跨社区割点、桥边和高介数通路的显式约束还不足：

- CCE 平均排名仅 #5.40，说明控制效率/代价相关指标仍弱。
- WCP 平均排名 #6.00，说明控制器部署对最坏连通路径的保护不足。
- UNITY+RL 在 4/5 个网络上取得 GCC 第一，说明其社区融合或跨社区连接策略更适合介数攻击。

---

## 6. 下一步优化计划

### 6.1 优先修复 random：新增 pure-random profile

在 `src/topology/reconstruction.py` 中新增独立 `random` scoring profile，不再复用 `wgcc`：

- 提升纯 random AUC 权重，降低 degree/top-hub 权重；
- 增加低度节点冗余连接、平均局部聚类、非桥边比例、2-core 覆盖率；
- 惩罚过强 hub 依赖和低度层碎片化；
- 目标：`random` GCC Avg Rank 从 #6.20 提升到前 3，Overall 从 #6.60 提升到前 4。

### 6.2 针对 betweenness：加入 bridge/community-aware 评分

为 `targeted` profile 增加介数攻击专用项：

- 惩罚高 edge-betweenness 桥边集中；
- 奖励跨社区多路径连接；
- 增加 articulation point / bridge 数量约束；
- 目标：`betweenness` GCC Avg Rank 从 #2.20 提升到 #1.50 以内，同时改善 CCE/WCP。

### 6.3 保持 wgcc 优势：避免破坏当前最优场景

`wgcc` 当前已经 5/5 第一，后续修改必须保留回归验证：

```bash
python main.py --dataset GtsCe --attack wgcc --batch 1 --rate 0.1 --bimodal-score-profile auto
python main.py --dataset Chinanet --attack wgcc --batch 1 --rate 0.1 --bimodal-score-profile auto
python main.py --dataset Colt --attack wgcc --batch 1 --rate 0.1 --bimodal-score-profile auto
python main.py --dataset UsCarrier --attack wgcc --batch 1 --rate 0.1 --bimodal-score-profile auto
python main.py --dataset Cogentco --attack wgcc --batch 1 --rate 0.1 --bimodal-score-profile auto
```

### 6.4 建议验证顺序

1. 先实现 `random` 独立 profile，并在 5 个网络上跑 `random`。
2. 若 `random` 改善且 `wgcc` 不退化，再增加 betweenness bridge-aware 项。
3. 最后跑 `random / wgcc / betweenness × 5` 的完整回归。
4. 对最终版本使用 `batch=3` 复核，降低单次运行随机性。

---

## 7. 当前版本判断

当前版本适合作为 `wgcc` 优势版本保存：

- `wgcc`：已达到论文结果可用级别，5/5 网络 GCC 第一。
- `betweenness`：可作为次优结果，但还需要 CCE/WCP 改善。
- `random`：暂不适合作为最终结果，需要下一轮重点优化。

因此下一步不建议继续盲目增强 targeted/hub 防御，而应把优化重心转向 pure-random 连通冗余与 bridge/community-aware 结构约束。
