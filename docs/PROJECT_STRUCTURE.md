# 项目结构维护地图

> 用途：接手或长期维护本仓库时**先读这一篇**。
> 它回答四个问题：代码分几层、一次运行的数据怎么流、改东西该动哪个文件、哪些坑还在。
> 方法论文档见 `docs/BiT_HyRL_Framework.md`，公式速查见 `BiT-HyRL.md`。

最后更新：2026-10-28

---

## 1. 项目是什么

SDN 拓扑韧性研究框架，核心方法 **BiT-HyRL**：

```
真实/合成拓扑  →  双峰度分布重构  →  GAT+PPO 选控制器  →  多攻击拆解  →  四指标评估  →  与基线对比
```

- 目标图集：`dataset/testdata/*.gml` 共 **11 个**（BA-50/100/200/300/400/500、
  Chinanet、Cogentco、Colt、GtsCe、UsCarrier）
- 攻击模式：**11 种**（random / degree / betweenness / pagerank / eigenvector /
  adaptive_degree / adaptive_betweenness / adaptive_bridge / lookahead_gcc / bruteforce / wgcc）
- 评估指标：**GCC**（Giant Connected Component）、**CSA**（Control Supply Availability）、
  **CCE**（Control Centralization Entropy）、**WCP**（Weighted Control Potential），均越高越好
- 拆解基线：`network_dismantling.unified_interface` 中注册 **26 个**方法

---

## 2. 目录总览

```
BiT-HyRL/
│
├── main.py                     ★ 全方法对比入口（50 KB，参数最多）
├── main_bit_hyrl.py            ★ 只跑 BiT-HyRL 单方法
├── README.md                   项目总说明（含修复记录、指标定义）
├── README_TRAINING.md          训练模式说明（Scaled / Optimized）
├── BiT-HyRL.md                 公式与模块速查
├── requirements.txt            依赖
├── .gitignore                  入库规约（见 §6）
│
├── src/                        ★ 算法主体（54 个 .py，1.7 MB）
│   ├── topology/               拓扑构造与优化器（含 reconstruction.py 核心）
│   ├── bit_hyrl/               GAT 策略网络 + PPO + 特征 + 奖励
│   ├── controller/             控制器部署与批量调度
│   ├── unified_ppo/            拓扑微调 + 控制器选择的统一 PPO
│   ├── metrics/                指标计算（GCC/CSA/CCE/WCP）
│   ├── simulation/             攻击仿真
│   ├── utils/                  IO、日志、GPU、可视化
│   └── train/v1/               本地产物目录（仅剩训练曲线，权重已缺失）
│
├── network_construction/       重构基线适配层（Daimo/FRED_ABL/ONION/QDLM/ROMEN/UNITY）
├── network_dismantling/        拆解基线适配层（ADAPT/CI/CoreHD/EGND/EI/FINDER_ND/GDM/GND/…）
├── network_metrics/            指标基线适配层（functional/ + structural/）
│
├── scripts/                    ★ 63 个 .py + 3 .bat + 1 .sh（索引见 scripts/README.md）
├── docs/                       设计文档、方法文档、验证报告、推导记录
├── dataset/                    testdata（11 个 gml，入库）+ 生成脚本
│
├── results/                    运行产物（不入库）
├── logs/                       运行日志（不入库，当前为空）
├── archive/                    ★ 归档区（不入库，见 archive/README.md）
└── .kimi/                      协作技能约定（bit-hyrl-convention）
```

**入库 / 不入库的分界线**：源码、文档、脚本、11 个 testdata 图 → 入库；
模型权重（除 ADAPT 默认权重）、日志、results、`dataset/all` 训练集、`archive/` → 不入库。

---

## 3. 四个层次与职责

| 层 | 位置 | 职责 | 改动风险 |
|---|---|---|---|
| **入口层** | `main.py`、`main_bit_hyrl.py`、`scripts/alltest_main.py` | 参数解析、批量循环、调度各模块 | 高（外部调用都走这里） |
| **算法层** | `src/topology`、`src/bit_hyrl`、`src/unified_ppo`、`src/controller` | BiT-HyRL 三步走：重构 → 选控制器 → 微调 | 高（直接影响结论） |
| **适配层** | `network_construction`、`network_dismantling`、`network_metrics` | 把外部/基线算法包成统一接口，注册即插即用 | 低（只加不改） |
| **产物层** | `results/`、`logs/`、`archive/` | 曲线、CSV、图表、归档 | 无 |

适配层的三个统一接口，新增基线时只需实现对应函数 + 注册：

```python
# 构造基线：network_construction/unified_interface.py
construct(method, G, ...) -> nx.Graph

# 拆解基线：network_dismantling/unified_interface.py
@register_method("my_method")
def my_dismantle(G, ...) -> List[int]   # 返回移除顺序

# 指标：network_metrics/unified_interface.py
compute(...)
```

---

## 4. 一次完整运行的数据流（`main.py`）

```
1. 参数解析      --dataset GtsCe --attack random --bimodal-strategy adaptive_robust
2. 读图          src/topology/generators.load_graph("dataset/testdata/GtsCe.gml")
3. 双峰重构      src/topology/reconstruction.create_bimodal_adaptive_robust(...)
                   内部：k_max = A·N^(2/3) 推导 → 度序列 → 图化 → 连通修补 → 边数回补
                   默认 strategy=adaptive_robust，会采样多个 hub 数并打分选优
4. 控制器选择    src/bit_hyrl/gnn_model.GATPolicy + selection.py（贪心选 k=ceil(N*0.1)）
                   ⚠ 权重缺失时静默降级为 CI 算法，结果会失效（见 §7-②）
5. 可选微调      src/unified_ppo/inference.unified_solve(...) 产出 Unified-PPO / BiT-HyRL
6. 攻击拆解      src/controller/manager.ControllerManager：每个 batch 固定 attack_seed=42+ijk
                   保证所有方法面对同一攻击序列
7. 指标计算      network_metrics.compute(...) → GCC/CSA/CCE/WCP 逐步曲线
8. 落盘          src/utils/io.py → results/{dataset}/{attack}/{exp}/
                   curves/ + collapse_point + area + comprehensive_metrics.csv + execution_times.csv
```

**关键约定**：R 值（曲线下面积）必须以**实际移除比例**为横轴，不能用 `attack_step`。
`comprehensive_metrics.csv` 中的 R 才是准的（见 §7-①）。

---

## 5. 关键文件速查

| 我想改… | 去看 |
|---|---|
| 双峰构造逻辑（k_max 公式、hub 数、后处理） | `src/topology/reconstruction.py` |
| 构造策略的评分口径（balanced / random / wgcc / targeted / targeted_bridge） | `reconstruction.py::_candidate_topology_score` |
| 攻击序列与 R 值统计 | `reconstruction.py::_attack_auc_lcc`（正确口径）、`_evaluate_robustness`（旧口径） |
| GAT 网络结构 / 特征维度 | `src/bit_hyrl/gnn_model.py`、`src/bit_hyrl/features.py`、`config.py` |
| 控制器贪心选择规则 | `src/bit_hyrl/selection.py` |
| 奖励函数（拓扑微调用） | `src/bit_hyrl/reward.py`、`src/unified_ppo/reward_unified.py` |
| 批量调度、随机种子约定 | `src/controller/manager.py` |
| 四项指标的数学定义 | `src/metrics/metrics.py`、`network_metrics/functional/`、`structural/` |
| 结果文件名与目录结构 | `src/utils/io.py` |
| 加一个新拆解基线 | `network_dismantling/unified_interface.py`（照抄 CI 的 `@register_method` 写法） |
| 加一个新构造基线 | `network_construction/unified_interface.py` |

---

## 6. 产物存放规约

```
results/{dataset}/{attack}/{experiment_id}/
├── curves/                     折线图（GCC/CSA/CCE/WCP vs 移除比例）
├── collapse_point/             崩溃点数据
├── area/                       曲线下面积
├── comprehensive_metrics.csv   ★ 汇总指标（R 值以此为准）
└── execution_times.csv         各方法耗时
```

- 单次实验产物进 `results/`，**不要**写到项目根或 `src/`。
- 一次性诊断/扫描结果也进 `results/<主题>/`（如 `results/bimodal_diagnosis/`）。
- 已被取代但仍想留证据的 → `archive/`，并在 `archive/README.md` 记一行清单。
- 根目录不放大文件、日志、临时记录；一次性运行记录直接进 `archive/`（如 `problem_run_log.txt`）。
- 命名带数据集、攻击、实验号或日期，避免互相覆盖。

---

## 7. 已知技术债（按影响排序）

### ① 双峰构造把 `k_min` 破坏了 —— 影响论文主张，最高优先级

- **现象**：`docs/Bimodal.txt` 明确要求 `k_min ≥ 2`，但实测各真实网络最终 `k_min = 1`，
  约 20% 节点变成叶子（GtsCe 实测：设计 `k_min=2` → 最终 `1`）。
- **根因**：`create_bimodal_theoretical` 的后处理链自相矛盾——
  Havel-Hakimi 生成的单 hub 图不连通 → `_ensure_connectivity` 加边补连通（同时把 hub 度数
  抬到理论值约 1.7 倍）→ `_adjust_edge_count` 为了把边数改回 `m` 删边，而它的保护条件只有
  `degree > 1`，于是把低度层从 2 削到 1。
- **量化后果**：5 个真实网络平均 `R_degree` 从原始图 0.0896 降到 0.0325（0.36×）。
- **建议修法**：用**保度 double-edge swap** 替代"先加边再删边"；删边时禁止触碰 `k_min` 层。
  参考已存在的 `_ensure_connected_degree_preserving`。
- **诊断工具**：`scripts/diagnose_bimodal_construction.py`、`scripts/trace_bimodal_stages.py`。

### ② GNN 权重缺失时会静默降级

- `src/train/v1/checkpoints/` 现在只有一个训练历史 CSV，`.pth` 全部缺失（被 `.gitignore` 排除）。
- 后果：`main.py` 不报错，但控制器选择会降级成 CI 算法——**结果不再代表 BiT-HyRL**。
- 跑实验前必须确认权重路径存在，或在日志里搜索是否有降级提示。

### ③ 训练集 `dataset/all/` 未入库

- `.gitignore` 忽略 `dataset/all/`，当前 checkout 只有 `testdata`。
- 需要重训时先跑生成脚本。规模公式是 `count × len(sizes) × 拓扑类型数(8)`：

  ```bash
  # 全量（8 个规模 × 300 实例 × 8 类拓扑 = 19200 张图）
  python dataset/generate_datasets.py --split all --category syn --count 300 \
      --sizes 20 50 100 150 200 300 500 800 --seed 42
  # 快速验证（默认 5 个规模 × 100 实例 × 8 类 = 4000 张）
  python dataset/generate_datasets.py --split train --count 100
  ```

### ④ R 值存在两套统计口径

- `_attack_auc_lcc(G, attack_sequence)`：正确（以实际移除比例为横轴）。
- `_evaluate_robustness(G, seed)`：旧口径（用 `attack_step` 作横轴 → 严重低估）。
- 仍在使用旧口径的地方：`reconstruction.py` 中 `create_bimodal_adaptive` 的两处调用
  、`scripts/verify_bimodal_percolation.py`。
- 影响：`adaptive_*` 策略的**选优排序**基于旧口径，横向比较时不要直接拿它的 R 当最终结论。

### ⑤ 文档与实现不同步

- `docs/BiT_HyRL_Full_Pipeline.md` §2.3 描述的是"leaf 只连 hub、不连通就取 `largest` 子图"，
  那会**减少节点数**，与"固定 N"冲突；实际代码已改为 `_ensure_connectivity` + 边数回补。
- 修代码时请以 `src/topology/reconstruction.py` 为准，顺手把该文档一起更正。

---

## 8. 常见维护任务

### 重训一条完整链路

```bash
# 0) 环境
conda activate kanResilience          # 需 torch + networkx + torch_geometric

# 1) 训练集（规模换成自己的，公式见 §7-③）
python dataset/generate_datasets.py --split all --category syn --count 100
python scripts/generate_testdata.py   # 如需重建 testdata

# 2) 预计算嵌入（训练前建议）
python scripts/precompute_embeddings.py

# 3) 训练（README_TRAINING.md 首推 optimized）
python scripts/train_bit_hyrl_optimized.py --epochs 200 --attack-methods "degree,random"

# 4) 评测
python main.py --dataset GtsCe --attack wgcc --bimodal-strategy adaptive_robust

# 5) 汇总
python scripts/analyze_validation_results.py
python scripts/extract_metrics_table.py
```

### 只想复现某个数字

```bash
python scripts/diagnose_bimodal_construction.py --datasets Colt,GtsCe
python scripts/trace_bimodal_stages.py
```

### 给仓库瘦身 / 清理

- 新产生的大文件先判断：能不能重新生成？能 → `results/` 或 `archive/`，不要提交。
- 提交前跑 `git status` 确认没有 `*.pt`、`*.log`、`results/` 混进来。

---

## 9. 归档与清理策略

| 内容 | 去处 | 是否入库 |
|---|---|---|
| 当前实验的权重、曲线、CSV | `results/`、`src/train/v1/` | 否 |
| 被取代的历史权重快照、旧日志 | `archive/`（附 `MANIFEST.csv`） | 否 |
| 被取代的脚本 | 先在 `scripts/README.md` 标 `△`，确实无用再进 `archive/scripts_legacy/` | 否 |
| 一次性调试代码块 | 直接删（`git diff` 可回溯） | — |

**不要**用 `rm -rf` 清理 `results/` 或 `archive/`——里面有论文数据；先移到归档并记录清单。
仓库体积的大头历来是模型权重，加文件前先想"这个字节值得占用 git 历史吗"。

---

## 10. 文档分工

| 文档 | 读它是为了 |
|---|---|
| `README.md` | 项目全貌、指标定义、修复记录、结果目录结构 |
| `docs/PROJECT_STRUCTURE.md`（本文） | 代码分层、数据流、改哪儿、技术债 |
| `docs/BiT_HyRL_Framework.md` | 框架设计：为什么这样组合 |
| `docs/BiT_HyRL_Full_Pipeline.md` | 端到端流程细节（注意 §7-⑤ 的不同步） |
| `docs/GNN_RL_Architecture.md` | GAT + PPO 网络结构与训练细节 |
| `docs/CURRICULUM_TRAINING.md` | 课程学习训练 |
| `BiT-HyRL.md` | 公式速查（k_max、度分布、奖励） |
| `docs/Bimodal.txt` | 双峰构造的公式纠错推导（三条修正） |
| `docs/revision.txt` | 问题诊断过程与历史 R 值记录 |
| `docs/validation_results_*.md` | 多攻击多网络的排名验证结果 |
| `scripts/README.md` | 67 个脚本分别干什么、哪个是主用 |
| `archive/README.md` | 归档了什么、为什么、怎么恢复 |
