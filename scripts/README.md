# scripts/ — 脚本索引

目录内共 **78 个文件**：74 个 `.py` + 3 个 `.bat` + 1 个 `.sh`。全部为**平铺**结构，不要移动它们
到子目录——脚本内部统一用 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
（共 14 处 `parents[1]`）推算项目根，移动一层就会让 `sys.path` 指错位置。

运行前提：使用 `kanResilience` conda 环境（`torch` + `networkx` + `torch_geometric`），
在**项目根目录**下执行 `python scripts/<name>.py`。

状态标记：★ 论文主用入口 ／ ○ 可独立运行 ／ △ 历史迭代（保留参考，结果不必与当前结论一致）

---

## A. 主入口与流程编排（4）

| 文件 | 用途 | 状态 |
|---|---|---|
| `alltest_main.py` | 在 `dataset/testdata` 全部数据集上批量调用 `main.py` | ★ |
| `run_bit_hyrl.py` | 单方法端到端：重构 → 攻击仿真 → 绘图 | ★ |
| `test_main.py` | 主流程冒烟测试 | ○ |
| `bit_hyrl_reconstruction.py` | 重构入口的向后兼容包装 | ○ |

## B. 数据准备（5）

| 文件 | 用途 | 状态 |
|---|---|---|
| `generate_testdata.py` | 生成 `dataset/testdata` 下的 GML 拓扑 | ★ |
| `generate_training_data.py` | 按节点规模区间生成训练用合成网络 | ★ |
| `generate_real_network_augmentation.py` | 按 `dataset/all/real` 的统计特征做真实网络增强 | ★ |
| `generate_random_topology.py` | 生成指定 N/M 的随机连通拓扑 GML | ○ |
| `precompute_embeddings.py` | 预计算 Node2Vec + DRE 嵌入到缓存目录（训练前建议先跑） | ★ |

## C. 训练（16）

BiT-HyRL 主线：

| 文件 | 用途 | 状态 |
|---|---|---|
| `train_bit_hyrl.py` | 主训练入口，支持 K 折交叉验证 | ★ |
| `train_bit_hyrl_optimized.py` | 优化研究训练（README 首推） | ★ |
| `train_bit_hyrl_scaled.py` | 大规模重训 | ★ |
| `train_bit_hyrl_adversarial.py` | 对抗式多攻击训练 | ★ |
| `train_curriculum.py` | 课程学习训练（DynamicGATPolicy） | ★ |
| `train_phase4_enhanced.py` | 阶段 4 增强训练 | ★ |
| `train_deep_gat.py` | 深层 GAT + PPO | ○ |
| `train_real_network_specialist.py` | 真实网络专家模型训练 | ○ |

历史迭代（同一目标的多轮尝试，保留用于追溯训练曲线与超参来源）：

| 文件 | 用途 | 状态 |
|---|---|---|
| `train_bit_hyrl_agent.py` | Agent 离线训练（早期优化版） | △ |
| `train_bit_hyrl_improved.py` | 改进版训练 | △ |
| `train_improved.py` | 改进版训练（更早） | △ |
| `train_small_network.py` | 小规模网络优化训练 | △ |
| `train_rl_agent.py` | 早期 RL Agent 训练 | △ |

Unified-PPO（拓扑微调 + 控制器选择）：

| 文件 | 用途 | 状态 |
|---|---|---|
| `train_unified_ppo.py` | 统一 PPO 训练 | ★ |
| `train_unified_ppo_full.py` | 统一 PPO 完整训练 | ○ |
| `train_unified_model.py` | 统一模型训练（早期） | △ |

## D. 评测与横向对比（19）

| 文件 | 用途 | 状态 |
|---|---|---|
| `benchmark_full_comparison.py` | 全方法大规模对比（主评测） | ★ |
| `cross_algorithm_benchmark.py` | 构造算法横向基准 | ○ |
| `benchmark_unified_vs_baseline.py` | Unified-PPO vs BiT-HyRL A/B | ○ |
| `benchmark_optimization.py` | 运行耗时基准 | ○ |
| `fair_topology_comparison.py` | 公平拓扑对比 + Unified-PPO 差距诊断 | ○ |
| `compare_attack_auc_remove.py` | 各重构拓扑在 random/degree 下的 R 值对比 | ○ |
| `compare_basic_topology.py` | 基础拓扑消融（Baseline/GA/ONION/ROMEN/UNITY） | ○ |
| `compare_adversarial_vs_baseline.py` | 对抗训练模型 vs Baseline 模型 | ○ |
| `compare_single_layer.py` | 单层网络拓扑对比 | ○ |
| `compare_single_layer_attacks.py` | 单层构造在 random/degree/PageRank 下对比 | ○ |
| `compare_bilayer_singlelayer.py` | 双层 vs 单层拆解对比 | ○ |
| `evaluate_bit_hyrl_adversarial.py` | 对抗式多攻击验证 | ★ |
| `evaluate_curriculum.py` | 课程学习模型评估 | ○ |
| `evaluate_phase_models.py` | 各阶段模型在真实网络上的评估 | ○ |
| `evaluate_real_network_specialist.py` | 专家模型在 Colt/Chinanet/UsCarrier 上评估 | ○ |
| `eval_unified_ppo_v5.py` | Unified-PPO v5 快速评估（动态 B 预算） | ○ |
| `run_real_network_comparison.py` | 真实网络上批量跑 `main.py` 对比 | ○ |
| `run_unified_ppo_comparison.py` | 真实网络上 Unified-PPO 对比汇总 | ○ |
| `run_bimodal_docx_experiment.py` | 按 `docs/双峰网络实验代码.docx` 伪代码复现实验 | ○ |

## E. 结果分析与汇总（8）

| 文件 | 用途 | 状态 |
|---|---|---|
| `analyze_validation_results.py` | 汇总 `main.py` 的 degree/wgcc 结果并排名 | ★ |
| `analyze_multi_attack_results.py` | 多攻击结果汇总 | ★ |
| `extract_metrics_table.py` | 从 `results/` 提取 CCE/CSA/GCC/WCP 成表 | ★ |
| `analyze_topology_differences.py` | 重构拓扑 vs QDLM/GA 构造的拓扑差异 | ○ |
| `analyze_unified_ppo_gap_model.py` | Unified-PPO 差距建模（sklearn） | ○ |
| `analyze_unified_ppo_gap_simple.py` | 差距轻量分析 | ○ |
| `plot_training_history.py` | 训练曲线绘图 | ○ |
| `plot_degree_distribution.py` | 度分布柱状图（Bimodal/BA/Random，论文用图），输出到 `results/figures/` | ○ |
| `monitor_unified_ppo.py` | 解析日志监控 epoch/reward，检测平台期 | ○ |

## F. 双峰构造与理论验证（18）—— 本方向当前重点

| 文件 | 用途 | 状态 |
|---|---|---|
| `test_phase_theory.py` | **相图理论六项自检**：恒等式、μ≥2 与残差约束等价、ρ→0 的 `N^(−1/3)` 收敛速率、最优标度、regime 与 applicability 一致性、`n_h*` 单调性 | ★ |
| `phase_diagram.py` | **`(N,⟨k⟩)` 网格实验**：6 档规模 × 6 档平均度，验证三区划分与 `N_min` 预测；产出 `phase_experiment.csv` / `phase_best.csv` / `phase_tables.md` | ★ |
| `ratio_surface.py` | **`(n_h, k_max)` 联合曲面**：回答「两个峰各占多少」，检验 `R̄(k_max)` 的平台结构；产出 `surface.csv` / `surface_best.csv` | ★ |
| `topology_vs_baselines.py` | **同预算 (N,M) 基准**：双峰族（1hub/ftot/oracle）vs 随机连通图/BA/近正则/保度重连，7 数据集 × 5 seeds，输出排名与配对差（`results/topology_vs_baselines/`） | ★ |
| `service_domain_semantics.py` | **服务域语义检验**：复制式 vs 划分式的 CSA/$H_C/$k_i$ 对照，量化 $H_C$ 在复制式下的退化（`results/service_domain_semantics/`） | ★ |
| `theory_range.py` | **理论主脚本**：把书 §7.16 的 $f^{tot}$ 推导落到 $B(n_h)$ 族上，同时输出解析 $f^{rand}/f^{target}/f^{tot}$、残余层 $\mu$、两步分解 $A(\mu)$、实测 $R$ 与 $\bar L$，产出 `results/theory_range/`（含可直接进论文的 `family_tables.md`） | ★ |
| `test_theory_formulas.py` | 公式自查：验证 $r\to0$ 时 $f^{tot}\to2-1/(\langle k\rangle-1)$（书 7.68）与 $k_{\max}^*\cdot r^{2/3}\to A$（书 7.67） | ★ |
| `test_construction_sweep.py` | 构造器回归：7 个拓扑 × 13 个 $n_h$ 检查 I1–I5，统计成功率与不可行原因 | ★ |
| `diagnose_bimodal_construction.py` | 各构造策略的 R 值/结构指标横向诊断（产出 `results/bimodal_diagnosis/`） | ★ |
| `trace_bimodal_stages.py` | 逐阶段追踪后处理链，定位 `k_min`/边数在哪步被破坏 | ★ |
| `theory_verify_optimality.py` | △ 被 `theory_range.py` 取代。区别：它在**固定 $n_h$** 下扫 $k_{\max}$，等价于同时改 $\langle k\rangle$，不是书中那个族；结论方向正确但机制解释不准，仅保留历史结果（`results/theory_opt/`） | △ |
| `statistical_analysis.py` | 统计口径复核：等权/池化/配对三种聚合并列，聚类 bootstrap CI 与 Wilcoxon，用于解释 Table 5 与 Table 11 的差异 | ★ |
| `verify_bimodal_topology.py` | 验证保 N/M 前提下的鲁棒性与重构耗时 | ★ |
| `verify_bimodal_percolation.py` | 按 `docs/Bimodal.txt` 的综合指标（无控制器，仅 GCC）验证 | ○ |
| `scan_bimodal_hub_ratio.py` | 扫描 hub 比例对韧性的影响 | ○ |
| `bimodal_degree_share_scan.py` | 扫描最大度节点度数占比 `r_max` 对 AUC 的影响 | ○ |
| `inspect_bimodal_colt.py` | 在 Colt 上输出度分布特征与参数 | ○ |
| `test_bimodal_hubs_robustness.py` | hub 数量递增时的 R 值变化 | ○ |

## G. 工具与环境（8）

| 文件 | 用途 | 状态 |
|---|---|---|
| `normalize_edge_count.py` | 边数中性归一化工具 | ○ |
| `test_improved_training.py` | 改进版训练效果验证 | △ |
| `test_unified_model.py` | 统一模型集成测试 | ○ |
| `install_gnn.bat` | 安装 GNN 依赖（Windows） | ○ |
| `fix_pyg_dependencies.bat` | 修复 PyG 依赖（Windows） | ○ |
| `run_real_network_specialist_pipeline.sh` | 专家模型全流程（Linux） | ○ |
| `run_real_network_specialist_pipeline.bat` | 专家模型全流程（Windows） | ○ |
| （`__pycache__/` 运行后自动生成，已忽略） | — | — |

---

## 新增脚本的约定

1. 放本目录根，**不建子目录**；用 `ROOT = Path(__file__).resolve().parents[1]` 定位项目根。
2. 顶部写模块 docstring：一句话说明脚本产出什么、写到哪里。
3. 产出统一写到 `results/<dataset>/<attack>/<exp>/`，不要往根目录或 `archive/` 写。
4. 需要人工判断的中间结果（对比表、诊断 CSV）请带上日期或 `--tag`，便于回溯。
5. 若某个脚本被新版本取代，**不要删**，在本文档状态列标 `△` 并注明取代关系；
   确实不再需要的移到 `archive/scripts_legacy/`（建目录时同步更新本索引）。
