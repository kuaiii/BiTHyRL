# 审稿意见应答矩阵

> 9 条意见逐条对应到**已完成的工作**、**证据位置**与**仍需作者决策/补做的事**。
> 状态标记：✅ 已解决（有代码/数据支撑）｜🟡 已给方案与工具，待用真实数据跑｜🔴 需作者确认

| # | 审稿要点 | 状态 | 已做的事 | 证据位置 | 仍缺 |
|---|---|---|---|---|---|
| 1 | 关联 $M_t(v_i)$ 与 $M_t^{-1}(u_j)$ 混淆；需求是复制/分配/分割未定义；缺需求与容量参数、失效重分配、迭代至稳定；功能失效节点是否连边移除 | ✅ | 形式化区分两个关系；明确**需求=单位、容量=∞、服务域=复制式**；给出"一轮即收敛"的**证明**（删除无控制器分量不会产生新的无控制器分量）；明确功能失效节点连同关联边移除；新增 `removed/failed/skipped` 分项计数 | `docs/protocol/cascade_and_metrics.md` §1–§3；`src/simulation/dismantling.py::simulate_dismantling(return_diagnostics=True)` | 若审稿人坚持要容量约束，需另跑一套 `capacity_mode='finite'` 结果（当前等价 ∞） |
| 2 | 度序列调整需显式算法；BA-100 的 $\beta N^{2/3}\approx115.24$、495 vs 588、奇数和不可图化；理论先验与最终图要分开 | ✅ | 给出 `CONSTRUCTBIMODALSEQUENCE` 伪代码；**证明** $\sum=2M$ 恒成立且自动为偶（495/奇数是不可能出现的状态）；BA-100 逐步演算到 $[99]+[5]\times93+[4]\times6=588$；双字段 `theory_k_max`/`k_max`；附四条不变量 | `docs/protocol/construction.md` §2、§4；`src/topology/bimodal_construction.py::construct_bimodal_sequence` | 论文附录需贴出各数据集的实际度分布直方图（`BimodalSpec.degree_sequence` 可直接导出） |
| 3 | hub 参数究竟是描述错还是代码改了；说明用的版本与重跑范围 | 🟡 | 用 `git log -S` 考证：`num_hubs=1` 自初始提交 `147bbe4`（2026-06-10）起硬编码于主构造；多 hub 接口只被消融脚本 `compare_single_layer.py` 调用 → **结论是"描述错、代码没改"**；给出可直接引用的英文表述 | `docs/protocol/construction.md` §5 | **必须重跑**：新构造修正了 $k_{\min}$ 与残余层连通性，旧构造结果不能与新结果混用；请在 response 中填入本次提交号 |
| 4 | Table 5 均值给出 0.01456/0.01611，Table 11 却是 0.0091/0.0108；要观测数、样本量、拓扑权重、CI 方法、配对过程 | 🟡 | 写出三种聚合口径（A 等权拓扑均值 / B 池化按 run 数 / C 配对差值）的公式，并用合成数据复现出同量级的落差（A=0.0144、B=0.0117），说明**两个数都对，只是聚合单位不同**；提供聚类 bootstrap（以 topology 为聚类单位）+ Wilcoxon 的脚本 | `scripts/statistical_analysis.py`（`--demo` 即复现）；`docs/protocol/cascade_and_metrics.md` §7 | 用真实 per-run 数据跑一次，确认 Table 11 到底是 B 还是 C；若确实是 B，正文需改写并重算 |
| 5 | CSA/H_C 不显著不能等于"没有退化"；UsCarrier 上 CSA 0.366 vs 0.409、H_C 0.370 vs 0.484 要讨论；"without significant degradation" 要改成有限定表述 | ✅（措辞已给） | 给出替代表述与 trade-off 的写法；指出非劣性需要**预设 margin + 非劣性检验**，不能用"不显著"替代 | `docs/theory/bimodal_optimality.md` §8；本文档 §5 下方 | 需在正文补 UsCarrier 的机制解释（hub 少 → CSA 低；负载集中 → H_C 低） |
| 6 | Figure 5 列名（CCE/H_C/GCC）与图注（GCC/CSA/H_C）不一致；CCE 超过 1（修订后指标应 ∈[0,1]）；下排标 GtsCe 而图注写 Colt | 🟡 | 从代码层面确认：当前 `calculate_csa` 恒 $\le1$（上界 $1-e^{-0.2\vert\mathcal C\vert}$），出现 >1 只可能来自旧实现或"CCE"这一旧列名；建议用统一的指标实现重画 | `docs/protocol/cascade_and_metrics.md` §4；`src/metrics/metrics.py::calculate_csa` | 需要真实结果 CSV 才能重画；脚本可基于 `results/{ds}/{attack}/{exp}/comprehensive_metrics.csv` 直接出图 |
| 7 | PPO 训练协议不明：每拓扑单独训练还是迁移？回合预算、checkpoint 选择、训练/评估攻击隔离、重训成本是否单列 | ✅（并发现新问题） | 按代码事实给出协议表：**BiT-HyRL 走预训练策略前向推理，不按拓扑重训**；RL 基线 `train_and_select` 是逐拓扑在线训练；**并指出耗时口径不公平**——基线把训练时间并入 `deployment_time`，BiT-HyRL 只计推理，必须拆两列 | `docs/protocol/cascade_and_metrics.md` §7 | 补 checkpoint 的 sha256 与训练集清单；建议加 `--require-gnn-checkpoint` 防止静默降级为 CI（见同节 §7.2） |
| 8 | CSA 是饱和服务得分不是服务比例；要报 $\alpha_j$；移除/失效节点的贡献；$H_C$ 在"有控制器但总负载为 0"时如何定义；负载变小≠熵降低 | ✅ | 明确 $\alpha$ 是**全局常数 0.2**（无 $\alpha_j$）；单控制器可达节点贡献 $1-e^{-0.2}=0.1813$，不是 1；移除与失效节点贡献均为 0；$H_C$ 分母固定为初始总负载 ⇒ **同比缩放不改变 $H_C$**，只在比例重分配时变化；零负载情形代码已加显式分支并注明"应报告为未定义" | `docs/protocol/cascade_and_metrics.md` §4–§5；`src/metrics/metrics.py::calculate_control_entropy` | 正文 §5.6 需删除所有"负载变小⇒熵降低"的表述 |
| 9 | Table 7 要不确定性、多阶段边界、平局规则、前沿穷尽、GDM 实现/权重；自适应目标在不同重构图上重算；Table 8 的 0.456 与 Table 5 四均值 0.46575 的差异 | 🟡 | 攻击协议细节可从 `src/simulation/dismantling.py` 的 `get_dismantling_sequence` 逐条读出；GDM 权重路径确认在 `network_dismantling/GDM/models_newpg/gdm_nx_best.pth`（34 KB，已随仓库跟踪） | `scripts/statistical_analysis.py`；`docs/protocol/cascade_and_metrics.md` | Table 8 vs Table 5 的 0.456 / 0.46575 需用同一口径重算；差异很可能同 Q4（子集或聚合方式不同），应显式声明 |

---

## 一、必须先做的三件事（否则其余都无法定稿）

1. **重跑构造相关实验**。新构造（`5d90fa1` 起）修正了 $k_{\min}$ 被削到 1、以及残余层
   不连通两个硬缺陷，旧构造结果与新结果**不可混用**。涉及 `--bimodal-strategy theoretical`
   的全部表格，以及 `adaptive_robust` 的对照行。
2. **确认 Table 11 的聚合口径**（Q4/Q9 共同依赖）。用 `scripts/statistical_analysis.py`
   在真实 per-run 数据上跑一次，确定它是口径 B 还是 C，然后正文统一。
3. **确认 GNN checkpoint 存在**。当前 checkout 缺 `src/train/v1/checkpoints/*.pth`，
   `hybrid_gnn_select` 会静默降级为 CI，导致 BiT-HyRL 实际不是论文方法。这是最危险的
   一条：它不报错，只是结果悄悄变成另一个算法。

## 二、可以直接写进 response 的三段（已具备证据）

**关于 Q2 的 495/588**
> The discrepancy is an artefact of the previous implementation, not a property of the
> construction. The revised algorithm treats $k_{\min}$ as a lower bound rather than an
> input: after fixing $k_{\max}$, the entire residual degree $2M-n_hk_{\max}$ is distributed
> over the $N-n_h$ low-degree nodes with quotient and remainder, so $\sum_i k_i=2M$ holds
> identically and, since $2M$ is even, the parity repair is vacuous. For BA-100
> ($N{=}100,M{=}294$) this gives $[99]+[5]^{93}+[4]^{6}$, summing to 588.

**关于 Q3 的 hub 参数**
> The earlier manuscript misdescribed an unchanged implementation. The construction used
> for all reported results has specified a single structural hub since the first commit of
> the code base (147bbe4, 2026-06-10); the multi-hub parameter belongs to an ablation script
> that is not part of the main experiment.

**关于 Q7 的耗时**
> No per-topology retraining is performed for the proposed method: the policy is trained
> once offline and deployed by a single forward pass. We now report inference time and
> training time as separate columns; the RL baselines' `train_and_select` performs online
> per-topology training, and its cost is no longer folded into the deployment time.

## 三、需要作者拍板的两处学术取舍

| 取舍 | 选项 | 影响 |
|---|---|---|
| 服务域语义 | 复制式（当前）vs 划分式（每交换机归一个主控制器） | 改 CSA 与 $H_C$ 的数值与定义，需重跑全部指标表。**建议保持复制式并在文中写明** |
| 目标函数 | 只最大化 $\bar R$（当前 `adaptive_robust`）vs 鲁棒–效率联合 $J$ | **会翻转最优点**：只优化 $\bar R$ 得到近均匀解（$k_{\max}=4$–$16$），已不满足双峰定义；用 $J$ 得到单 hub 解（$k_{\max}$ 接近残差上限）。见 `docs/theory/bimodal_optimality.md` §7 |
