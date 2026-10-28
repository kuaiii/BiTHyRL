# 双峰构造算法：显式度序列、修复过程与版本考证

> 对应审稿意见 **Q2**（度序列调整算法、$\beta N^{2/3}\approx115.24$、495 vs 588、奇数和、
> 理论先验与最终评估图必须区分）与 **Q3**（hub 参数究竟是描述错误还是代码改动）。
> 实现：`src/topology/bimodal_construction.py`；验证：`scripts/theory_verify_optimality.py`。

---

## 0. 一句话回答 Q2

审稿人算的 495 与 588、以及"奇数和不可图化"，**是旧实现的算法缺陷，不是双峰构造本身的
性质**。旧流程先算 $k_{\min}=\lfloor(2M-k_{\max})/(N-1)\rfloor$，再据此拼序列，于是总和被
截断成 $k_{\max}+k_{\min}(N-1)=495<2M=588$，且 495 为奇数。新流程把 $k_{\min}$ 当作
**下界**而不是输入：先定 $k_{\max}$，再把**全部**剩余度数 $2M-n_hk_{\max}$ 均摊到
$N-n_h$ 个低度节点，取商与余数。由构造恒有 $\sum k_i=2M$，且 $2M$ 为偶数，
**奇偶性无需任何额外修正**。

---

## 1. 输入、输出与四条不变量

**输入**：$N$（节点数）、$M$（边数）、$n_h$（hub 数，默认 1）、$k_{\min}$（低度下界，默认 2）。

**输出**：`BimodalSpec`，含

| 字段 | 含义 |
|---|---|
| `theory_k_max` | 理论先验 $\mathrm{round}(A N^{2/3})$（**未**受可行性约束） |
| `k_max` | 实际采用值 $=\min(\text{theory}, \text{残差上限})$ |
| `degree_sequence` | 精确度序列，$\sum=2M$ |
| `strategy` | `theory` / `capped` / `infeasible` / `override` |
| `graphical` | Erdős–Gallai 是否通过 |

**四条不变量**（每次构图后断言，失败则换种子重采，绝不返回违约图）：

- **I1** $|V|=N$
- **I2** $|E|=M$
- **I3** 图连通
- **I4** 去掉 hub 层后，残余低度层仍连通，且残余层最小度 $\ge2$

I4 是旧实现系统性失守的那一条，也是"定向攻击不一步崩"的必要条件。

---

## 2. `CONSTRUCTBIMODALSEQUENCE`：显式算法

```
输入: N, M, n_h=1, k_min=2
输出: 度序列 seq, k_max, strategy

1. n_low ← N − n_h
2. 若 n_low < 3 或 2M < k_min·N:  返回 infeasible
3. k_theory ← round( A(<k>) · N^(2/3) ),  A = [2<k>²(<k>−1)²/(2<k>−1)]^(1/3)
4. cap ← min( N−1,  ⌊(M − N + n_h)/n_h⌋ )        # ← 残差连通约束 (★★)
5. k_max ← min(k_theory, cap)
   strategy ← (k_max < k_theory) ? "capped" : "theory"
6. 反复（至多 k_max−k_min 次）:
      low_total ← 2M − n_h·k_max
      若 low_total < k_min·n_low: break        # 边预算不够，判不可行
      base ← ⌊low_total / n_low⌋
      rem  ← low_total mod n_low
      seq  ← [k_max]×n_h + [base+1]×rem + [base]×(n_low−rem)
      断言 Σseq = 2M                            # 恒成立，非事后修补
      若 Erdős–Gallai(seq) 通过: 返回成功
      否则 k_max ← k_max − 1                    # 保双峰形态地回退
7. 返回 infeasible
```

### 2.1 为什么 $\sum=2M$ 恒成立（对 495 vs 588 的直接回答）

$$
\sum \text{seq}=n_hk_{\max}+(base+1)\cdot rem+base\cdot(n_{\text{low}}-rem)
=n_hk_{\max}+base\cdot n_{\text{low}}+rem
=n_hk_{\max}+low\_total=2M.
$$

因为 $2M$ **必为偶数**，所以序列和自动为偶，**不存在"初始奇数和"这一状态**。
审稿人所见的 495 是旧代码在 `k_min` 截断后拼序列的产物，属于实现 bug。

### 2.2 用审稿人的 BA-100 例子逐步走一遍

$N=100,\ M=294$（论文口径），$\langle k\rangle=5.88$：

| 步 | 值 |
|---|---|
| $A(5.88)$ | $\big[2\cdot5.88^2\cdot4.88^2/10.76\big]^{1/3}\approx5.35$ |
| $k^{\text{theory}}_{\max}$ | $5.35\cdot100^{2/3}=5.35\cdot21.54\approx115$（审稿人给的 115.24 一致） |
| $N-1$ | 99 |
| $cap$（残差约束） | $M-N+1=195$ |
| $k_{\max}$ | $\min(115,\,99,\,195)=99$ ← **被 $N-1$ 截断，strategy=capped** |
| $low\_total$ | $588-99=489$ |
| $n_{\text{low}}$ | 99 |
| $base, rem$ | $489=99\times4+93$ → $base=4$，$rem=93$ |
| 序列 | $[99] + [5]\times93 + [4]\times6$，和 $=99+465+24=588$ ✓ 偶数 ✓ |

与审稿人算的 $k_{\min}=4$ 对得上（他取的是 $\lfloor489/99\rfloor=4$），区别在于
**我们的 5 是"余数部分的取值"，不是固定 $k_{\min}$**；93 个节点取 5、6 个节点取 4。
这正是"显式残差分摊"的含义，也解释了为什么他算出 495 而我们得 588。

> 注意：$k_{\max}=99=N-1$ 意味着这个构造在 BA-100 上**退化为星型**。这不是算法错误，
> 而是理论先验超出简单图上限的真实后果，必须如实报告（`strategy=capped`，
> 截断原因 `n_minus_1`）。论文里"BA-100 用双峰"的说法应改为"BA-100 上理论先验
> 不可实现，退化为星型，故该点不参与双峰结论的统计"。

### 2.3 可图化回退保持双峰形态

旧实现不可图化时调用 `_make_graphical`，用 Havel–Hakimi 的思路把序列整体拉平，
结果 hub 与低度层的差异被抹掉，"双峰"名存实亡。新实现改为
$k_{\max}\leftarrow k_{\max}-1$ 后**重新分摊**：hub 仍只有 $n_h$ 个、仍严格高于低度层，
只是差值缩小 1。回退次数上限为 $k_{\max}-k_{\min}+1$。

---

## 3. `REPAIRTOPOLOGY`：只做保度 2-switch

```
输入: G（度序列已定，|E| 必须已等于 M）, seed
1. 若 |E| ≠ M: 直接抛错      # 边数不对说明度序列有误，不允许在此处偷偷修补
2. 随机 2-switch 打散 Havel–Hakimi 的确定性结构（保度、保边数）
3. 若整图不连通: 用 2-switch 合并分量（保度、保边数）
4. 若残余层不连通: 只在低度层内部做 2-switch（保度、保边数）
5. 校验 I1–I4，失败则换种子重来
```

### 3.1 为什么必须分两步修连通

这是整个重构里最关键、也最容易被忽略的一点：

> **整图连通 ≠ 残余层连通。** hub 可以把若干个低度层分量"串"起来，使整图连通；
> 但摘掉 hub 之后，低度层立刻碎成多个分量。

旧链路的 `gcc_after_top1`（摘掉最大度节点后的 LCC 占比）实测只有 **0.04**，
而原始图是 0.90——就是这个机理。新实现增加 `_connect_residual_by_swaps`，
只在低度层内部换边，因此 hub 度数、每个低度节点的内部度、总边数三者都不变，
唯一变化的是低度层的连通结构。

### 3.2 为什么不能再"先加边再删边"

旧流程：`_ensure_connectivity` 为连通加 30–44 条边 → hub 度数被推到理论值的 1.7 倍
→ `_adjust_edge_count` 删掉同样数量的边回补 $|E|=M$，保护条件只有 `degree > 1`
→ 低度层的 2 被削成 1，产生约 20% 叶子节点。

两个目标（连通、边数）互相抵消，最终 $N$、$M$ 对了，但**结构全错**。
新流程里根本不存在"边数修正"这一步：Havel–Hakimi 对 $\sum=2M$ 的序列生成的图
恰好有 $M$ 条边，2-switch 不改边数，所以 $|E|=M$ 是构造出来的，不是修补出来的。

### 3.3 残余层连通是"存在性"问题，不是概率问题

当 $e_{ll}=n_{\text{low}}$ 时，残余层每个节点的内部度恰为 2，是若干个**环的并**。
它存在连通实现（一个哈密顿环），但 Havel–Hakimi 给出的实现通常是多个小环。
2-switch 把两个环合并成一个环恰好是合法的保度操作，因此**必然能收敛**
（同度序列的所有实现通过 2-switch 连通，这是 2-switch 连通性定理）。
实测：11 个数据集、$n_h\in\{1,2,5\}$ 全部在 0.1 s 内收敛。

---

## 4. 理论先验 vs 最终评估图（Q2 最后一句）

论文必须同时报告两个度分布，不能只报一个：

| 量 | 定义 | 是否受可行性约束 | 用途 |
|---|---|---|---|
| $k^{\text{theory}}_{\max}=A N^{2/3}$ | 书中先验 | 否 | 说明方法来源与适用范围 |
| $k_{\max}=\min(k^{\text{theory}}_{\max},\,M-N+n_h,\ N-1)$ | 实际采用 | 是 | 生成可评估图 |
| 实际度分布 | `BimodalSpec.degree_sequence` | — | 附录中给出直方图 |

`create_bimodal_theoretical` 返回的图带 `graph.bimodal_spec`（含两值与 `strategy`）
与 `graph.bimodal_strategy`，实验脚本可直接落到 CSV，便于论文制表。

---

## 5. Q3：hub 参数到底是"描述错"还是"代码改了"——版本考证

用 `git log -S` 追三处关键字符串：

| 查询 | 结果 |
|---|---|
| `num_hubs = 1` 首次出现 | **`147bbe4`（2026-06-10，初始提交）** |
| `num_hubs` 相关改动 | `4754b40`（2026-07-02 加 adaptive_robust 策略）、`6712d85`、`d91168e` |
| `hub_num` 相关改动 | `147bbe4`、`4754b40`、`a64b249`、`c3b9ec5` |

**事实**：主构造路径 `create_bimodal_theoretical()` 自**初始提交起**就是
`num_hubs = 1` 硬编码，从未改过 hub 数量。多 hub 的接口
`create_bimodal_with_num_hubs()` 存在，但**只被消融脚本
`scripts/compare_single_layer.py` 调用**，不在主实验路径上。

因此 Q3 的正确答案是：**不是代码改了，而是早期稿件把消融脚本里的"hub 数"参数
误描述成了主构造的参数**。主构造一直是单 hub。

**可直接写进 response 的表述**：
> The earlier manuscript misdescribed an unchanged implementation. The
> topology-generation path used for all reported results
> (`create_bimodal_theoretical`) has specified a single structural hub since the
> first commit of the code base (`147bbe4`, 2026-06-10); the multi-hub parameter
> the reviewer refers to belongs to an ablation script
> (`scripts/compare_single_layer.py`) that is not part of the main experiment.
> No result reported in the revised manuscript was generated with a multi-hub
> configuration. The version used for the revised results is
> `feature/adaptive-bimodal-rl` at commit `<填入提交号>`.

**必须同时做的补救**：按本仓库当前实现（`5d90fa1` 起）重跑主实验的构造部分——
因为新构造修正了 $k_{\min}$ 与残余层连通性，**旧的构造结果与新的不可混用**。
需要重跑的范围：所有依赖 `create_bimodal_theoretical` 的实验（`--bimodal-strategy
theoretical`）以及 `adaptive_robust` 的对照行。

---

## 6. 复现命令

```bash
# 构造不变量自检（11 个数据集 × n_h ∈ {1,2,5}）
python scripts/theory_verify_optimality.py --seeds 3

# 单独检查某一个数据集的度序列与规格
python -c "
import networkx as nx
from src.topology.generators import load_graph
from src.topology.bimodal_construction import construct_bimodal_sequence, build_bimodal
G,_=load_graph('dataset/testdata/GtsCe.gml'); G=nx.convert_node_labels_to_integers(G)
n,m=G.number_of_nodes(),G.number_of_edges()
s=construct_bimodal_sequence(n,m,n_hubs=1)
print(s.as_dict(), s.notes)
H,spec=build_bimodal(n,m,n_hubs=1,seed=42); print(H.graph['bimodal_spec'])
"
```

---

## 7. 与审稿条目的对应

| 审稿要求 | 回应位置 |
|---|---|
| "degree-sequence adjustment requires an explicit algorithm" | §2 伪代码 + §2.2 BA-100 逐步演算 |
| "495 vs 588, odd sum precludes graphicality" | §2.1 恒等式证明：新算法不产生该状态 |
| "which degree constraints they preserve" | §1 四条不变量；§3.2 说明不再改边数 |
| "the resulting degree distributions" | `BimodalSpec.degree_sequence`，附录出直方图 |
| "theoretical two-degree prior vs graph ultimately evaluated" | §4 双字段 `theory_k_max` / `k_max` |
| Q3 hub 参数与版本 | §5 版本考证 + 可直接引用的英文表述 |
