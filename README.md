# BiT-HyRL: SDN Topology Resilience Analysis Framework

This project provides a framework for analyzing the resilience of Software-Defined Network (SDN) topologies under various attack scenarios and controller placement strategies, with a focus on the **BiT-HyRL** (Bimodal Topology Hybrid Reinforcement Learning) algorithm.

## Project Structure

```text
BiT-HyRL/
├── dataset/
│   ├── all/                     # 全部可用数据
│   │   ├── syn/                 # 合成训练数据集（用于 K-折交叉验证）
│   │   └── real/                # 真实网络数据集
│   └── testdata/                # 最终验证集（独立测试集，不参与训练）
├── src/
│   ├── bit_hyrl/                # BiT-HyRL 核心算法
│   │   ├── config.py            # 配置参数
│   │   ├── features.py          # 特征提取 (Node2Vec等)
│   │   ├── model.py             # RL模型定义
│   │   ├── reward.py            # 奖励函数
│   │   ├── selection.py         # 节点选择策略
│   │   ├── topology.py          # 拓扑重构
│   │   └── training.py          # 训练逻辑
│   ├── topology/                # 拓扑管理
│   │   ├── generators.py        # 网络生成与加载
│   │   ├── optimization.py      # 拓扑优化 (GA, Onion, ROMEN等)
│   │   └── reconstruction.py    # 拓扑重构策略
│   ├── controller/              # 控制器管理
│   │   ├── manager.py           # SDN Manager (部署、状态)
│   │   └── strategies.py        # 控制器放置策略
│   ├── metrics/                 # 旧版指标计算（保留兼容）
│   │   └── metrics.py
│   ├── utils/                   # 工具函数
│   │   ├── io.py                # I/O 工具
│   │   ├── logger.py            # 日志工具
│   │   ├── gpu_utils.py         # GPU 工具
│   │   └── visualization/       # 可视化模块
│   │       ├── plots.py
│   │       ├── result_plotter.py
│   │       └── training_history.py
│   └── train/v1/checkpoints/    # 保存的模型和训练历史
├── network_construction/        # 统一拓扑重构接口
├── network_dismantling/         # 统一网络拆解接口
├── network_metrics/             # 统一指标计算接口
│   ├── structural/              # 纯拓扑指标 (GCC, AUC, R-value)
│   └── functional/              # SDN功能指标 (CSA, CCE, WCP)
├── scripts/
│   ├── train_bit_hyrl.py        # BiT-HyRL 训练入口
│   └── run_bit_hyrl.py          # BiT-HyRL 推理入口
├── results/                     # 实验结果
├── logs/                        # 日志文件
├── main.py                      # 主实验入口 (对比实验)
├── main_bit_hyrl.py             # BiT-HyRL 独立运行入口
├── requirements.txt             # 依赖
└── README.md                    # 本文件
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

数据集结构说明：

| 目录 | 用途 | 说明 |
|------|------|------|
| `dataset/all/syn/` | 训练集 | 合成网络，用于 K-折交叉验证 |
| `dataset/all/real/` | 真实网络 | 真实拓扑数据集 |
| `dataset/testdata/` | 最终验证集 | 独立测试集，不参与训练，用于最终模型评估 |

**数据集划分策略:**
- `dataset/all/syn/` 中的数据用于 K-折交叉验证（内部划分训练集和验证集）
- `dataset/testdata/` 中的数据作为最终独立验证集，评估模型泛化能力

---

## BiT-HyRL 算法使用流程

### 训练-验证-测试 三阶段流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BiT-HyRL 工作流程                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 训练(K-折交叉验证) │───▶│ 最终验证集评估 │───▶│   对比实验    │      │
│  └──────────────────┘    └──────────────┘    └──────────────┘      │
│         │                      │                   │                │
│         ▼                      ▼                   ▼                │
│  dataset/all/syn/        dataset/testdata/    main.py              │
│  (内部K-折划分)          (独立验证集)         (对比实验)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 阶段 1: 训练 (Training)

使用 `scripts/train_bit_hyrl.py` 进行模型训练：

```bash
# 从头训练（使用全部训练数据）
python scripts/train_bit_hyrl.py --epochs 100 --mode robustness

# K-折交叉验证（推荐）
python scripts/train_bit_hyrl.py --epochs 100 --kfold 5 --mode robustness

# 增量训练（在已有模型基础上继续训练）
python scripts/train_bit_hyrl.py --epochs 50 --resume

# 训练后在 testdata 验证集上评估
python scripts/train_bit_hyrl.py --epochs 100 --evaluate
```

**训练参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | combined | 优化目标: combined, robustness, csa, entropy, wcp |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 0.0001 | 学习率 |
| `--kfold` | 0 | K-折交叉验证折数（0表示使用全部数据训练） |
| `--use_node2vec` | True | 是否使用 Node2Vec 特征 |
| `--use_stepwise` | True | 是否使用 Step-wise Reward |
| `--use_bimodal` | True | 训练时是否进行双峰拓扑重构 |
| `--hub_ratio` | 0.15 | 双峰拓扑的 Hub 节点比例 |
| `--resume` | False | 是否增量训练 |
| `--evaluate` | False | 训练后是否在 dataset/testdata 上评估 |
| `--log_level` | WARNING | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

**训练输出:**
- 模型文件: `src/train/v1/checkpoints/rl_agent_{mode}.pth`
- 训练历史: `src/train/v1/checkpoints/training_history_{mode}.csv`
- 训练曲线: `src/train/v1/checkpoints/training_curve_{mode}.png`

### 阶段 2: 验证 (Validation)

验证阶段使用 `dataset/testdata/` 中的数据集评估模型性能：

```bash
# 仅评估已有模型（不训练）
python scripts/train_bit_hyrl.py --evaluate --no_train

# 训练后自动评估
python scripts/train_bit_hyrl.py --epochs 100 --evaluate
```

**评估输出:**
- 平均奖励值
- 最小/最大奖励值
- 约束满足情况（连通性）
- 性能分布（优秀/中等/较差）

### 阶段 3: 测试/对比实验 (Testing)

使用 `main.py` 运行完整的对比实验：

```bash
# 基本用法
python main.py --dataset GtsCe --attack degree --batch_size 10

# 使用已训练的模型（测试模式）
python main.py --dataset Chinanet --attack random --batch_size 5 --test_mode

# 指定 BiT-HyRL 参数
python main.py --dataset GtsCe --attack betweenness --batch_size 10 \
    --metric_type robustness --episodes 100 --hub_ratio 0.15
```

**main.py 参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | GtsCe | 数据集名称（自动搜索 dataset/testdata 和 dataset/all 目录） |
| `--attack` | random | 攻击模式: random, target, degree, pagerank, betweenness, eigenvector, bruteforce |
| `--batch_size` | 1 | 仿真批次数 |
| `--rate` | 0.1 | 控制器放置比例 |
| `--metric` | gcc | 评估指标: gcc, efficiency, coverage |
| `--test_mode` | False | 测试模式（仅加载模型推理，不训练） |
| `--debug` | False | 启用调试日志 |

**BiT-HyRL 特定参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hub_ratio` | 0.15 | Hub 节点比例 |
| `--episodes` | 100 | RL 训练轮数 |
| `--metric_type` | robustness | 优化目标类型 |
| `--use_node2vec` | True | 使用 Node2Vec 特征 |
| `--use_stepwise` | True | 使用 Step-wise Reward |
| `--use_ci` | True | 使用 CI 算法 |

---

## 单独运行 BiT-HyRL

如果只需要运行 BiT-HyRL 算法（不进行对比实验），可以使用 `scripts/run_bit_hyrl.py`：

```bash
# 基本用法
python scripts/run_bit_hyrl.py --dataset GtsCe --rate 0.1

# 运行攻击仿真并绘图
python scripts/run_bit_hyrl.py --dataset GtsCe --rate 0.1 --run_attack --attack_mode degree

# 使用合成网络
python scripts/run_bit_hyrl.py --synthetic --nodes 100 --rate 0.1

# 指定模型路径
python scripts/run_bit_hyrl.py --dataset Chinanet --model_path src/train/v1/checkpoints/rl_agent_robustness.pth
```

---

## 可视化

### 如何画出训练过程图

训练过程图有两种获取方式：

**方式一：训练时自动生成（推荐）**

运行训练脚本时，结束后会自动保存训练曲线图，无需额外操作：

```bash
# MLP 或 GNN 训练结束后会自动保存
python scripts/train_bit_hyrl.py --epochs 100 --mode robustness
# 输出: src/train/v1/checkpoints/rl_agent_robustness_training_curve.png

python scripts/train_bit_hyrl.py --epochs 100 --mode robustness --use_gnn
# 输出: src/train/v1/checkpoints/gnn_ppo_agent_robustness_training_curve.png
```

- 图片路径：与模型同目录，文件名为 `{模型名}_training_curve.png`
- 内容：横轴 Epoch、纵轴 Average Reward，含移动平均线和最佳点标注
- 若不想生成图，可加参数：`--no_plot`

**方式二：用已有训练历史 CSV 重新画图**

若之前训练时生成了 `*_training_history.csv`（MLP 训练会生成），可用可视化模块画图或对比多条曲线：

```bash
# 画出 src/train/v1/checkpoints/ 下所有训练历史 CSV
python -m src.visualization.training_history

# 只画某一个文件
python -m src.visualization.training_history --file src/train/v1/checkpoints/rl_agent_robustness_training_history.csv

# 输出到指定目录
python -m src.visualization.training_history --output_dir results/training_curves

# 比较多条训练曲线（同一张图）
python -m src.visualization.training_history --compare \
    src/train/v1/checkpoints/rl_agent_robustness_training_history.csv \
    src/train/v1/checkpoints/rl_agent_combined_training_history.csv \
    --labels Robustness Combined

# 弹窗显示图形
python -m src.visualization.training_history --show
```

支持的 CSV 格式：表头需含 `Epoch` 和 `Reward` 或 `Avg_Reward`。MLP 与 GNN 训练都会在模型目录下生成 `*_training_history.csv`，可直接用上述命令重画或对比。

### 1. 训练进度可视化

训练过程中会自动显示进度条和实时统计信息：

```
============================================================
BiT-HyRL Training Started
============================================================
  Mode: robustness
  Epochs: 100
  Graphs: 169
  Features: 69D (Node2Vec)
  Learning Rate: 0.0001
  Device: cuda
============================================================

Training:  50%|█████████████████████           | 50/100 [05:23<05:20, 1.07epoch/s]
  Epoch  50 | Reward: 0.7234 | Best: 0.7456 ★ | Graphs: 169 | Time: 323s | ETA: 320s
```

**进度信息包括：**
- 当前 Epoch / 总 Epochs
- 当前平均奖励值
- 历史最佳奖励值（标 ★ 表示当前是最佳）
- 处理的图数量
- 已用时间和预估剩余时间

### 2. 可视化训练历史

使用 `src/visualization/training_history.py` 模块可视化训练过程：

```bash
# 绘制所有训练历史（自动搜索 models/ 目录）
python -m src.visualization.training_history

# 绘制指定文件
python -m src.visualization.training_history --file src/train/v1/checkpoints/training_history_robustness.csv

# 指定输出目录
python -m src.visualization.training_history --output_dir results/training_curves

# 比较多个训练历史
python -m src.visualization.training_history --compare \
    src/train/v1/checkpoints/training_history_robustness.csv \
    src/train/v1/checkpoints/training_history_combined.csv \
    --labels Robustness Combined

# 显示图形（而不仅仅是保存）
python -m src.visualization.training_history --show
```

**Python API:**

```python
from src.visualization.training_history import (
    plot_training_history,
    plot_all_training_histories,
    plot_comparison
)

# 绘制单个文件
plot_training_history('src/train/v1/checkpoints/training_history_robustness.csv')

# 绘制所有历史
plot_all_training_histories(models_dir='models', output_dir='results/curves')

# 比较多个训练
plot_comparison(
    ['src/train/v1/checkpoints/training_history_robustness.csv', 'src/train/v1/checkpoints/training_history_combined.csv'],
    labels=['Robustness', 'Combined'],
    output_path='results/comparison.png'
)
```

### 2. 可视化实验结果

实验结果自动保存在 `results/` 目录，包含以下图表：

```
results/{dataset}/{attack_mode}/{experiment_id}/
├── {metric_type}/
│   ├── curves/                  # 折线图（GCC/CSA/CCE/WCP vs 移除比例）
│   │   └── {attack_mode} - {METRIC} - Batch{N}.png
│   ├── collapse_point/          # 崩溃点图表
│   │   ├── collapse_point_bar_{attack_mode}_{metric}.png
│   │   └── collapse_point_box_{attack_mode}_{metric}.png
│   └── area/                    # 面积/攻击步数图表
│       ├── {attack_mode} - Attack Steps ({METRIC}) - Batch{N}.png
│       └── {attack_mode} - {METRIC} vs Attack Steps - Batch{N}.png
├── extradata/                   # 详细曲线数据 (CSV)
├── logs/                        # 实验日志
└── execution_times.csv          # 运行时间统计
```

### 3. 旧版训练曲线可视化

也可以使用根目录的简化脚本：

```bash
python plot_training_history.py
```

---

## 输出说明

### 实验结果

| 文件/目录 | 说明 |
|-----------|------|
| `curves/` | 各策略的 GCC/CSA/CCE/WCP 随攻击进展的变化曲线 |
| `collapse_point/` | 网络崩溃点（指标降至20%时的移除比例）统计 |
| `area/` | R值（曲线下面积）和攻击步数分析 |
| `extradata/` | 每次迭代的详细 CSV 数据 |
| `execution_times.csv` | 各算法运行时间统计 |
| `comprehensive_metrics.csv` | 综合指标汇总 |

### 指标说明

| 指标 | 全称 | 说明 | 方向 |
|------|------|------|------|
| GCC | Giant Connected Component | 最大连通分量大小 | 越高越好 |
| CSA | Control Supply Availability | 控制供应可用性 | 越高越好 |
| CCE | Control Centralization Entropy | 控制集中熵 | 越高越好 |
| WCP | Weighted Control Potential | 加权控制潜力 | 越高越好 |

**注意**: 所有评价指标都是**越高越好**，表示网络在攻击下的韧性更强。

### 训练奖励函数

训练时根据 `--mode` 参数使用不同的奖励函数：

| Mode | 奖励计算 | 适用场景 |
|------|----------|----------|
| `robustness` | 0.6×鲁棒性 + 0.25×覆盖率 + 0.15×分散度 | 抗攻击场景（推荐） |
| `csa` | CSA 值 | 控制可用性优化 |
| `entropy` | 控制熵 | 控制均衡性优化 |
| `wcp` | WCP 值 | 加权控制潜力优化 |
| `combined` | 综合加权（多指标） | 平衡优化 |

### 对比策略

| 策略 | 说明 |
|------|------|
| Baseline | 原始网络 + 随机控制器部署 |
| GA+RL | 遗传算法优化 + RL控制器部署 |
| Onion+RL | Onion结构优化 + RL控制器部署 |
| ROMEN+RL | ROMEN优化 + RL控制器部署 |
| UNITY+RL | UNITY优化 + RL控制器部署 |
| **BiT-HyRL** | 双模态拓扑 + 混合RL优化 (本算法) |

---

## 常见问题

### Q: 如何选择训练模式 (mode)?

- `robustness`: 专注于网络鲁棒性优化，推荐用于抗攻击场景
- `combined`: 综合多个指标的平衡优化
- `csa/entropy/wcp`: 针对特定指标的优化

### Q: 训练需要多长时间?

- 取决于数据集大小和 epochs 数量
- 典型的 100 epochs 训练约需 10-30 分钟
- 可以使用 `--resume` 进行增量训练

### Q: 如何判断模型是否训练好?

使用评估功能查看验证集上的性能：
```bash
python scripts/train_bit_hyrl.py --evaluate --no_train
```
- 平均奖励 > 0.6: 效果良好
- 平均奖励 0.4-0.6: 一般，建议继续训练
- 平均奖励 < 0.4: 较差，建议调参或重新训练

### Q: 测试模式 (--test_mode) 和正常模式的区别?

- 正常模式: 每次运行都会进行 RL 训练
- 测试模式: 仅加载已有模型进行推理，如果模型不存在会降级为 CI 算法

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{bit-hyrl,
  title={BiT-HyRL: Bimodal Topology with Hybrid Reinforcement Learning for SDN Controller Placement},
  author={...},
  journal={...},
  year={2024}
}
```

## 更新日志

### 2026-01-26: 训练流程优化和 K-折交叉验证支持

**主要更新:**

1. **数据集结构调整:**
   - 训练集路径改为 `dataset/all/syn/`
   - `dataset/testdata/` 作为独立最终验证集
   - 真实网络数据移至 `dataset/all/real/`

2. **统一接口层:**
   - `network_construction` — 统一拓扑重构算法调用
   - `network_dismantling` — 统一网络拆解算法调用
   - `network_metrics` — 统一指标计算接口

2. **K-折交叉验证:**
   - 支持 `--kfold K` 参数进行 K-折交叉验证
   - 自动保存每折模型，选择最佳折作为最终模型
   - 生成交叉验证结果图

3. **训练时双峰拓扑重构:**
   - 训练数据自动进行双峰拓扑重构
   - 严格检查约束（节点数、边数、连通性）
   - 约束不满足时降级为原始图

4. **日志级别调整:**
   - 默认日志级别改为 WARNING
   - 只输出关键信息、警告和错误
   - 进度信息通过 tqdm 进度条显示
   - 可通过 `--log_level DEBUG` 开启详细日志

5. **评价指标说明:**
   - 所有指标（GCC、CSA、CCE、WCP）都是越高越好
   - 添加了各训练模式的奖励函数说明

---

### 2026-01-26: 修复双峰拓扑重构约束问题

**问题描述:**
- 在 `test_main.py` 验证时发现，重构后的拓扑不满足"节点数-边数与输入拓扑相同"的约束
- 原因：`src/topology/reconstruction.py` 中的 `create_bimodal_network_exact` 函数在删除多余边时没有检查连通性，可能导致图不连通或边数不准确

**修复内容 (`src/topology/reconstruction.py`):**

1. **添加连通性保证函数 `_ensure_connectivity()`:**
   - 检测并连接所有不连通的分量
   - 优先使用高度节点进行连接

2. **添加边数精确调整函数 `_adjust_edge_count()`:**
   - 在添加/删除边时保持连通性
   - 删除边前检查是否会导致图不连通或产生孤立节点
   - 优先调整高度节点之间的边（保持双峰特性）

3. **添加度序列调整函数 `_make_graphical()`:**
   - 将度序列调整为可图化的序列
   - 基于 Erdős–Gallai 定理进行验证

4. **增强 `create_bimodal_network_exact()` 主函数:**
   - 确保低度节点的度数至少为1（保证连通性）
   - 添加最终约束验证（节点数、边数、连通性）
   - 添加详细的日志记录

**约束保证:**
- 节点数 = 原图节点数（严格相等）
- 边数 = 原图边数（严格相等）
- 图必须连通

---

## License

MIT License
