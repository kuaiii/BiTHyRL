# 修复 torch-scatter 和 torch-sparse 警告

## 问题描述
在 Windows 上运行训练脚本时出现警告：
```
UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage.
UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage.
```

## 解决方案

### 方案 1: 忽略警告（推荐，如果不需要 GNN 功能）

如果你不使用 `--use_gnn` 参数，这个警告**不影响训练**。代码会自动降级使用 MLP+REINFORCE 架构。

当前命令：
```bash
python scripts/train_bit_hyrl.py --epochs 100 --mode robustness
```
这个命令不使用 GNN，警告可以忽略。

### 方案 2: 修复依赖（如果需要使用 GNN 功能）

你的环境：
- PyTorch: 2.8.0
- CUDA: 12.6

#### 步骤 1: 卸载旧版本（如果有）
```bash
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv -y
```

#### 步骤 2: 安装预编译版本

由于 PyTorch 2.8.0 较新，可能需要从源码编译。推荐使用 conda 安装：

```bash
# 使用 conda（推荐）
conda install pyg -c pyg

# 或者使用 pip（尝试预编译版本）
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

#### 步骤 3: 如果上述方法失败，使用 CPU 版本（临时方案）

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

#### 步骤 4: 验证安装

```bash
python -c "import torch_scatter; import torch_sparse; print('安装成功！')"
```

### 方案 3: 修改代码延迟导入（避免警告）

如果不需要 GNN 功能，可以修改代码让 `torch-geometric` 的导入更懒加载，避免启动时的警告。

## 推荐做法

**对于当前训练（不使用 GNN）：**
- 直接忽略警告，继续训练即可
- 警告不影响 MLP+REINFORCE 训练

**如果将来需要使用 GNN 功能：**
- 使用 `--use_gnn` 参数时，需要先修复依赖
- 推荐使用 conda 安装 PyTorch Geometric 及其依赖
