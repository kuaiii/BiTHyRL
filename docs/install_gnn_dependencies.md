# 安装 GNN 依赖包（torch-scatter, torch-sparse）

## 问题
在 Windows 上安装 `torch-scatter` 和 `torch-sparse` 时出现错误：
```
OSError: [WinError 127] 找不到指定的程序
```

## 解决方案

### 方案 1: 使用 Conda（推荐，最简单）

如果你安装了 Anaconda 或 Miniconda：

```bash
# 安装 PyTorch Geometric 及其所有依赖
conda install pyg -c pyg

# 或者单独安装
conda install pytorch-scatter pytorch-sparse -c pyg
```

### 方案 2: 使用 pip + 预编译包（如果网络正常）

```bash
# 方法 A: 从官方源安装（需要稳定的网络）
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

# 方法 B: 如果上述失败，尝试使用信任的源
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html --trusted-host data.pyg.org
```

### 方案 3: 降级 PyTorch 到有预编译包的版本

PyTorch 2.8.0 较新，可能没有预编译包。可以降级到 2.5.0：

```bash
# 卸载当前 PyTorch
pip uninstall torch torchvision torchaudio -y

# 安装 PyTorch 2.5.0 + CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 然后安装 PyG 依赖
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

### 方案 4: 从源码编译（需要 Visual Studio）

如果以上方法都失败，需要从源码编译：

**前置要求：**
- Visual Studio 2019 或更新版本（包含 C++ 构建工具）
- CUDA Toolkit

```bash
# 安装构建依赖
pip install ninja

# 从源码安装
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv --no-binary :all:
```

### 方案 5: 使用 CPU 版本（临时方案）

如果只需要测试功能，可以使用 CPU 版本：

```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import torch_scatter; import torch_sparse; print('✓ 安装成功！')"
```

## 推荐步骤

1. **首先尝试 Conda**（如果已安装）：
   ```bash
   conda install pyg -c pyg
   ```

2. **如果 Conda 不可用，尝试降级 PyTorch**（方案 3）

3. **如果都不行，使用 CPU 版本**（方案 5）进行测试

## 当前环境信息

- PyTorch: 2.8.0+cu126
- CUDA: 12.6
- Python: 3.9
- 操作系统: Windows

## 注意事项

- PyTorch 2.8.0 是最新版本，可能缺少预编译包
- Windows 上从源码编译需要 Visual Studio 构建工具
- 如果遇到 SSL 证书错误，可能需要配置代理或使用 conda
