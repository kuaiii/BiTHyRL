@echo off
REM 修复 PyTorch Geometric 依赖的 Windows 批处理脚本

echo ========================================
echo 修复 torch-scatter 和 torch-sparse
echo ========================================
echo.

REM 检查 PyTorch 版本
echo [1/4] 检查 PyTorch 版本...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 (
    echo 错误: 无法导入 PyTorch
    pause
    exit /b 1
)

echo.
echo [2/4] 卸载旧版本（如果有）...
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv -y

echo.
echo [3/4] 尝试安装预编译版本...
echo 注意: PyTorch 2.8.0 较新，可能需要从源码编译
echo.

REM 尝试安装 CUDA 版本
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
if errorlevel 1 (
    echo.
    echo 预编译版本安装失败，尝试 CPU 版本...
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
)

echo.
echo [4/4] 验证安装...
python -c "import torch_scatter; import torch_sparse; print('✓ 安装成功！')" 2>nul
if errorlevel 1 (
    echo ✗ 验证失败，可能需要使用 conda 安装
    echo.
    echo 推荐使用 conda:
    echo   conda install pyg -c pyg
) else (
    echo.
    echo ========================================
    echo 修复完成！
    echo ========================================
)

pause
