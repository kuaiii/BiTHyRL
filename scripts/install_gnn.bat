@echo off
REM 安装 GNN 依赖包的批处理脚本

echo ========================================
echo 安装 PyTorch Geometric 依赖包
echo ========================================
echo.

echo [方法 1] 尝试使用 Conda 安装（推荐）...
conda install pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg -y
if %errorlevel% == 0 (
    echo.
    echo ✓ Conda 安装成功！
    goto :verify
)

echo.
echo [方法 2] Conda 安装失败，尝试使用 pip...
echo 注意: 可能需要解决网络或 SSL 问题
echo.

REM 尝试使用 pip 安装（可能需要配置代理或使用其他源）
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
if %errorlevel% == 0 (
    echo.
    echo ✓ pip 安装成功！
    goto :verify
)

echo.
echo [方法 3] 尝试安装 CPU 版本（用于测试）...
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
if %errorlevel% == 0 (
    echo.
    echo ✓ CPU 版本安装成功（注意：将使用 CPU 而非 GPU）！
    goto :verify
)

echo.
echo ✗ 所有自动安装方法都失败了
echo.
echo 请手动尝试以下方法：
echo 1. 在 Anaconda Prompt 中运行: conda install pyg -c pyg
echo 2. 或者降级 PyTorch 到 2.5.0 后重新安装
echo 3. 查看 install_gnn_dependencies.md 获取详细说明
goto :end

:verify
echo.
echo ========================================
echo 验证安装...
echo ========================================
python -c "import torch_scatter; import torch_sparse; print('✓ 导入成功！')"
if %errorlevel% == 0 (
    echo.
    echo ========================================
    echo 安装完成！现在可以使用 --use_gnn 参数了
    echo ========================================
) else (
    echo.
    echo ✗ 验证失败，包可能无法正常工作
    echo 请检查错误信息或尝试其他安装方法
)

:end
pause
