# -*- coding: utf-8 -*-
"""
可视化模块。

包含:
  - result_plotter: 实验结果可视化
  - training_history: 训练历史可视化
  - progress: 训练进度可视化
  - plots: 网络绘图
  - plot_curve: 曲线绘图
  - plot_bar: 柱状图绘图
"""

from .training_history import (
    plot_training_history,
    plot_all_training_histories,
    plot_comparison,
)

from .result_plotter import (
    plot_curve,
    plot_curves_for_batches,
    plot_collapse_point_charts,
    plot_attack_steps_for_batches,
    plot_metric_vs_attack_steps,
    plot_bar_chart,
    plot_box_chart,
    plot_different_topo,
)

from .progress import (
    TrainingProgress,
    SimpleProgress,
    create_progress_bar,
    print_training_status,
)

__all__ = [
    # 训练历史可视化
    'plot_training_history',
    'plot_all_training_histories',
    'plot_comparison',
    # 实验结果可视化
    'plot_curve',
    'plot_curves_for_batches',
    'plot_collapse_point_charts',
    'plot_attack_steps_for_batches',
    'plot_metric_vs_attack_steps',
    'plot_bar_chart',
    'plot_box_chart',
    'plot_different_topo',
    # 进度可视化
    'TrainingProgress',
    'SimpleProgress',
    'create_progress_bar',
    'print_training_status',
]
