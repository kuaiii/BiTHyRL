# -*- coding: utf-8 -*-
"""
network_metrics.structural: 纯结构拓扑指标

不依赖控制器部署，仅基于网络图结构本身计算的指标。
"""

from .gcc import gcc_size, gcc_ratio, second_lcc_size, all_component_sizes
from .robustness import (
    attack_trajectory,
    calculate_auc,
    calculate_r_value_interpolated,
    calculate_r_value_simple,
)

__all__ = [
    "gcc_size",
    "gcc_ratio",
    "second_lcc_size",
    "all_component_sizes",
    "attack_trajectory",
    "calculate_auc",
    "calculate_r_value_interpolated",
    "calculate_r_value_simple",
]
