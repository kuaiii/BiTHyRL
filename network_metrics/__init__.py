# -*- coding: utf-8 -*-
"""
network_metrics: 网络结构评价指标统一包

包含两大类别：
- structural: 纯结构拓扑指标（不依赖控制器部署）
- functional: 功能指标（依赖控制器部署）

入口函数：
    from network_metrics import compute, list_metrics
"""

from .unified_interface import compute, list_metrics, register_metric

__all__ = ["compute", "list_metrics", "register_metric"]
