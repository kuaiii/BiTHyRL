# -*- coding: utf-8 -*-
"""
network_metrics unified interface
=================================
统一的网络指标计算接口。所有指标均以 ``networkx.Graph`` 为核心输入。

Usage
-----
>>> from network_metrics import compute, list_metrics
>>> gcc_val = compute('gcc_size', G)
>>> traj = compute('attack_trajectory', G, attack_sequence=[...])
>>> auc = compute('auc', trajectory=traj, initial_nodes=100)
"""
import logging
from typing import Dict, Callable, Any, List
import networkx as nx

from .structural import gcc, robustness
from .functional import network as fnetwork, controller

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_METRIC_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_metric(name: str):
    """装饰器：注册指标计算函数。"""
    def decorator(func: Callable):
        _METRIC_REGISTRY[name] = func
        return func
    return decorator


def list_metrics() -> List[str]:
    """返回所有已注册指标名称列表。"""
    return list(_METRIC_REGISTRY.keys())


def compute(metric_name: str, *args, **kwargs) -> Any:
    """
    统一指标计算入口。

    Args:
        metric_name: 指标名称
        *args: 位置参数（通常第一个为网络拓扑 G）
        **kwargs: 指标所需的额外参数（如 attack_sequence、centers 等）

    Returns:
        指标计算结果（类型取决于具体指标）
    """
    if metric_name not in _METRIC_REGISTRY:
        raise ValueError(
            f"未知指标 '{metric_name}'。可用指标: {list_metrics()}"
        )
    return _METRIC_REGISTRY[metric_name](*args, **kwargs)


# ---------------------------------------------------------------------------
# 结构指标 (pure topology)
# ---------------------------------------------------------------------------
@register_metric("gcc_size")
def _gcc_size(G: nx.Graph) -> int:
    return gcc.gcc_size(G)


@register_metric("gcc_ratio")
def _gcc_ratio(G: nx.Graph, initial_nodes: int = None) -> float:
    return gcc.gcc_ratio(G, initial_nodes=initial_nodes)


@register_metric("second_lcc")
def _second_lcc(G: nx.Graph) -> int:
    return gcc.second_lcc_size(G)


@register_metric("all_component_sizes")
def _all_component_sizes(G: nx.Graph) -> List[int]:
    return gcc.all_component_sizes(G)


@register_metric("gcc_snapshot")
def _gcc_snapshot(G: nx.Graph, initial_nodes: int = None) -> tuple:
    """
    一次性返回 GCC 及其占比、次大LCC 及其占比。

    Returns:
        Tuple[int, float, int, float]: (LCC大小, LCC占比, 次大LCC大小, 次大LCC占比)
    """
    n = initial_nodes if initial_nodes is not None else G.number_of_nodes()
    lcc = gcc.gcc_size(G)
    slcc = gcc.second_lcc_size(G)
    lcc_ratio = lcc / n if n > 0 else 0.0
    slcc_ratio = slcc / n if n > 0 else 0.0
    return (lcc, lcc_ratio, slcc, slcc_ratio)


@register_metric("attack_trajectory")
def _attack_trajectory(
    G: nx.Graph,
    attack_sequence: List[int],
    stop_threshold: float = 0.1
) -> List[tuple]:
    return robustness.attack_trajectory(G, attack_sequence, stop_threshold)


@register_metric("auc")
def _auc(
    G: nx.Graph = None,
    *,
    trajectory: List[tuple] = None,
    initial_nodes: int = None
) -> float:
    """
    计算 AUC。可通过传入 trajectory 直接计算，也可自动从 G 推断 initial_nodes。
    """
    if trajectory is None:
        raise ValueError("auc 指标需要提供 trajectory 参数")
    return robustness.calculate_auc(trajectory, initial_nodes=initial_nodes)


@register_metric("r_value")
@register_metric("r_value_interpolated")
def _r_value(*args, num_points: int = 101) -> float:
    """兼容 compute('r_value_interpolated', x_list, y_list, num_points=101)。"""
    if len(args) < 2:
        raise TypeError("r_value_interpolated 需要至少 x_curve 和 y_curve 两个参数")
    x_curve, y_curve = args[-2], args[-1]
    return robustness.calculate_r_value_interpolated(x_curve, y_curve, num_points)


@register_metric("r_value_simple")
def _r_value_simple(*args) -> float:
    if len(args) < 2:
        raise TypeError("r_value_simple 需要至少 x_curve 和 y_curve 两个参数")
    x_curve, y_curve = args[-2], args[-1]
    return robustness.calculate_r_value_simple(x_curve, y_curve)


@register_metric("attack_robustness")
def _attack_robustness(
    G: nx.Graph,
    attack_sequence: List[int],
    stop_threshold: float = 0.1,
    initial_nodes: int = None
) -> dict:
    """
    一次性计算攻击轨迹和 AUC，避免重复计算。

    Returns:
        dict: {
            'trajectory': List[tuple],  # 攻击轨迹
            'auc': float,               # 曲线下面积
            'initial_nodes': int,       # 初始节点数
            'final_step': int,          # 最后一步的移除数量
        }
    """
    n = initial_nodes if initial_nodes is not None else G.number_of_nodes()
    traj = robustness.attack_trajectory(G, attack_sequence, stop_threshold)
    auc_val = robustness.calculate_auc(traj, initial_nodes=n)
    return {
        'trajectory': traj,
        'auc': auc_val,
        'initial_nodes': n,
        'final_step': traj[-1][0] if traj else 0,
    }


# ---------------------------------------------------------------------------
# Functional metrics (require controllers)
# ---------------------------------------------------------------------------
@register_metric("gcc_with_controllers")
def _gcc_with_controllers(G: nx.Graph, centers: List[int]) -> int:
    """计算包含控制器的最大连通分量大小（与原 src.metrics.metrics.calculate_gcc 等价）。"""
    return fnetwork.max_component_size_compute(G, centers)


@register_metric("coverage")
def _coverage(G: nx.Graph, centers: List[int]) -> float:
    return fnetwork.calculate_coverage(G, centers)


@register_metric("efficiency")
def _efficiency(G: nx.Graph, centers: List[int]) -> float:
    return fnetwork.calculate_efficiency(G, centers)


@register_metric("max_controlled_component")
def _max_controlled_component(G: nx.Graph, centers: List[int]) -> int:
    return fnetwork.max_component_size_compute(G, centers)


@register_metric("csa")
def _csa(
    G: nx.Graph,
    centers: List[int],
    alpha: float = 0.2,
    initial_num_switches: int = None
) -> float:
    return controller.calculate_csa(G, centers, alpha=alpha, initial_num_switches=initial_num_switches)


@register_metric("cce")
def _cce(
    G: nx.Graph,
    centers: List[int],
    initial_sum_A: float = None
) -> float:
    return controller.calculate_cce(G, centers, initial_sum_A=initial_sum_A)


@register_metric("wcp")
def _wcp(
    G: nx.Graph,
    centers: List[int],
    beta: float = 1.0,
    initial_num_switches: int = None
) -> float:
    return controller.calculate_wcp(G, centers, beta=beta, initial_num_switches=initial_num_switches)


@register_metric("cce_trajectory")
def _cce_trajectory(
    G: nx.Graph,
    attack_sequence: List[int],
    centers: List[int],
    stop_threshold: float = 0.1
) -> List[tuple]:
    return controller.simulate_cce_trajectory(G, attack_sequence, centers, stop_threshold)


@register_metric("csa_robustness")
def _csa_robustness(
    G: nx.Graph,
    attack_sequence: List[int],
    centers: List[int],
    stop_threshold: float = 0.1,
    alpha: float = 0.2
) -> dict:
    """一次性返回 CSA 攻击轨迹和 CSA_AUC。"""
    return controller.simulate_csa_trajectory(G, attack_sequence, centers, stop_threshold, alpha)


@register_metric("cce_robustness")
def _cce_robustness(
    G: nx.Graph,
    attack_sequence: List[int],
    centers: List[int],
    stop_threshold: float = 0.1
) -> dict:
    """一次性返回 CCE 攻击轨迹和 CCE_AUC。"""
    return controller.simulate_cce_trajectory_with_auc(G, attack_sequence, centers, stop_threshold)


@register_metric("wcp_robustness")
def _wcp_robustness(
    G: nx.Graph,
    attack_sequence: List[int],
    centers: List[int],
    stop_threshold: float = 0.1,
    beta: float = 1.0
) -> dict:
    """一次性返回 WCP 攻击轨迹和 WCP_AUC。"""
    return controller.simulate_wcp_trajectory(G, attack_sequence, centers, stop_threshold, beta)
