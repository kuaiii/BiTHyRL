# -*- coding: utf-8 -*-
"""
鲁棒性相关结构指标：攻击轨迹、AUC、R-value
"""
import numpy as np
import networkx as nx
from typing import List, Tuple


def attack_trajectory(
    G: nx.Graph,
    attack_sequence: List[int],
    stop_threshold: float = 0.1
) -> List[Tuple[int, float, int, int]]:
    """
    模拟逐节点攻击过程，返回结构状态轨迹列表。

    每一步记录：
        (已移除节点数量, 已移除节点比例, 最大LCC大小, 次大LCC大小)

    攻击在最大LCC降到初始节点数的 stop_threshold 比例时停止。

    Args:
        G: 初始网络拓扑
        attack_sequence: 节点移除顺序列表
        stop_threshold: 停止阈值，默认 0.1（最大LCC占初始节点数的 10%）

    Returns:
        List[Tuple[int, float, int, int]]: 攻击轨迹。
        首元素为初始状态 (0, 0.0, initial_gcc, second_gcc)。
    """
    n_init = G.number_of_nodes()
    if n_init == 0:
        return []

    H = G.copy()
    trajectory = []

    # 记录初始状态
    sizes = sorted([len(c) for c in nx.connected_components(H)], reverse=True)
    max_lcc = sizes[0] if sizes else 0
    second_lcc = sizes[1] if len(sizes) >= 2 else 0
    trajectory.append((0, 0.0, max_lcc, second_lcc))

    stop_size = max(1, int(np.ceil(n_init * stop_threshold)))

    for step, node in enumerate(attack_sequence, start=1):
        if node in H:
            H.remove_node(node)

        if H.number_of_nodes() == 0:
            trajectory.append((step, step / n_init, 0, 0))
            break

        sizes = sorted([len(c) for c in nx.connected_components(H)], reverse=True)
        max_lcc = sizes[0] if sizes else 0
        second_lcc = sizes[1] if len(sizes) >= 2 else 0

        trajectory.append((step, step / n_init, max_lcc, second_lcc))

        if max_lcc <= stop_size:
            break

    return trajectory


def calculate_auc(trajectory: List[Tuple[int, float, int, int]], initial_nodes: int = None) -> float:
    """
    根据攻击轨迹计算鲁棒性曲线下面积 (AUC)。

    采用梯形法对 GCC 比例 ~ 移除节点比例 曲线求面积。

    Args:
        trajectory: attack_trajectory 返回的轨迹列表
        initial_nodes: 初始节点总数。若为 None，则从轨迹首元素推断。

    Returns:
        float: AUC 值，范围约 [0, 1]。
    """
    if not trajectory:
        return 0.0

    n_init = initial_nodes if initial_nodes is not None else trajectory[0][2]
    if n_init == 0:
        return 0.0

    x = [t[1] for t in trajectory]  # 移除比例
    y = [t[2] / n_init for t in trajectory]  # GCC 比例

    if len(x) < 2:
        return 0.0

    # 补全到 x=1（若提前停止）
    if x[-1] < 1.0:
        x.append(1.0)
        y.append(0.0)

    return float(np.trapz(y, x))


def calculate_r_value_interpolated(x_curve: List[float], y_curve: List[float], num_points: int = 101) -> float:
    """
    使用统一 x 网格插值后计算 R 值（鲁棒性指标）。

    适用于不同方法采样点不均匀的情况（如级联失效导致 x 跳跃）。

    Args:
        x_curve: x 轴数据（移除节点比例）
        y_curve: y 轴数据（韧性指标值，如 GCC 比例）
        num_points: 统一 x 网格采样点数，默认 101

    Returns:
        float: R 值（归一化后的曲线下面积）
    """
    if len(x_curve) < 2 or len(y_curve) < 2:
        return 0.0

    x_arr = np.array(x_curve, dtype=np.float64)
    y_arr = np.array(y_curve, dtype=np.float64)

    # 去重，保持单调递增
    unique_indices = []
    seen = set()
    for i, xv in enumerate(x_arr):
        if xv not in seen:
            unique_indices.append(i)
            seen.add(xv)

    if len(unique_indices) < 2:
        return 0.0

    x_arr = x_arr[unique_indices]
    y_arr = y_arr[unique_indices]

    # 补全边界
    if x_arr[0] > 0:
        x_arr = np.concatenate([[0.0], x_arr])
        y_arr = np.concatenate([[y_arr[0]], y_arr])
    if x_arr[-1] < 1.0:
        last_y = y_arr[-1] if y_arr[-1] > 1e-10 else 0.0
        x_arr = np.concatenate([x_arr, [1.0]])
        y_arr = np.concatenate([y_arr, [last_y]])

    x_uniform = np.linspace(0.0, 1.0, num_points)
    y_interp = np.interp(x_uniform, x_arr, y_arr)
    return float(np.trapz(y_interp, x_uniform))


def calculate_r_value_simple(x_curve: List[float], y_curve: List[float]) -> float:
    """
    简单 R 值计算：直接使用梯形法则。

    注意：x 采样点不均匀时可能不够准确，建议使用 calculate_r_value_interpolated。

    Args:
        x_curve: x 轴数据
        y_curve: y 轴数据

    Returns:
        float: R 值
    """
    if len(x_curve) < 2:
        return 0.0
    return float(np.trapz(y_curve, x_curve))
