# -*- coding: utf-8 -*-
"""
Network reconstruction algorithm base class.
All topology construction / optimization algorithms should inherit from this.
"""
import networkx as nx
from typing import Optional, Dict, Any


class NetworkReconstructionAlgorithm:
    """网络重构算法基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def construct(self, G: nx.Graph, target_edges: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        重构/优化网络拓扑。

        Returns:
            dict 必须包含 'G_constructed' 键，推荐包含 'added_edges', 'improvement_auc', 'improvement_pc'
        """
        raise NotImplementedError("Subclasses must implement construct()")

    def dismantle(self, G: nx.Graph, stop_condition: str = 'threshold', stop_threshold: float = 0.5) -> Dict[str, Any]:
        """
        对给定网络执行攻击拆解并返回指标。

        Returns:
            dict 包含 attack_sequence, gcc_sizes 以及 compute_all_metrics 返回的指标
        """
        raise NotImplementedError("Subclasses must implement dismantle()")

    @staticmethod
    def run_attack(G: nx.Graph, attack_sequence, stop_condition='threshold', stop_threshold=0.5):
        """
        执行攻击序列并记录每步的GCC大小。

        Returns:
            list[int]: 每步移除后的GCC大小
        """
        gcc_sizes = []
        H = G.copy()
        n = H.number_of_nodes()
        if n == 0:
            return gcc_sizes

        initial_gcc = len(max(nx.connected_components(H), key=len)) if H.number_of_nodes() > 0 else 0
        gcc_sizes.append(initial_gcc)

        for node in attack_sequence:
            if node in H:
                H.remove_node(node)
            if H.number_of_nodes() == 0:
                gcc_sizes.append(0)
                break
            gcc = len(max(nx.connected_components(H), key=len))
            gcc_sizes.append(gcc)

            if stop_condition == 'threshold' and gcc / n < stop_threshold:
                break

        return gcc_sizes
