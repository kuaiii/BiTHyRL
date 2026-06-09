# -*- coding: utf-8 -*-
"""
network_construction: 网络重构/拓扑优化算法统一包

提供统一的拓扑重构算法接口，所有算法均以 ``networkx.Graph`` 为输入，
返回优化后的图。

示例::

    from network_construction import construct, list_algorithms
    G2 = construct(G, algorithm='Onion', max_iterations=2000)
"""

from .base import NetworkReconstructionAlgorithm
from .unified_interface import (
    construct,
    list_algorithms,
    register_algorithm,
    CONSTRUCTION_REGISTRY,
)

__all__ = [
    "NetworkReconstructionAlgorithm",
    "construct",
    "list_algorithms",
    "register_algorithm",
    "CONSTRUCTION_REGISTRY",
]
