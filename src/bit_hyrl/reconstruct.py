# -*- coding: utf-8 -*-
"""BiT-HyRL 重构流程编排。"""
from math import ceil
from src.utils.logger import get_logger
from . import topology
from . import training
from . import reward

logger = get_logger(__name__)


def bit_hyrl_reconstruct(G, controller_rate=0.1, hub_ratio=0.15, episodes=100,
                         metric_type='robustness', seed=None, model_path=None,
                         use_node2vec=True, use_stepwise_reward=True, use_ci=True):
    """
    BiT-HyRL 完整重构：双峰拓扑 + RL 控制器选择。
    """
    logger.debug("=" * 60)
    logger.debug("开始 BiT-HyRL 算法重构")
    logger.debug("=" * 60)

    logger.debug(f"阶段1: 双峰拓扑重构 (hub_ratio={hub_ratio})")
    logger.debug(f"  原始网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    G_bimodal = topology.create_bimodal_network_exact(G, hub_ratio=hub_ratio, seed=seed)
    logger.debug(f"  重构后: {G_bimodal.number_of_nodes()} 节点, {G_bimodal.number_of_edges()} 边")

    k = ceil(G_bimodal.number_of_nodes() * controller_rate)
    logger.debug(f"  控制器数量: {k} (rate={controller_rate})")

    logger.debug(f"阶段2: RL 控制器选择 (episodes={episodes}, metric={metric_type})")
    scale = 1.4 if G_bimodal.number_of_nodes() > 100 else 1.0
    rate_f = 1.3 if controller_rate >= 0.15 else 1.0
    adj_ep = int(episodes * scale * rate_f)
    if adj_ep != episodes:
        logger.debug(f"  动态调整训练轮数: {episodes} -> {adj_ep}")

    controllers = training.train_and_select(
        G_bimodal, k, episodes=adj_ep, metric_type=metric_type,
        use_stepwise_reward=use_stepwise_reward, use_node2vec=use_node2vec, use_ci=use_ci
    )
    logger.debug(f"  选中的控制器: {sorted(controllers)}")
    r = reward.calculate_reward(G_bimodal, controllers, mode=metric_type, use_stepwise=False)
    logger.debug(f"  奖励值 ({metric_type}): {r:.4f}")
    logger.debug("=" * 60)
    logger.debug("BiT-HyRL 算法重构完成")
    logger.debug("=" * 60)
    return G_bimodal, controllers
