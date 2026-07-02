# -*- coding: utf-8 -*-
"""
统一奖励函数：韧性 + 分散性 - 重构成本 - 双峰偏离惩罚 + 拓扑结构奖励。

v5 改进：
- 增加对高 edge_betweenness / 高度方差的惩罚
- 增加对高代数连通度的奖励
- 课程学习优先 betweenness / pagerank 攻击
- 降低 bimodal 惩罚的默认权重
"""
import numpy as np
import networkx as nx
from scipy import stats

from src.bit_hyrl.reward import (
    _simulate_attack_sequence,
    _get_dispersion_score,
    _get_adaptive_aggregation,
    _get_curriculum_methods,
    ADVERSARIAL_ATTACK_METHODS,
)
from network_dismantling.unified_interface import dismantle


# 全局攻击序列缓存（按图哈希+方法）
_ATTACK_SEQ_CACHE = {}
_ATTACK_SEQ_CACHE_MAX_SIZE = 5000

# 拓扑结构分数缓存
_TOPO_SCORE_CACHE = {}
_TOPO_SCORE_CACHE_MAX_SIZE = 2000


def _get_topo_structure_score_cached(G_final, G_original):
    """带缓存的拓扑结构分数计算。"""
    global _TOPO_SCORE_CACHE
    h = _graph_hash(G_final)
    if h in _TOPO_SCORE_CACHE:
        return _TOPO_SCORE_CACHE[h]
    scores = _topology_structure_score(G_final, G_original)
    if len(_TOPO_SCORE_CACHE) >= _TOPO_SCORE_CACHE_MAX_SIZE:
        keys = list(_TOPO_SCORE_CACHE.keys())
        for k in keys[:_TOPO_SCORE_CACHE_MAX_SIZE // 2]:
            del _TOPO_SCORE_CACHE[k]
    _TOPO_SCORE_CACHE[h] = scores
    return scores


def _graph_hash(G):
    """基于边集合的简单图哈希。"""
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in G.edges()))
    return hash(edges)


def _get_attack_sequence(G, method):
    """获取单种攻击方法的攻击序列，带 LRU 缓存。"""
    global _ATTACK_SEQ_CACHE
    h = (_graph_hash(G), method)
    if h in _ATTACK_SEQ_CACHE:
        return _ATTACK_SEQ_CACHE[h]
    try:
        seq = dismantle(G, method)
        if seq is None or len(seq) == 0:
            return None
        if len(_ATTACK_SEQ_CACHE) >= _ATTACK_SEQ_CACHE_MAX_SIZE:
            # 简单清空一半缓存避免无限增长
            keys = list(_ATTACK_SEQ_CACHE.keys())
            for k in keys[:_ATTACK_SEQ_CACHE_MAX_SIZE // 2]:
                del _ATTACK_SEQ_CACHE[k]
        _ATTACK_SEQ_CACHE[h] = seq
        return seq
    except Exception:
        return None


def _topology_structure_score(G_final, G_original):
    """
    计算拓扑结构奖励分量，返回 dict。
    关键观察（来自 topology_comparison_v4 报告）：
    - edge_betweenness_max 与 Robustness 强负相关（r=-0.74）
    - algebraic_connectivity 与 Robustness 强正相关（r=+0.70）
    - degree_std / degree_skew 越低，GA/QDLM 拓扑越鲁棒
    """
    scores = {}
    n = G_final.number_of_nodes()
    if n <= 1:
        return scores

    # 1. 边 betweenness 集中度（越低越好）
    # 使用固定小样本近似，避免大图过慢
    try:
        n = G_final.number_of_nodes()
        k = min(n, max(20, int(n * 0.05)))
        bc = nx.edge_betweenness_centrality(G_final, k=k)
        bc_vals = np.array(list(bc.values()))
        scores['edge_betweenness_max'] = float(bc_vals.max()) if len(bc_vals) > 0 else 1.0
        scores['edge_betweenness_std'] = float(bc_vals.std()) if len(bc_vals) > 0 else 0.0
    except Exception:
        scores['edge_betweenness_max'] = 1.0
        scores['edge_betweenness_std'] = 0.0

    # 2. 度分布均匀性（越低越好）
    degrees = np.array([d for _, d in G_final.degree()])
    if len(degrees) > 0:
        scores['degree_std'] = float(degrees.std())
        scores['degree_skew'] = float(stats.skew(degrees)) if len(degrees) > 2 else 0.0
    else:
        scores['degree_std'] = 0.0
        scores['degree_skew'] = 0.0

    # 3. 代数连通度（越高越好），对原图归一化
    try:
        if nx.is_connected(G_final):
            L = nx.laplacian_matrix(G_final).astype(np.float32)
            eigvals = np.linalg.eigvalsh(L.toarray())
            scores['algebraic_connectivity'] = float(sorted(eigvals)[1]) if len(eigvals) > 1 else 0.0
        else:
            scores['algebraic_connectivity'] = 0.0
    except Exception:
        scores['algebraic_connectivity'] = 0.0

    # 4. 与原始图的密度相对变化（适度增加密度好，但避免过大）
    orig_density = 2 * G_original.number_of_edges() / (n * (n - 1)) if n > 1 else 0.0
    final_density = 2 * G_final.number_of_edges() / (n * (n - 1)) if n > 1 else 0.0
    scores['density_ratio'] = final_density / (orig_density + 1e-9)

    return scores


def _curriculum_methods_v5(epoch, total_epochs, attack_methods, focus_methods=None):
    """
    课程学习：早期重点训练 betweenness / pagerank 等薄弱攻击，后期均匀采样。
    """
    if focus_methods is None:
        focus_methods = ['betweenness', 'pagerank']
    if epoch is None or total_epochs is None or total_epochs <= 1:
        return attack_methods

    progress = epoch / (total_epochs - 1)
    if progress < 0.4:
        # 前 40%  epoch：优先 focus_methods
        pool = [m for m in attack_methods if m in focus_methods]
        if not pool:
            pool = attack_methods
        # 40% 概率只取 focus，60% 概率扩展全部
        if np.random.random() < 0.6:
            return pool
        return attack_methods
    elif progress < 0.7:
        # 中期：focus_methods 权重加倍
        weighted = list(attack_methods)
        for m in focus_methods:
            if m in weighted:
                weighted.append(m)
        return weighted
    else:
        return attack_methods


def calculate_unified_reward(
    G_final,
    controllers,
    G_original,
    attack_ratio=0.15,
    attack_methods=None,
    aggregation='min',
    w_gcc=0.6,
    w_disp=0.2,
    w_cost=0.2,
    alpha=0.01,
    target_hub_ratio=0.15,
    bimodal_weight=0.05,
    sample_methods=None,
    epoch=None,
    total_epochs=None,
    use_curriculum=False,
    use_adaptive_aggregation=False,
    # v5 新增参数
    use_topology_reward=False,
    w_topo=0.15,
    w_edge_betweenness_max=0.08,
    w_edge_betweenness_std=0.04,
    w_degree_std=0.03,
    w_algebraic_connectivity=0.08,
    focus_attacks=None,
):
    """
    计算统一奖励。

    R = w_gcc * GCC_score + w_disp * dispersion_score + w_topo * topo_reward
        - w_cost * cost_penalty - bimodal_weight * bimodal_penalty

    其中 topo_reward 鼓励：
    - 低 edge_betweenness_max / std
    - 低 degree_std
    - 高 algebraic_connectivity

    Args:
        ...（原有参数）
        use_topology_reward: 是否启用拓扑结构奖励
        w_topo: 拓扑奖励总权重
        w_edge_betweenness_max: 最大边 betweenness 惩罚权重
        w_edge_betweenness_std: 边 betweenness 标准差惩罚权重
        w_degree_std: 度标准差惩罚权重
        w_algebraic_connectivity: 代数连通度奖励权重
        focus_attacks: 课程学习重点攻击方法列表

    Returns:
        float: 奖励值
    """
    if not controllers or G_final.number_of_nodes() == 0:
        return 0.0

    if attack_methods is None:
        attack_methods = ADVERSARIAL_ATTACK_METHODS

    # 课程 / 自适应聚合
    if use_curriculum:
        # v5：使用更聚焦的课程
        attack_methods = _curriculum_methods_v5(epoch, total_epochs, attack_methods, focus_methods=focus_attacks)
    if use_adaptive_aggregation:
        aggregation = _get_adaptive_aggregation(epoch, total_epochs, aggregation)

    if sample_methods and sample_methods < len(attack_methods):
        import random
        attack_methods = random.sample(attack_methods, sample_methods)

    # GCC 项（使用缓存）
    gcc_scores = []
    for method in attack_methods:
        seq = _get_attack_sequence(G_final, method)
        if seq is None or len(seq) == 0:
            continue
        try:
            retention = _simulate_attack_sequence(G_final, controllers, seq, attack_ratio)
            gcc_scores.append(retention)
        except Exception:
            continue

    if len(gcc_scores) == 0:
        gcc_reward = 0.0
    elif aggregation == 'min':
        gcc_reward = min(gcc_scores)
    elif aggregation == 'mean':
        gcc_reward = sum(gcc_scores) / len(gcc_scores)
    elif aggregation == 'max':
        gcc_reward = max(gcc_scores)
    else:
        gcc_reward = min(gcc_scores)

    # 分散性项
    disp_reward = _get_dispersion_score(G_final, controllers)

    # 重构成本：边差异
    E_orig = set(tuple(sorted((u, v))) for u, v in G_original.edges())
    E_final = set(tuple(sorted((u, v))) for u, v in G_final.edges())
    edge_diff = len(E_orig.symmetric_difference(E_final))
    max_diff = max(1, int(0.3 * G_original.number_of_edges()))
    cost_penalty = alpha * (edge_diff / max_diff)

    # 双峰偏离惩罚（v5 降低默认权重）
    if G_final.number_of_nodes() > 0:
        degrees = dict(G_final.degree())
        sorted_degrees = sorted(degrees.values(), reverse=True)
        hub_count = max(1, int(G_final.number_of_nodes() * target_hub_ratio))
        hub_threshold = sorted_degrees[hub_count - 1] if hub_count <= len(sorted_degrees) else sorted_degrees[-1]
        hub_nodes = [n for n, d in degrees.items() if d >= hub_threshold]
        hub_ratio_actual = len(hub_nodes) / G_final.number_of_nodes()
    else:
        hub_ratio_actual = 0.0
    bimodal_penalty = abs(hub_ratio_actual - target_hub_ratio)

    reward = (w_gcc * gcc_reward
              + w_disp * disp_reward
              - w_cost * cost_penalty
              - bimodal_weight * bimodal_penalty)

    # v5：拓扑结构奖励
    if use_topology_reward:
        topo_scores = _get_topo_structure_score_cached(G_final, G_original)
        if topo_scores:
            n = G_final.number_of_nodes()
            # 归一化参考值（来自观察：好的拓扑 edge_betweenness_max ~ 0.05，degree_std ~ 2）
            ref_edge_betweenness_max = 0.05
            ref_degree_std = max(1.0, 2.0)
            ref_algebraic_conn = max(0.1, np.log1p(n) / 10)

            topo_reward = 0.0
            if 'edge_betweenness_max' in topo_scores:
                eb_max = topo_scores['edge_betweenness_max']
                topo_reward -= w_edge_betweenness_max * (eb_max / ref_edge_betweenness_max)
            if 'edge_betweenness_std' in topo_scores:
                eb_std = topo_scores['edge_betweenness_std']
                topo_reward -= w_edge_betweenness_std * (eb_std / ref_edge_betweenness_max)
            if 'degree_std' in topo_scores:
                d_std = topo_scores['degree_std']
                topo_reward -= w_degree_std * (d_std / ref_degree_std)
            if 'algebraic_connectivity' in topo_scores:
                ac = topo_scores['algebraic_connectivity']
                topo_reward += w_algebraic_connectivity * min(ac / ref_algebraic_conn, 3.0)

            reward += w_topo * topo_reward

    return reward
