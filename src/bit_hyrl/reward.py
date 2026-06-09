# -*- coding: utf-8 -*-
"""BiT-HyRL 奖励函数（Step-wise + 端到端 + 自适应）。"""
import networkx as nx
import numpy as np
from network_metrics.functional.network import max_component_size_compute as calculate_gcc
from network_metrics.functional.controller import calculate_csa, calculate_cce as calculate_control_entropy, calculate_wcp

# 网络规模阈值
SMALL_NETWORK_THRESHOLD = 50
MEDIUM_NETWORK_THRESHOLD = 200


def calculate_stepwise_reward(G, previous_centers, new_center, mode='combined', w1=2.0, w2=1.0, w3=0.5):
    """Step-wise 奖励: R_t = w1 * ΔGCC + w2 * ΔCoverage - w3 * Cost_path。"""
    if new_center is None or not G.has_node(new_center):
        return 0.0
    centers_before = set(previous_centers)
    centers_after = centers_before | {new_center}
    gcc_before = calculate_gcc(G, list(centers_before))
    gcc_after = calculate_gcc(G, list(centers_after))
    delta_gcc = (gcc_after - gcc_before) / max(G.number_of_nodes(), 1)
    components = list(nx.connected_components(G))
    controlled_before = set()
    for comp in components:
        if not centers_before.isdisjoint(comp):
            controlled_before.update(comp)
    controlled_after = set()
    for comp in components:
        if not centers_after.isdisjoint(comp):
            controlled_after.update(comp)
    delta_coverage = len(controlled_after - controlled_before) / max(G.number_of_nodes(), 1)
    cost_path = 0.0
    if centers_before:
        min_dists = []
        for c in centers_before:
            try:
                if nx.has_path(G, new_center, c):
                    min_dists.append(nx.shortest_path_length(G, new_center, c))
            except Exception:
                pass
        if min_dists:
            cost_path = min(min_dists) / max(G.number_of_nodes(), 1)
        else:
            cost_path = 1.0
    return w1 * delta_gcc + w2 * delta_coverage - w3 * cost_path


def calculate_reward(G, centers, mode='combined', use_stepwise=False, previous_centers=None, new_center=None):
    """综合奖励。use_stepwise 时委托 calculate_stepwise_reward。"""
    if use_stepwise and previous_centers is not None and new_center is not None:
        return calculate_stepwise_reward(G, previous_centers, new_center, mode=mode)
    if not centers:
        return 0.0
    centers_set = set(centers)
    total_nodes = G.number_of_nodes()
    r_val = 0.0
    if mode in ['combined', 'robustness']:
        G_copy = G.copy()
        num_remove = max(1, int(total_nodes * 0.1))
        degrees = dict(G_copy.degree())
        targets = sorted(degrees, key=degrees.get, reverse=True)[:num_remove]
        remaining = centers_set.copy()
        for node in targets:
            if node in G_copy:
                G_copy.remove_node(node)
                remaining.discard(node)
        if G_copy.number_of_nodes() > 0 and remaining:
            comps = list(nx.connected_components(G_copy))
            controlled = sum(len(c) for c in comps if not remaining.isdisjoint(c))
            r_val = controlled / total_nodes
    csa_val = calculate_csa(G, centers) if mode in ['combined', 'csa'] else 0.0
    hc_val = calculate_control_entropy(G, centers) if mode in ['combined', 'entropy'] else 0.0
    wcp_val = calculate_wcp(G, centers) if mode in ['combined', 'wcp'] else 0.0
    components = list(nx.connected_components(G))
    coverage = sum(1 for c in components if not centers_set.isdisjoint(c)) / max(1, len(components)) if mode in ['combined', 'robustness'] else 1.0
    dispersion = 1.0
    if mode in ['combined', 'robustness'] and len(centers) > 1:
        try:
            min_dist = float('inf')
            for i, c1 in enumerate(centers):
                for c2 in centers[i + 1:]:
                    if G.has_node(c1) and G.has_node(c2) and nx.has_path(G, c1, c2):
                        min_dist = min(min_dist, nx.shortest_path_length(G, c1, c2))
            dispersion = min(min_dist / 3.0, 1.0) if min_dist != float('inf') else 0.5
        except Exception:
            dispersion = 0.5
    if mode == 'robustness':
        return r_val * 0.6 + coverage * 0.25 + dispersion * 0.15
    if mode == 'csa':
        return csa_val
    if mode == 'entropy':
        return hc_val
    if mode == 'wcp':
        return wcp_val
    if mode == 'gcc':
        # 纯 GCC 模式，使用新的 GCC 专注奖励
        return calculate_gcc_focused_reward(G, centers)
    if mode == 'survival':
        # 生存能力模式（推荐）：优化逐步攻击过程中的累积生存能力
        return calculate_survival_reward(G, centers)
    if mode == 'integral':
        # R值积分模式：直接优化曲线下面积
        return calculate_robustness_integral_reward(G, centers)
    return 2.0 * r_val + 1.0 * csa_val + 0.2 * hc_val + 1.0 * wcp_val + 0.5 * coverage + 0.3 * dispersion


def calculate_gcc_focused_reward(G, centers, attack_ratio=0.1, w_gcc=0.7, w_disp=0.3):
    """
    GCC 专注的奖励函数
    
    专门优化攻击后的 GCC 保持率，同时鼓励控制器分散部署。
    
    奖励计算:
    R = w_gcc * GCC_retention + w_disp * dispersion_score
    
    - GCC_retention: 攻击后 GCC 占原始节点数的比例
    - dispersion_score: 控制器分散程度（平均距离归一化）
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        attack_ratio: 攻击比例（移除多少比例的高度数节点）
        w_gcc: GCC 保持率权重
        w_disp: 分散性权重
        
    Returns:
        float: 奖励值
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    total_nodes = G.number_of_nodes()
    
    if total_nodes == 0:
        return 0.0
    
    # ===== 1. 计算攻击后的 GCC 保持率 =====
    G_attacked = G.copy()
    
    # 模拟 degree attack（移除高度数节点）
    num_remove = max(1, int(total_nodes * attack_ratio))
    degrees = dict(G_attacked.degree())
    targets = sorted(degrees, key=degrees.get, reverse=True)[:num_remove]
    
    # 记录被攻击后剩余的控制器
    remaining_centers = centers_set.copy()
    for node in targets:
        if node in G_attacked:
            G_attacked.remove_node(node)
            remaining_centers.discard(node)
    
    # 计算攻击后包含控制器的最大连通分量
    gcc_retention = 0.0
    if G_attacked.number_of_nodes() > 0 and remaining_centers:
        components = list(nx.connected_components(G_attacked))
        
        # 找出包含控制器的最大连通分量
        controlled_sizes = []
        for comp in components:
            if not remaining_centers.isdisjoint(comp):
                controlled_sizes.append(len(comp))
        
        if controlled_sizes:
            # GCC 保持率 = 最大控制分量大小 / 原始节点数
            gcc_retention = max(controlled_sizes) / total_nodes
    
    # ===== 2. 计算控制器分散性 =====
    dispersion_score = 0.5  # 默认值
    
    if len(centers) > 1:
        # 计算控制器之间的平均最短路径距离
        distances = []
        centers_list = list(centers)
        
        for i, c1 in enumerate(centers_list):
            for c2 in centers_list[i + 1:]:
                if G.has_node(c1) and G.has_node(c2):
                    try:
                        if nx.has_path(G, c1, c2):
                            d = nx.shortest_path_length(G, c1, c2)
                            distances.append(d)
                    except Exception:
                        pass
        
        if distances:
            avg_dist = sum(distances) / len(distances)
            # 归一化：假设理想距离是图直径的一半
            try:
                diameter = nx.diameter(G) if nx.is_connected(G) else total_nodes // 2
            except:
                diameter = total_nodes // 2
            
            # 分散性得分：距离越大越好，但也不能太大
            # 使用 sigmoid 函数将平均距离映射到 [0, 1]
            ideal_dist = diameter / 3.0
            dispersion_score = min(avg_dist / ideal_dist, 1.0)
    else:
        # 只有一个控制器，分散性设为中等
        dispersion_score = 0.5
    
    # ===== 3. 计算总奖励 =====
    reward = w_gcc * gcc_retention + w_disp * dispersion_score
    
    return reward


def calculate_gcc_stepwise_reward(G, previous_centers, new_center, attack_ratio=0.1):
    """
    GCC 专注的 Step-wise 奖励函数
    
    计算添加新控制器后 GCC 保持率的增量。
    
    Args:
        G: NetworkX 图
        previous_centers: 之前已选的控制器
        new_center: 新选择的控制器
        attack_ratio: 攻击比例
        
    Returns:
        float: 增量奖励
    """
    if new_center is None or not G.has_node(new_center):
        return 0.0
    
    centers_before = list(previous_centers) if previous_centers else []
    centers_after = centers_before + [new_center]
    
    # 计算前后的 GCC 专注奖励
    reward_before = calculate_gcc_focused_reward(G, centers_before, attack_ratio) if centers_before else 0.0
    reward_after = calculate_gcc_focused_reward(G, centers_after, attack_ratio)
    
    # 增量奖励
    delta_reward = reward_after - reward_before
    
    # 添加一个小的存活奖励，避免稀疏奖励问题
    survival_bonus = 0.01
    
    return delta_reward + survival_bonus


def calculate_multi_attack_reward(G, centers, attack_ratios=[0.05, 0.1, 0.15, 0.2]):
    """
    多轮攻击奖励函数
    
    模拟多个攻击比例，计算平均 GCC 保持率。
    这能更好地评估控制器配置的整体鲁棒性。
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        attack_ratios: 攻击比例列表
        
    Returns:
        float: 平均奖励值
    """
    if not centers:
        return 0.0
    
    rewards = []
    for ratio in attack_ratios:
        r = calculate_gcc_focused_reward(G, centers, attack_ratio=ratio, w_gcc=1.0, w_disp=0.0)
        rewards.append(r)
    
    # 平均奖励
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    
    # 加入分散性奖励（只计算一次）
    disp_reward = calculate_gcc_focused_reward(G, centers, w_gcc=0.0, w_disp=1.0)
    
    # 组合：80% 平均 GCC + 20% 分散性
    return 0.8 * avg_reward + 0.2 * disp_reward


# ---------------------------------------------------------------------------
# 对抗式多攻击奖励函数（使用 network_dismantling 统一接口）
# ---------------------------------------------------------------------------

# 默认攻击方法池（扩展版：包含 network_dismantling 中可用的主要方法）
# 训练时会自动过滤不可用的方法，因此列出全部候选是安全的
ADVERSARIAL_ATTACK_METHODS = [
    'degree', 'betweenness', 'pagerank', 'eigenvector', 'random',
    'CI_L1', 'CI_L2', 'CoreHD', 'GND', 'EGND',
    'EI_s1', 'GDM'
]

# 攻击方法按计算成本和攻击强度分组，用于课程式采样
_ATTACK_METHOD_GROUPS = {
    'easy': ['degree', 'random', 'pagerank'],
    'medium': ['betweenness', 'CI_L1', 'CI_L2', 'CoreHD'],
    'hard': ['GND', 'EGND', 'EI_s1', 'GDM', 'eigenvector'],
}

# ---------------------------------------------------------------------------
# Epoch 跟踪器（用于在 collect_trajectory 中传递当前 epoch 信息）
# ---------------------------------------------------------------------------
class _EpochTracker:
    epoch = 0
    total_epochs = 1

_epoch_tracker = _EpochTracker()


def set_current_epoch(epoch, total_epochs):
    """由训练循环调用，设置当前 epoch 信息。"""
    _epoch_tracker.epoch = epoch
    _epoch_tracker.total_epochs = max(1, total_epochs)

# 模块级共享缓存：graph id -> {method: sequence}
_ADVERSARIAL_SEQUENCE_CACHE = {}
_MAX_CACHE_SIZE = 5000


def _get_cached_sequence(G, method):
    """从共享缓存获取攻击序列。"""
    gid = id(G)
    return _ADVERSARIAL_SEQUENCE_CACHE.get(gid, {}).get(method)


def _set_cached_sequence(G, method, sequence):
    """将攻击序列存入共享缓存，带 LRU 淘汰。"""
    gid = id(G)
    if gid not in _ADVERSARIAL_SEQUENCE_CACHE:
        # 简单 LRU：缓存满时清空一半
        if len(_ADVERSARIAL_SEQUENCE_CACHE) >= _MAX_CACHE_SIZE:
            keys = list(_ADVERSARIAL_SEQUENCE_CACHE.keys())
            for k in keys[:len(keys) // 2]:
                del _ADVERSARIAL_SEQUENCE_CACHE[k]
        _ADVERSARIAL_SEQUENCE_CACHE[gid] = {}
    _ADVERSARIAL_SEQUENCE_CACHE[gid][method] = sequence


def _simulate_attack_sequence(G, centers, attack_sequence, attack_ratio=0.1):
    """
    根据给定的攻击序列模拟攻击，返回 GCC 保持率。
    
    Args:
        G: NetworkX 图
        centers: 控制器节点集合
        attack_sequence: 节点移除序列
        attack_ratio: 攻击比例
        
    Returns:
        float: GCC 保持率
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        return 0.0
    
    G_attacked = G.copy()
    num_remove = max(1, int(total_nodes * attack_ratio))
    remaining_centers = centers_set.copy()
    
    for node in attack_sequence[:num_remove]:
        if node in G_attacked:
            G_attacked.remove_node(node)
            remaining_centers.discard(node)
    
    if G_attacked.number_of_nodes() == 0 or not remaining_centers:
        return 0.0
    
    components = list(nx.connected_components(G_attacked))
    controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
    return max(controlled_sizes) / total_nodes if controlled_sizes else 0.0


def _get_dispersion_score(G, centers):
    """计算控制器分散性得分（复用 calculate_gcc_focused_reward 中的逻辑）。"""
    if len(centers) <= 1:
        return 0.5
    
    total_nodes = G.number_of_nodes()
    distances = []
    centers_list = list(centers)
    
    for i, c1 in enumerate(centers_list):
        for c2 in centers_list[i + 1:]:
            if G.has_node(c1) and G.has_node(c2):
                try:
                    if nx.has_path(G, c1, c2):
                        distances.append(nx.shortest_path_length(G, c1, c2))
                except Exception:
                    pass
    
    if not distances:
        return 0.5
    
    avg_dist = sum(distances) / len(distances)
    try:
        diameter = nx.diameter(G) if nx.is_connected(G) else total_nodes // 2
    except Exception:
        diameter = total_nodes // 2
    
    ideal_dist = diameter / 3.0
    return min(avg_dist / ideal_dist, 1.0)


def _get_adaptive_aggregation(epoch, total_epochs, base_aggregation='min'):
    """
    自适应聚合策略：早期用 mean 鼓励泛化，后期逐步偏向 min 强化最坏情况鲁棒性。
    
    Returns:
        str: 当前 epoch 使用的聚合方式
    """
    if epoch is None or total_epochs is None or total_epochs <= 0:
        return base_aggregation
    progress = epoch / total_epochs
    if progress < 0.25:
        return 'mean'
    elif progress < 0.5:
        return 'mean' if base_aggregation == 'min' else base_aggregation
    elif progress < 0.75:
        # 混合：取 mean 和 min 的平均值（通过返回 'mean' 并在后续处理）
        return 'mean'
    else:
        return base_aggregation


def _get_curriculum_methods(epoch, total_epochs, available_methods):
    """
    课程式攻击方法采样：早期用简单攻击，后期逐步加入复杂攻击。
    
    Returns:
        list: 当前 epoch 可用的攻击方法子集
    """
    if epoch is None or total_epochs is None or total_epochs <= 0:
        return available_methods
    
    progress = epoch / total_epochs
    import random
    
    # 按分组筛选
    easy = [m for m in available_methods if m in _ATTACK_METHOD_GROUPS['easy']]
    medium = [m for m in available_methods if m in _ATTACK_METHOD_GROUPS['medium']]
    hard = [m for m in available_methods if m in _ATTACK_METHOD_GROUPS['hard']]
    
    # 确保每个分组至少有一些方法
    if not easy:
        easy = [m for m in available_methods if m in ['degree', 'random']]
    if not medium:
        medium = [m for m in available_methods if m not in easy and m not in hard]
    if not hard:
        hard = [m for m in available_methods if m not in easy and m not in medium]
    
    selected = []
    if progress < 0.25:
        # 前 25%：只用 easy
        selected = easy
    elif progress < 0.5:
        # 25-50%：easy + 50% medium
        selected = easy + random.sample(medium, min(len(medium), max(1, len(medium) // 2)))
    elif progress < 0.75:
        # 50-75%：全部 medium + 50% hard
        selected = easy + medium + random.sample(hard, min(len(hard), max(1, len(hard) // 2)))
    else:
        # 后 25%：全部可用
        selected = available_methods
    
    return selected


def calculate_adversarial_reward(
    G,
    centers,
    attack_ratio=0.1,
    attack_methods=None,
    aggregation='min',  # 'min', 'mean', 'worst'
    w_gcc=0.7,
    w_disp=0.3,
    sample_methods=None,  # 每次奖励计算随机采样方法数 (None 表示全部)
    cache_sequences=None,
    epoch=None,
    total_epochs=None,
    use_curriculum=False,
    use_adaptive_aggregation=False,
):
    """
    对抗式多攻击奖励函数（增强版）。
    
    使用 network_dismantling 中的多种拆解方法生成攻击序列，
    综合评估控制器部署在不同攻击模式下的鲁棒性。
    
    新增功能：
        - 课程式攻击采样 (use_curriculum=True)
        - 自适应聚合策略 (use_adaptive_aggregation=True)
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        attack_ratio: 攻击比例
        attack_methods: 攻击方法列表，None 时使用默认池
        aggregation: 聚合方式 'min'(最脆弱攻击下的表现), 'mean'(平均), 'worst'(最差)
        w_gcc: GCC 保持率权重
        w_disp: 分散性权重
        sample_methods: 每次随机采样方法数，用于加速训练 (None 表示使用全部)
        cache_sequences: 攻击序列缓存 dict{(method, graph_id): sequence} (已弃用，使用模块级缓存)
        epoch: 当前训练 epoch (用于课程/自适应)
        total_epochs: 总训练 epoch 数 (用于课程/自适应)
        use_curriculum: 是否启用课程式攻击采样
        use_adaptive_aggregation: 是否启用自适应聚合
        
    Returns:
        float: 奖励值
    """
    if not centers:
        return 0.0
    
    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        return 0.0
    
    if attack_methods is None:
        attack_methods = ADVERSARIAL_ATTACK_METHODS
    
    # 过滤掉不可用或无效的方法
    available_methods = []
    for method in attack_methods:
        try:
            import network_dismantling as nd
            if method in nd.list_methods():
                available_methods.append(method)
        except Exception:
            pass
    
    if not available_methods:
        # 降级为单一 degree attack
        available_methods = ['degree']
    
    # 如果没有显式传入 epoch，尝试从全局跟踪器读取
    if epoch is None:
        epoch = _epoch_tracker.epoch
    if total_epochs is None:
        total_epochs = _epoch_tracker.total_epochs
    
    # 课程式方法筛选
    if use_curriculum and epoch is not None and total_epochs is not None:
        available_methods = _get_curriculum_methods(epoch, total_epochs, available_methods)
    
    # 随机采样方法子集（用于加速训练）
    if sample_methods is not None and 0 < sample_methods < len(available_methods):
        import random
        available_methods = random.sample(available_methods, sample_methods)
    
    # 自适应聚合
    effective_aggregation = aggregation
    if use_adaptive_aggregation and epoch is not None and total_epochs is not None:
        effective_aggregation = _get_adaptive_aggregation(epoch, total_epochs, aggregation)
    
    # 为每种攻击方式计算 GCC 保持率
    gcc_scores = []
    
    for method in available_methods:
        try:
            import network_dismantling as nd
            # 优先使用模块级共享缓存
            sequence = _get_cached_sequence(G, method)
            if sequence is None:
                sequence = nd.dismantle(G, method=method)
                _set_cached_sequence(G, method, sequence)
            
            retention = _simulate_attack_sequence(G, centers, sequence, attack_ratio)
            gcc_scores.append(retention)
        except Exception:
            # 单个方法失败时跳过
            continue
    
    if not gcc_scores:
        gcc_scores = [0.0]
    
    # 聚合 GCC 分数
    if effective_aggregation == 'min' or effective_aggregation == 'worst':
        gcc_reward = min(gcc_scores)
    elif effective_aggregation == 'mean':
        gcc_reward = sum(gcc_scores) / len(gcc_scores)
    elif effective_aggregation == 'max':
        gcc_reward = max(gcc_scores)
    else:
        gcc_reward = min(gcc_scores)
    
    # 如果是自适应混合阶段 (0.5-0.75 progress 且 use_adaptive_aggregation)，
    # 取 mean 和 min 的平均值作为折中
    if use_adaptive_aggregation and epoch is not None and total_epochs is not None:
        progress = epoch / total_epochs
        if 0.5 <= progress < 0.75:
            mean_val = sum(gcc_scores) / len(gcc_scores)
            min_val = min(gcc_scores)
            gcc_reward = 0.5 * mean_val + 0.5 * min_val
    
    # 分散性奖励
    dispersion_score = _get_dispersion_score(G, centers)
    
    return w_gcc * gcc_reward + w_disp * dispersion_score


def calculate_adversarial_reward_with_metrics(
    G,
    centers,
    attack_ratio=0.1,
    attack_methods=None,
    metric_weights=None,
):
    """
    扩展版对抗奖励：同时考虑 GCC、CSA、CCE、WCP 四种指标。
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        attack_ratio: 攻击比例
        attack_methods: 攻击方法列表
        metric_weights: 指标权重字典 {'gcc': 0.5, 'csa': 0.2, 'cce': 0.15, 'wcp': 0.15}
        
    Returns:
        float: 综合奖励值
    """
    if not centers:
        return 0.0
    
    if metric_weights is None:
        metric_weights = {'gcc': 0.5, 'csa': 0.2, 'cce': 0.15, 'wcp': 0.15}
    
    # 复用 adversarial 奖励的 GCC 部分
    gcc_reward = calculate_adversarial_reward(
        G, centers, attack_ratio=attack_ratio,
        attack_methods=attack_methods, aggregation='min',
        w_gcc=1.0, w_disp=0.0
    )
    
    # 计算当前网络的功能指标（作为补充奖励）
    from network_metrics.functional.controller import calculate_csa, calculate_cce, calculate_wcp
    csa_val = calculate_csa(G, centers) if 'csa' in metric_weights else 0.0
    cce_val = calculate_cce(G, centers) if 'cce' in metric_weights else 0.0
    wcp_val = calculate_wcp(G, centers) if 'wcp' in metric_weights else 0.0
    
    reward = (
        metric_weights.get('gcc', 0.0) * gcc_reward
        + metric_weights.get('csa', 0.0) * csa_val
        + metric_weights.get('cce', 0.0) * cce_val
        + metric_weights.get('wcp', 0.0) * wcp_val
    )
    
    return reward


def calculate_survival_reward(G, centers, max_attack_steps=50):
    """
    生存能力奖励函数（优化版）
    
    模拟逐步攻击过程，计算网络的累积生存能力。
    目标：使 GCC 尽可能慢地分解。
    
    奖励计算:
    R = Σ(GCC_t / total_nodes) * decay^t + bonus_terms
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        max_attack_steps: 最大攻击步数（默认50步或节点数的一半）
        
    Returns:
        float: 累积生存奖励
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    total_nodes = G.number_of_nodes()
    
    if total_nodes == 0:
        return 0.0
    
    # 创建图副本进行模拟
    G_sim = G.copy()
    remaining_centers = centers_set.copy()
    
    # 获取攻击序列（按度数降序）
    degrees = dict(G_sim.degree())
    attack_sequence = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    # 限制攻击步数
    max_steps = min(max_attack_steps, len(attack_sequence), total_nodes // 2)
    
    # 累积生存奖励
    survival_score = 0.0
    decay = 0.98  # 时间衰减因子
    
    for step in range(max_steps):
        if step >= len(attack_sequence):
            break
            
        node = attack_sequence[step]
        
        # 如果节点已被移除，跳过
        if not G_sim.has_node(node):
            continue
        
        # 移除节点
        G_sim.remove_node(node)
        remaining_centers.discard(node)
        
        # 计算当前 GCC（包含控制器的最大连通分量）
        if G_sim.number_of_nodes() == 0 or len(remaining_centers) == 0:
            break
        
        components = list(nx.connected_components(G_sim))
        controlled_sizes = []
        for comp in components:
            if not remaining_centers.isdisjoint(comp):
                controlled_sizes.append(len(comp))
        
        if controlled_sizes:
            gcc_ratio = max(controlled_sizes) / total_nodes
            survival_score += gcc_ratio * (decay ** step)
    
    # 归一化到 [0, 1]
    max_possible_score = sum(decay ** i for i in range(max_steps))
    normalized_score = survival_score / max_possible_score if max_possible_score > 0 else 0.0
    
    # 添加控制器保护奖励（控制器不在高度数节点中）
    protection_bonus = 0.0
    high_degree_nodes = set(attack_sequence[:max(1, total_nodes // 10)])
    protected_controllers = len(centers_set - high_degree_nodes)
    protection_bonus = protected_controllers / len(centers_set) if centers_set else 0.0
    
    # 添加连通性覆盖奖励
    components = list(nx.connected_components(G))
    coverage = sum(1 for c in components if not centers_set.isdisjoint(c)) / max(1, len(components))
    
    # 组合奖励：60% 生存能力 + 20% 保护奖励 + 20% 覆盖奖励
    return 0.6 * normalized_score + 0.2 * protection_bonus + 0.2 * coverage


def calculate_robustness_integral_reward(G, centers, attack_ratios=None):
    """
    鲁棒性积分奖励（R值最大化）
    
    直接优化曲线下面积（R值），这是最终评价指标。
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        attack_ratios: 攻击比例列表（默认从0到0.5，步长0.05）
        
    Returns:
        float: R值（曲线下面积）
    """
    if not centers:
        return 0.0
    
    if attack_ratios is None:
        attack_ratios = [i * 0.05 for i in range(11)]  # 0, 0.05, 0.1, ..., 0.5
    
    centers_set = set(centers)
    total_nodes = G.number_of_nodes()
    
    if total_nodes == 0:
        return 0.0
    
    # 计算不同攻击比例下的 GCC 保持率
    gcc_values = []
    
    for ratio in attack_ratios:
        G_attacked = G.copy()
        remaining_centers = centers_set.copy()
        
        # 模拟 degree attack
        num_remove = max(1, int(total_nodes * ratio))
        degrees = dict(G_attacked.degree())
        targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_remove]
        
        for node in targets:
            if G_attacked.has_node(node):
                G_attacked.remove_node(node)
                remaining_centers.discard(node)
        
        # 计算 GCC 保持率
        if G_attacked.number_of_nodes() > 0 and remaining_centers:
            components = list(nx.connected_components(G_attacked))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            gcc_ratio = max(controlled_sizes) / total_nodes if controlled_sizes else 0.0
        else:
            gcc_ratio = 0.0
        
        gcc_values.append(gcc_ratio)
    
    # 使用梯形法计算曲线下面积（R值）
    r_value = 0.0
    for i in range(len(attack_ratios) - 1):
        dx = attack_ratios[i + 1] - attack_ratios[i]
        avg_y = (gcc_values[i] + gcc_values[i + 1]) / 2
        r_value += dx * avg_y
    
    # 归一化（最大可能面积 = 0.5，因为 x 从 0 到 0.5）
    max_area = attack_ratios[-1] - attack_ratios[0]
    normalized_r = r_value / max_area if max_area > 0 else 0.0
    
    return normalized_r


def calculate_enhanced_reward(G, centers, mode='survival'):
    """
    增强版奖励函数（统一入口）
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        mode: 奖励模式
            - 'survival': 生存能力奖励（推荐）
            - 'integral': R值积分奖励
            - 'multi_attack': 多攻击比例奖励
            - 'gcc': 基础 GCC 奖励
            
    Returns:
        float: 奖励值
    """
    if mode == 'survival':
        return calculate_survival_reward(G, centers)
    elif mode == 'integral':
        return calculate_robustness_integral_reward(G, centers)
    elif mode == 'multi_attack':
        return calculate_multi_attack_reward(G, centers)
    elif mode == 'gcc':
        return calculate_gcc_focused_reward(G, centers)
    else:
        return calculate_gcc_focused_reward(G, centers)


# ============================================================
# 自适应奖励函数（根据网络规模自动调整）
# ============================================================

def calculate_adaptive_reward(G, centers, mode='combined'):
    """
    自适应奖励函数：根据网络规模自动调整计算参数
    
    小规模网络特殊处理：
    - 更高的攻击比例（获得更强信号）
    - 更重视控制器分散性
    - 更重视控制器保护（避免高度数节点）
    
    Args:
        G: NetworkX图
        centers: 控制器列表
        mode: 优化模式
        
    Returns:
        float: 奖励值
    """
    if not centers:
        return 0.0
    
    num_nodes = G.number_of_nodes()
    centers_set = set(centers)
    
    if num_nodes == 0:
        return 0.0
    
    # 根据网络规模调整参数
    if num_nodes < SMALL_NETWORK_THRESHOLD:
        # 小规模网络：更高攻击比例，更重视分散性
        attack_ratios = [0.15, 0.25, 0.35]
        gcc_weight = 0.40
        dispersion_weight = 0.25
        protection_weight = 0.20
        coverage_weight = 0.15
    elif num_nodes < MEDIUM_NETWORK_THRESHOLD:
        # 中等规模网络
        attack_ratios = [0.1, 0.2, 0.3]
        gcc_weight = 0.45
        dispersion_weight = 0.22
        protection_weight = 0.18
        coverage_weight = 0.15
    else:
        # 大规模网络：标准参数
        attack_ratios = [0.05, 0.1, 0.15, 0.2]
        gcc_weight = 0.50
        dispersion_weight = 0.20
        protection_weight = 0.15
        coverage_weight = 0.15
    
    # 1. 多攻击比例GCC奖励
    gcc_scores = []
    for ratio in attack_ratios:
        G_attacked = G.copy()
        remaining_centers = centers_set.copy()
        
        num_remove = max(1, int(num_nodes * ratio))
        degrees = dict(G_attacked.degree())
        targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_remove]
        
        for node in targets:
            if G_attacked.has_node(node):
                G_attacked.remove_node(node)
                remaining_centers.discard(node)
        
        if G_attacked.number_of_nodes() > 0 and remaining_centers:
            components = list(nx.connected_components(G_attacked))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            gcc_ratio = max(controlled_sizes) / num_nodes if controlled_sizes else 0.0
        else:
            gcc_ratio = 0.0
        
        gcc_scores.append(gcc_ratio)
    
    # 加权平均（后期攻击权重更高）
    weights = np.linspace(0.5, 1.5, len(gcc_scores))
    weights = weights / weights.sum()
    gcc_reward = sum(w * s for w, s in zip(weights, gcc_scores))
    
    # 2. 分散性奖励
    dispersion_reward = _calculate_adaptive_dispersion(G, centers)
    
    # 3. 保护奖励
    protection_reward = _calculate_adaptive_protection(G, centers)
    
    # 4. 覆盖奖励
    coverage_reward = _calculate_adaptive_coverage(G, centers)
    
    # 组合奖励
    total_reward = (
        gcc_weight * gcc_reward +
        dispersion_weight * dispersion_reward +
        protection_weight * protection_reward +
        coverage_weight * coverage_reward
    )
    
    return total_reward


def _calculate_adaptive_dispersion(G, centers):
    """计算自适应分散性奖励"""
    if len(centers) <= 1:
        return 0.5
    
    centers_list = list(centers)
    distances = []
    
    for i, c1 in enumerate(centers_list):
        for c2 in centers_list[i+1:]:
            if G.has_node(c1) and G.has_node(c2):
                try:
                    if nx.has_path(G, c1, c2):
                        d = nx.shortest_path_length(G, c1, c2)
                        distances.append(d)
                except:
                    pass
    
    if not distances:
        return 0.5
    
    avg_dist = np.mean(distances)
    min_dist = min(distances)
    
    try:
        diameter = nx.diameter(G) if nx.is_connected(G) else G.number_of_nodes() // 3
    except:
        diameter = G.number_of_nodes() // 3
    
    ideal_dist = max(diameter / 3, 2)
    
    avg_score = min(avg_dist / ideal_dist, 1.0)
    min_score = min(min_dist / (ideal_dist * 0.5), 1.0)
    
    return 0.6 * avg_score + 0.4 * min_score


def _calculate_adaptive_protection(G, centers):
    """计算自适应保护奖励（控制器不在高度数节点）"""
    if not centers:
        return 0.0
    
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    # 根据网络规模调整"高度数"的定义
    num_nodes = G.number_of_nodes()
    if num_nodes < SMALL_NETWORK_THRESHOLD:
        top_ratio = 0.15  # 小图：前15%
    else:
        top_ratio = 0.10  # 大图：前10%
    
    top_k = max(1, int(len(sorted_nodes) * top_ratio))
    high_degree_nodes = set(sorted_nodes[:top_k])
    
    protected = len([c for c in centers if c not in high_degree_nodes])
    return protected / len(centers)


def _calculate_adaptive_coverage(G, centers):
    """计算自适应覆盖奖励"""
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    components = list(nx.connected_components(G))
    
    covered = sum(1 for c in components if not centers_set.isdisjoint(c))
    return covered / len(components) if components else 1.0


def calculate_adaptive_stepwise_reward(G, previous_centers, new_center, mode='combined'):
    """
    自适应Step-wise奖励：根据网络规模调整
    
    Args:
        G: NetworkX图
        previous_centers: 之前已选的控制器
        new_center: 新选择的控制器
        mode: 优化模式
    """
    if new_center is None or not G.has_node(new_center):
        return 0.0
    
    num_nodes = G.number_of_nodes()
    centers_before = set(previous_centers) if previous_centers else set()
    centers_after = centers_before | {new_center}
    
    # 根据网络规模调整权重
    if num_nodes < SMALL_NETWORK_THRESHOLD:
        w_gcc, w_coverage, w_cost = 2.5, 1.5, 0.3  # 小图：更重视GCC和覆盖
    elif num_nodes < MEDIUM_NETWORK_THRESHOLD:
        w_gcc, w_coverage, w_cost = 2.2, 1.2, 0.4
    else:
        w_gcc, w_coverage, w_cost = 2.0, 1.0, 0.5
    
    # GCC增量
    gcc_before = calculate_gcc(G, list(centers_before))
    gcc_after = calculate_gcc(G, list(centers_after))
    delta_gcc = (gcc_after - gcc_before) / max(num_nodes, 1)
    
    # 覆盖增量
    components = list(nx.connected_components(G))
    controlled_before = set()
    for comp in components:
        if not centers_before.isdisjoint(comp):
            controlled_before.update(comp)
    controlled_after = set()
    for comp in components:
        if not centers_after.isdisjoint(comp):
            controlled_after.update(comp)
    delta_coverage = len(controlled_after - controlled_before) / max(num_nodes, 1)
    
    # 路径成本
    cost_path = 0.0
    if centers_before:
        min_dists = []
        for c in centers_before:
            try:
                if nx.has_path(G, new_center, c):
                    min_dists.append(nx.shortest_path_length(G, new_center, c))
            except:
                pass
        if min_dists:
            cost_path = min(min_dists) / max(num_nodes, 1)
        else:
            cost_path = 1.0
    
    # 小图额外奖励：新控制器不是高度数节点
    protection_bonus = 0.0
    if num_nodes < SMALL_NETWORK_THRESHOLD:
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
        top_k = max(1, int(len(sorted_nodes) * 0.15))
        if new_center not in sorted_nodes[:top_k]:
            protection_bonus = 0.1
    
    return w_gcc * delta_gcc + w_coverage * delta_coverage - w_cost * cost_path + protection_bonus


# ============================================================
# 统一GCC导向奖励函数（用于UnifiedGATPolicy训练）
# ============================================================

def calculate_unified_gcc_reward(G, centers, config=None):
    """
    统一的GCC导向奖励函数
    
    专为统一模型设计，主要优化:
    1. GCC保持率：多攻击比例下的GCC保持率均值
    2. R值近似：近似计算鲁棒性曲线下面积
    3. 覆盖奖励：控制器覆盖的连通分量比例
    
    奖励计算公式:
    R = α * GCC_retention + β * R_approx + γ * coverage_bonus
    
    其中:
    - α = 0.6: GCC保持率权重
    - β = 0.3: R值近似权重
    - γ = 0.1: 覆盖奖励权重
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        config: 可选配置字典，支持:
            - 'gcc_weight': GCC保持率权重 (默认0.6)
            - 'r_weight': R值权重 (默认0.3)
            - 'coverage_weight': 覆盖权重 (默认0.1)
            
    Returns:
        float: 奖励值 [0, 1]
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    n = G.number_of_nodes()
    
    if n == 0:
        return 0.0
    
    # 默认配置
    if config is None:
        config = {}
    
    gcc_weight = config.get('gcc_weight', 0.6)
    r_weight = config.get('r_weight', 0.3)
    coverage_weight = config.get('coverage_weight', 0.1)
    
    # 根据网络规模自适应选择攻击比例
    if n < 50:
        attack_ratios = [0.20, 0.30, 0.40]
        sample_points = 6
    elif n < 100:
        attack_ratios = [0.15, 0.25, 0.35]
        sample_points = 8
    elif n < 200:
        attack_ratios = [0.10, 0.20, 0.30]
        sample_points = 10
    else:
        attack_ratios = [0.05, 0.10, 0.15, 0.20]
        sample_points = 12
    
    # ===== 1. 计算多攻击比例下的GCC保持率 =====
    gcc_scores = []
    for ratio in attack_ratios:
        gcc_ratio = _simulate_attack_and_get_gcc_ratio(G, centers_set, ratio)
        gcc_scores.append(gcc_ratio)
    
    # 使用加权平均（后期攻击权重更高，因为更重要）
    weights = np.linspace(0.5, 1.5, len(gcc_scores))
    weights = weights / weights.sum()
    gcc_retention = sum(w * s for w, s in zip(weights, gcc_scores))
    
    # ===== 2. 近似R值计算 =====
    r_approx = _calculate_approximate_r_value(G, centers_set, sample_points)
    
    # ===== 3. 覆盖奖励 =====
    coverage_bonus = _calculate_coverage_bonus(G, centers_set)
    
    # ===== 4. 组合奖励 =====
    total_reward = (
        gcc_weight * gcc_retention +
        r_weight * r_approx +
        coverage_weight * coverage_bonus
    )
    
    return total_reward


def _simulate_attack_and_get_gcc_ratio(G, centers_set, attack_ratio):
    """
    模拟攻击并返回GCC保持率
    
    Args:
        G: NetworkX 图
        centers_set: 控制器集合
        attack_ratio: 攻击比例
        
    Returns:
        float: GCC保持率 [0, 1]
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    
    G_attacked = G.copy()
    remaining_centers = centers_set.copy()
    
    # 模拟 degree attack
    num_remove = max(1, int(n * attack_ratio))
    degrees = dict(G_attacked.degree())
    targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_remove]
    
    for node in targets:
        if G_attacked.has_node(node):
            G_attacked.remove_node(node)
            remaining_centers.discard(node)
    
    # 计算包含控制器的最大连通分量
    if G_attacked.number_of_nodes() == 0 or len(remaining_centers) == 0:
        return 0.0
    
    components = list(nx.connected_components(G_attacked))
    controlled_sizes = []
    for comp in components:
        if not remaining_centers.isdisjoint(comp):
            controlled_sizes.append(len(comp))
    
    if controlled_sizes:
        return max(controlled_sizes) / n
    
    return 0.0


def _calculate_approximate_r_value(G, centers_set, sample_points=10):
    """
    近似计算R值（鲁棒性曲线下面积）
    
    使用较少的采样点来近似计算，提高效率。
    
    Args:
        G: NetworkX 图
        centers_set: 控制器集合
        sample_points: 采样点数量
        
    Returns:
        float: 近似R值 [0, 1]
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    
    # 根据规模调整采样范围
    if n < 100:
        max_ratio = 0.5  # 小图：最多攻击50%
    else:
        max_ratio = 0.4  # 大图：最多攻击40%
    
    ratios = np.linspace(0, max_ratio, sample_points)
    gcc_values = []
    
    for ratio in ratios:
        if ratio == 0:
            gcc_values.append(1.0)  # 初始状态，GCC = 100%
        else:
            gcc_ratio = _simulate_attack_and_get_gcc_ratio(G, centers_set, ratio)
            gcc_values.append(gcc_ratio)
    
    # 使用梯形法则计算曲线下面积
    r_value = np.trapz(gcc_values, ratios)
    
    # 归一化到[0, 1]
    max_area = max_ratio  # 理想情况下，GCC始终为1
    normalized_r = r_value / max_area if max_area > 0 else 0.0
    
    return normalized_r


def _calculate_coverage_bonus(G, centers_set):
    """
    计算覆盖奖励
    
    奖励控制器覆盖多个连通分量的情况。
    
    Args:
        G: NetworkX 图
        centers_set: 控制器集合
        
    Returns:
        float: 覆盖奖励 [0, 1]
    """
    if not centers_set:
        return 0.0
    
    components = list(nx.connected_components(G))
    
    if not components:
        return 0.0
    
    # 计算被控制器覆盖的连通分量数量
    covered_count = sum(1 for comp in components if not centers_set.isdisjoint(comp))
    
    # 基础覆盖率
    coverage_ratio = covered_count / len(components)
    
    # 额外奖励：控制器分布在不同的连通分量中
    centers_in_different_components = 0
    for center in centers_set:
        for comp in components:
            if center in comp:
                centers_in_different_components += 1
                break
    
    distribution_bonus = min(centers_in_different_components / max(1, len(components)), 1.0)
    
    return 0.7 * coverage_ratio + 0.3 * distribution_bonus


def calculate_unified_stepwise_reward(G, previous_centers, new_center, config=None):
    """
    统一模型的Step-wise奖励函数
    
    计算添加新控制器后的增量奖励。
    
    Args:
        G: NetworkX 图
        previous_centers: 之前已选的控制器
        new_center: 新选择的控制器
        config: 可选配置
        
    Returns:
        float: 增量奖励
    """
    if new_center is None or not G.has_node(new_center):
        return 0.0
    
    centers_before = set(previous_centers) if previous_centers else set()
    centers_after = centers_before | {new_center}
    
    # 计算前后的统一奖励
    reward_before = calculate_unified_gcc_reward(G, list(centers_before), config) if centers_before else 0.0
    reward_after = calculate_unified_gcc_reward(G, list(centers_after), config)
    
    # 增量奖励
    delta_reward = reward_after - reward_before
    
    # 存活奖励（避免稀疏奖励问题）
    survival_bonus = 0.01
    
    # 小网络的位置保护奖励
    n = G.number_of_nodes()
    protection_bonus = 0.0
    if n < 100:
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
        top_k = max(1, int(len(sorted_nodes) * 0.15))
        if new_center not in sorted_nodes[:top_k]:
            protection_bonus = 0.02  # 不选择高度数节点的奖励
    
    return delta_reward + survival_bonus + protection_bonus


# ============================================================
# 课程学习奖励函数 (Curriculum Learning Rewards)
# ============================================================

class CurriculumRewardPhase:
    """课程学习阶段枚举"""
    PHASE_1_CONNECTIVITY = 1  # 阶段1：只奖励连通性（GCC）
    PHASE_2_COVERAGE = 2      # 阶段2：加入覆盖率奖励
    PHASE_3_FULL = 3          # 阶段3：加入分散度和抗攻击奖励
    PHASE_4_REAL_NETWORK = 4  # 阶段4：真实网络增强训练
    PHASE_4_ENHANCED = 5      # 阶段4增强版：极致GCC+R值优化


def calculate_phase1_reward(G, centers):
    """
    阶段1奖励：只优化连通性（GCC保持）
    
    目标：让模型首先学会保持网络的最大连通性
    
    奖励计算:
    R = GCC_ratio (攻击后的GCC / 原始节点数)
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        
    Returns:
        float: 奖励值 [0, 1]
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    n = G.number_of_nodes()
    
    if n == 0:
        return 0.0
    
    # 单一攻击比例，简化学习
    attack_ratio = 0.15
    
    G_attacked = G.copy()
    remaining_centers = centers_set.copy()
    
    # 模拟 degree attack
    num_remove = max(1, int(n * attack_ratio))
    degrees = dict(G_attacked.degree())
    targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_remove]
    
    for node in targets:
        if G_attacked.has_node(node):
            G_attacked.remove_node(node)
            remaining_centers.discard(node)
    
    # 计算 GCC 保持率
    if G_attacked.number_of_nodes() == 0 or len(remaining_centers) == 0:
        return 0.0
    
    components = list(nx.connected_components(G_attacked))
    controlled_sizes = []
    for comp in components:
        if not remaining_centers.isdisjoint(comp):
            controlled_sizes.append(len(comp))
    
    if controlled_sizes:
        gcc_ratio = max(controlled_sizes) / n
        return gcc_ratio
    
    return 0.0


def calculate_phase2_reward(G, centers):
    """
    阶段2奖励：GCC + 覆盖率
    
    在阶段1的基础上，加入覆盖率奖励，
    鼓励控制器覆盖更多的连通分量。
    
    奖励计算:
    R = 0.7 * GCC_ratio + 0.3 * coverage_ratio
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        
    Returns:
        float: 奖励值 [0, 1]
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    n = G.number_of_nodes()
    
    if n == 0:
        return 0.0
    
    # === GCC 保持率（使用阶段1的方法）===
    gcc_reward = calculate_phase1_reward(G, centers)
    
    # === 覆盖率奖励 ===
    # 计算控制器覆盖的节点比例（2跳邻域内）
    covered_nodes = set()
    for c in centers_set:
        if not G.has_node(c):
            continue
        # 0跳
        covered_nodes.add(c)
        # 1跳
        for n1 in G.neighbors(c):
            covered_nodes.add(n1)
            # 2跳
            for n2 in G.neighbors(n1):
                covered_nodes.add(n2)
    
    coverage_ratio = len(covered_nodes) / n
    
    # 组合奖励
    reward = 0.7 * gcc_reward + 0.3 * coverage_ratio
    
    return reward


def calculate_phase3_reward(G, centers):
    """
    阶段3奖励：GCC + 覆盖率 + 分散度 + 抗攻击
    
    完整的多目标奖励函数，在前两个阶段的基础上，
    加入分散度和多轮抗攻击奖励。
    
    奖励计算:
    R = 0.4 * GCC_multi_attack + 0.2 * coverage + 0.2 * dispersion + 0.2 * protection
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        
    Returns:
        float: 奖励值 [0, 1]
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    n = G.number_of_nodes()
    
    if n == 0:
        return 0.0
    
    # === 1. 多轮攻击 GCC 保持率 ===
    attack_ratios = [0.10, 0.20, 0.30]
    gcc_scores = []
    
    for ratio in attack_ratios:
        G_attacked = G.copy()
        remaining_centers = centers_set.copy()
        
        num_remove = max(1, int(n * ratio))
        degrees = dict(G_attacked.degree())
        targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_remove]
        
        for node in targets:
            if G_attacked.has_node(node):
                G_attacked.remove_node(node)
                remaining_centers.discard(node)
        
        if G_attacked.number_of_nodes() > 0 and remaining_centers:
            components = list(nx.connected_components(G_attacked))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            gcc_ratio = max(controlled_sizes) / n if controlled_sizes else 0.0
        else:
            gcc_ratio = 0.0
        
        gcc_scores.append(gcc_ratio)
    
    # 加权平均（后期攻击权重更高）
    gcc_reward = 0.2 * gcc_scores[0] + 0.3 * gcc_scores[1] + 0.5 * gcc_scores[2]
    
    # === 2. 覆盖率奖励 ===
    covered_nodes = set()
    for c in centers_set:
        if not G.has_node(c):
            continue
        covered_nodes.add(c)
        for n1 in G.neighbors(c):
            covered_nodes.add(n1)
            for n2 in G.neighbors(n1):
                covered_nodes.add(n2)
    
    coverage_reward = len(covered_nodes) / n
    
    # === 3. 分散度奖励 ===
    dispersion_reward = 0.5
    if len(centers) > 1:
        distances = []
        centers_list = list(centers)
        
        for i, c1 in enumerate(centers_list):
            for c2 in centers_list[i + 1:]:
                if G.has_node(c1) and G.has_node(c2):
                    try:
                        if nx.has_path(G, c1, c2):
                            d = nx.shortest_path_length(G, c1, c2)
                            distances.append(d)
                    except:
                        pass
        
        if distances:
            avg_dist = np.mean(distances)
            min_dist = min(distances)
            
            try:
                diameter = nx.diameter(G) if nx.is_connected(G) else n // 3
            except:
                diameter = n // 3
            
            ideal_dist = max(diameter / 3, 2)
            
            # 平均距离得分 + 最小距离得分
            avg_score = min(avg_dist / ideal_dist, 1.0)
            min_score = min(min_dist / (ideal_dist * 0.5), 1.0)
            dispersion_reward = 0.6 * avg_score + 0.4 * min_score
    
    # === 4. 保护奖励（控制器不在高度数节点）===
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    top_k = max(1, int(len(sorted_nodes) * 0.15))
    high_degree_nodes = set(sorted_nodes[:top_k])
    
    protected_controllers = len([c for c in centers_set if c not in high_degree_nodes])
    protection_reward = protected_controllers / len(centers_set) if centers_set else 0.0
    
    # === 组合奖励 ===
    total_reward = (
        0.40 * gcc_reward +
        0.20 * coverage_reward +
        0.20 * dispersion_reward +
        0.20 * protection_reward
    )
    
    return total_reward


def calculate_phase4_reward(G, centers):
    """
    阶段4奖励：真实网络增强训练
    
    专门针对真实网络的特点设计，强调：
    1. R值积分（曲线下面积）- 直接优化最终评价指标
    2. 多攻击策略（degree + betweenness）
    3. 更强的鲁棒性要求（更高攻击比例）
    4. 拓扑感知的分散性
    
    奖励计算:
    R = 0.35 * R_value_degree + 0.25 * R_value_betweenness + 
        0.15 * dispersion + 0.15 * coverage + 0.10 * protection
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        
    Returns:
        float: 奖励值 [0, 1]
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    n = G.number_of_nodes()
    
    if n == 0:
        return 0.0
    
    # === 1. R值计算 (Degree Attack) ===
    # 使用更细粒度的攻击比例来计算曲线下面积
    attack_ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    # Degree attack 的 GCC 序列
    gcc_values_degree = []
    degrees = dict(G.degree())
    degree_targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    for ratio in attack_ratios:
        if ratio == 0:
            gcc_values_degree.append(1.0)
            continue
            
        G_attacked = G.copy()
        remaining_centers = centers_set.copy()
        
        num_remove = max(1, int(n * ratio))
        for node in degree_targets[:num_remove]:
            if G_attacked.has_node(node):
                G_attacked.remove_node(node)
                remaining_centers.discard(node)
        
        if G_attacked.number_of_nodes() > 0 and remaining_centers:
            components = list(nx.connected_components(G_attacked))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            gcc_ratio = max(controlled_sizes) / n if controlled_sizes else 0.0
        else:
            gcc_ratio = 0.0
        
        gcc_values_degree.append(gcc_ratio)
    
    # 使用梯形法则计算R值 (degree attack)
    r_value_degree = 0.0
    for i in range(len(attack_ratios) - 1):
        dx = attack_ratios[i + 1] - attack_ratios[i]
        avg_y = (gcc_values_degree[i] + gcc_values_degree[i + 1]) / 2
        r_value_degree += dx * avg_y
    
    # 归一化 R值
    max_area = attack_ratios[-1] - attack_ratios[0]
    r_value_degree = r_value_degree / max_area if max_area > 0 else 0.0
    
    # === 2. R值计算 (Betweenness Attack) ===
    # 真实网络中 betweenness 攻击往往更致命
    try:
        betweenness = nx.betweenness_centrality(G)
        betweenness_targets = sorted(betweenness.keys(), key=lambda x: betweenness[x], reverse=True)
    except:
        betweenness_targets = degree_targets  # 退化到 degree
    
    gcc_values_betweenness = []
    for ratio in attack_ratios:
        if ratio == 0:
            gcc_values_betweenness.append(1.0)
            continue
            
        G_attacked = G.copy()
        remaining_centers = centers_set.copy()
        
        num_remove = max(1, int(n * ratio))
        for node in betweenness_targets[:num_remove]:
            if G_attacked.has_node(node):
                G_attacked.remove_node(node)
                remaining_centers.discard(node)
        
        if G_attacked.number_of_nodes() > 0 and remaining_centers:
            components = list(nx.connected_components(G_attacked))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            gcc_ratio = max(controlled_sizes) / n if controlled_sizes else 0.0
        else:
            gcc_ratio = 0.0
        
        gcc_values_betweenness.append(gcc_ratio)
    
    # R值 (betweenness attack)
    r_value_betweenness = 0.0
    for i in range(len(attack_ratios) - 1):
        dx = attack_ratios[i + 1] - attack_ratios[i]
        avg_y = (gcc_values_betweenness[i] + gcc_values_betweenness[i + 1]) / 2
        r_value_betweenness += dx * avg_y
    r_value_betweenness = r_value_betweenness / max_area if max_area > 0 else 0.0
    
    # === 3. 拓扑感知的分散性奖励 ===
    dispersion_reward = 0.5
    if len(centers) > 1:
        distances = []
        centers_list = list(centers)
        
        for i, c1 in enumerate(centers_list):
            for c2 in centers_list[i + 1:]:
                if G.has_node(c1) and G.has_node(c2):
                    try:
                        if nx.has_path(G, c1, c2):
                            d = nx.shortest_path_length(G, c1, c2)
                            distances.append(d)
                    except:
                        pass
        
        if distances:
            avg_dist = np.mean(distances)
            min_dist = min(distances)
            
            # 使用平均路径长度作为理想距离参考
            try:
                avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else n // 4
            except:
                avg_path_length = n // 4
            
            ideal_dist = max(avg_path_length, 2)
            
            # 分散性得分：平均距离和最小距离的组合
            avg_score = min(avg_dist / ideal_dist, 1.2) / 1.2
            min_score = min(min_dist / (ideal_dist * 0.5), 1.0)
            dispersion_reward = 0.5 * avg_score + 0.5 * min_score
    
    # === 4. 覆盖率奖励 (2跳邻域) ===
    covered_nodes = set()
    for c in centers_set:
        if not G.has_node(c):
            continue
        covered_nodes.add(c)
        for n1 in G.neighbors(c):
            covered_nodes.add(n1)
            for n2 in G.neighbors(n1):
                covered_nodes.add(n2)
    
    coverage_reward = len(covered_nodes) / n
    
    # === 5. 保护奖励（控制器不在关键节点）===
    # 结合度数和介数中心性
    sorted_by_degree = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    sorted_by_betweenness = sorted(betweenness.keys(), key=lambda x: betweenness[x], reverse=True)
    
    # 取度数和介数排名前15%的节点的并集作为"高风险"节点
    top_k = max(1, int(n * 0.15))
    high_risk_nodes = set(sorted_by_degree[:top_k]) | set(sorted_by_betweenness[:top_k])
    
    protected_controllers = len([c for c in centers_set if c not in high_risk_nodes])
    protection_reward = protected_controllers / len(centers_set) if centers_set else 0.0
    
    # === 组合奖励 ===
    # 强调R值（直接优化目标）
    total_reward = (
        0.35 * r_value_degree +
        0.25 * r_value_betweenness +
        0.15 * dispersion_reward +
        0.15 * coverage_reward +
        0.10 * protection_reward
    )
    
    return total_reward


def calculate_phase4_enhanced_reward(G, centers):
    """
    阶段4增强版奖励：极致的GCC+R值优化
    
    **设计思路**：
    完全聚焦于最终评价指标，使用全攻击范围的R值计算。
    
    **奖励组成**：
    1. 完整R值积分（0%-100%攻击）- 70%权重
    2. 关键攻击点GCC保持（10%, 20%, 30%）- 20%权重  
    3. 控制器存活率（避免被早期攻击）- 10%权重
    
    Args:
        G: NetworkX 图
        centers: 控制器节点列表
        
    Returns:
        float: 奖励值 [0, 1]
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    n = G.number_of_nodes()
    
    if n == 0:
        return 0.0
    
    # === 1. 完整R值积分（0%-100%移除范围）===
    # 使用更密集的采样点来精确计算R值
    num_samples = min(50, n)  # 最多50个采样点
    attack_ratios = np.linspace(0, 1.0, num_samples)
    
    # 预先计算度数排序（避免重复计算）
    degrees = dict(G.degree())
    degree_targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    gcc_values = []
    
    for ratio in attack_ratios:
        if ratio == 0:
            gcc_values.append(1.0)
            continue
        
        # 模拟移除
        num_remove = max(1, int(n * ratio))
        
        # 创建攻击后的图（只移除节点，不重建整个图）
        remaining_nodes = set(G.nodes()) - set(degree_targets[:num_remove])
        remaining_centers = centers_set & remaining_nodes
        
        if not remaining_nodes or not remaining_centers:
            gcc_values.append(0.0)
            continue
        
        # 在剩余节点上计算GCC
        G_sub = G.subgraph(remaining_nodes)
        
        if G_sub.number_of_nodes() > 0:
            components = list(nx.connected_components(G_sub))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            gcc_ratio = max(controlled_sizes) / n if controlled_sizes else 0.0
        else:
            gcc_ratio = 0.0
        
        gcc_values.append(gcc_ratio)
    
    # 计算R值（曲线下面积）
    r_value = np.trapz(gcc_values, attack_ratios)
    # 归一化：理想情况下R=1.0（GCC始终为1）
    normalized_r = r_value / 1.0
    
    # === 2. 关键攻击点的GCC保持率 ===
    # 重点关注10%, 20%, 30%三个关键攻击点
    key_attack_ratios = [0.10, 0.20, 0.30]
    key_gcc_scores = []
    
    for ratio in key_attack_ratios:
        num_remove = max(1, int(n * ratio))
        remaining_nodes = set(G.nodes()) - set(degree_targets[:num_remove])
        remaining_centers = centers_set & remaining_nodes
        
        if remaining_nodes and remaining_centers:
            G_sub = G.subgraph(remaining_nodes)
            if G_sub.number_of_nodes() > 0:
                components = list(nx.connected_components(G_sub))
                controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
                gcc_ratio = max(controlled_sizes) / n if controlled_sizes else 0.0
            else:
                gcc_ratio = 0.0
        else:
            gcc_ratio = 0.0
        
        key_gcc_scores.append(gcc_ratio)
    
    # 加权平均：后期攻击权重更高
    weights = np.array([0.2, 0.3, 0.5])
    key_gcc_reward = np.sum(weights * np.array(key_gcc_scores))
    
    # === 3. 控制器存活率（避免控制器在高度数节点）===
    # 计算控制器在度数排名中的位置
    top_10_percent = max(1, int(n * 0.10))
    top_20_percent = max(1, int(n * 0.20))
    
    controllers_in_top10 = len([c for c in centers_set if c in degree_targets[:top_10_percent]])
    controllers_in_top20 = len([c for c in centers_set if c in degree_targets[:top_20_percent]])
    
    # 存活奖励：控制器不在高度数节点中
    survival_reward = 1.0 - (0.7 * controllers_in_top10 + 0.3 * controllers_in_top20) / len(centers_set)
    
    # === 组合奖励 ===
    total_reward = (
        0.70 * normalized_r +          # 完整R值积分（主要目标）
        0.20 * key_gcc_reward +        # 关键点GCC保持
        0.10 * survival_reward         # 控制器存活率
    )
    
    return total_reward


def get_curriculum_reward_fn(phase):
    """
    获取课程学习阶段对应的奖励函数
    
    Args:
        phase: 课程学习阶段 (1, 2, 3, 4, 或 5)
        
    Returns:
        callable: 奖励函数 (G, centers) -> float
    """
    if phase == CurriculumRewardPhase.PHASE_1_CONNECTIVITY:
        return calculate_phase1_reward
    elif phase == CurriculumRewardPhase.PHASE_2_COVERAGE:
        return calculate_phase2_reward
    elif phase == CurriculumRewardPhase.PHASE_3_FULL:
        return calculate_phase3_reward
    elif phase == CurriculumRewardPhase.PHASE_4_REAL_NETWORK:
        return calculate_phase4_reward
    elif phase == CurriculumRewardPhase.PHASE_4_ENHANCED:
        return calculate_phase4_enhanced_reward
    else:
        # 默认使用完整奖励
        return calculate_phase3_reward


def calculate_curriculum_reward(G, centers, phase=3, progress=0.0):
    """
    带平滑过渡的课程学习奖励
    
    允许在阶段之间平滑过渡，而不是硬切换。
    
    Args:
        G: NetworkX 图
        centers: 控制器列表
        phase: 当前阶段 (1, 2, 3)
        progress: 当前阶段内的进度 [0, 1]，用于平滑过渡到下一阶段
        
    Returns:
        float: 奖励值
    """
    if phase == 1:
        r1 = calculate_phase1_reward(G, centers)
        if progress > 0.8:
            # 阶段1末期，开始引入阶段2奖励
            r2 = calculate_phase2_reward(G, centers)
            blend = (progress - 0.8) / 0.2  # 0 -> 1
            return (1 - blend) * r1 + blend * r2
        return r1
    
    elif phase == 2:
        r2 = calculate_phase2_reward(G, centers)
        if progress > 0.8:
            # 阶段2末期，开始引入阶段3奖励
            r3 = calculate_phase3_reward(G, centers)
            blend = (progress - 0.8) / 0.2
            return (1 - blend) * r2 + blend * r3
        return r2
    
    else:
        return calculate_phase3_reward(G, centers)
