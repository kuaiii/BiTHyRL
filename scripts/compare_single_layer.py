# -*- coding: utf-8 -*-
"""
单层网络拓扑对比实验

对比不同拓扑构建方法在 Degree Attack 下的鲁棒性
输出各算法度分布特征，绘制性能最好的前3方法的度分布图；
结果保存在 results/single_layer_test/run_XXX 下，每次运行编号加1。
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import network_construction as nc  # 统一重构接口
from src.topology.reconstruction import (
    create_bimodal_with_num_hubs, 
    create_bimodal_theoretical,
    create_bimodal_adaptive,
    create_bimodal_adaptive_fast
)
from src.topology.generators import load_graph, construct_random, construct_ba
import os
import re
import random
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from src.utils.logger import get_logger
from collections import Counter

logger = get_logger(__name__)

# 单层测试结果根目录
SINGLE_LAYER_TEST_DIR = "results/single_layer_test"


def get_next_run_dir():
    """`
    获取本次运行的结果目录：results/single_layer_test/run_001, run_002, ...
    每次运行编号加1（根据已有 run_XXX 文件夹取最大值+1）。
    """
    os.makedirs(SINGLE_LAYER_TEST_DIR, exist_ok=True)
    existing = []
    for name in os.listdir(SINGLE_LAYER_TEST_DIR):
        m = re.match(r"^run_(\d+)$", name)
        if m:
            existing.append(int(m.group(1)))
    next_id = max(existing, default=0) + 1
    run_dir = os.path.join(SINGLE_LAYER_TEST_DIR, f"run_{next_id:03d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, next_id


def degree_distribution_summary(G):
    """返回度分布的简单概括：min, max, mean, std, 众数/典型值"""
    if G is None or G.number_of_nodes() == 0:
        return None
    degrees = [d for _, d in G.degree()]
    n = len(degrees)
    cnt = Counter(degrees)
    most_common = cnt.most_common(3)
    return {
        "min": min(degrees),
        "max": max(degrees),
        "mean": round(sum(degrees) / n, 2),
        "std": round(np.std(degrees), 2) if n > 1 else 0,
        "n": n,
        "m": G.number_of_edges(),
        "unique_degrees": len(cnt),
        "top_degrees": most_common,
    }


def print_degree_distribution_features(all_G):
    """打印每个算法的度分布特征（简单概括）"""
    print("\n" + "="*60)
    print("各算法度分布特征（简单概括）")
    print("="*60)
    for name, g in all_G.items():
        if g is None:
            print(f"  {name:<12}: (无图)")
            continue
        s = degree_distribution_summary(g)
        if s is None:
            continue
        top_str = ", ".join(f"度{k}({c}个)" for k, c in s["top_degrees"])
        print(f"  {name:<12}: N={s['n']}, M={s['m']}, "
              f"度 min={s['min']}, max={s['max']}, mean={s['mean']}, std={s['std']}; 典型: {top_str}")
    print("="*60 + "\n")


# 各算法颜色（优化后的配色方案，用于度分布子图）
DEGREE_DIST_COLORS = {
    'BA': '#2E86AB',           # 深蓝色
    'RA Network': '#A23B72',   # 深紫红色
    'ONION': '#F18F01',        # 橙色
    'ROMEN': '#C73E1D',        # 深红色
    'UNITY': '#6A994E',        # 绿色
    'FRED-ABL': '#BC4749',     # 红棕色
    'QDLM': '#7209B7',         # 紫色
    'Bimodal': '#06A77D',      # 青绿色
    # 其它方法
    'Original': '#495057',     # 深灰色
    'GA': '#D62828',           # 红色
}


def plot_all_degree_distribution(all_G, run_dir, dataset_name):
    """绘制所有方法的度分布图（每个方法一个子图）"""
    names = [name for name in all_G if all_G[name] is not None]
    if not names:
        return
    n_plots = len(names)
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, name in enumerate(names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        degrees = [d for _, d in all_G[name].degree()]
        bins = min(max(len(set(degrees)), 5), 30)
        color = DEGREE_DIST_COLORS.get(name, '#888888')
        ax.hist(degrees, bins=bins, color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Degree', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12, width=2.0, length=7)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.grid(True, alpha=0.3, linewidth=0.8)
    for j in range(n_plots, n_rows * n_cols):
        row, col = j // n_cols, j % n_cols
        axes[row, col].set_visible(False)
    plt.suptitle(f'Degree distribution: all methods ({dataset_name})', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    # 保存为 degree_dist_top3.png（内容为所有方法的度分布，便于与原有结果目录一致）
    save_path = os.path.join(run_dir, "degree_dist_top3.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"全部方法度分布图已保存: {save_path}")


def load_or_generate_graph(dataset_name):
    """
    从 GML 文件加载图。
    训练用的数据集在 dataset/all 中；测试的数据集在 dataset/testdata 中。
    """
    gml_path = os.path.join('dataset', 'testdata', f'{dataset_name}.gml')
    if not os.path.exists(gml_path):
        gml_path = os.path.join('dataset', 'all', f'{dataset_name}.gml')
    
    if os.path.exists(gml_path):
        G, _ = load_graph(gml_path)
    else:
        # 如果文件不存在，生成一个 BA 网络
        logger.warning(f"未找到 {gml_path}，生成 BA 网络")
        n = int(dataset_name.split('-')[-1]) if '-' in dataset_name else 100
        G = nx.barabasi_albert_graph(n, 4)
    
    return G


def construct_graph(G, node_num, edge_num, cover_rate, num_hubs=None, seed=42, use_adaptive=False):
    """
    生成对比拓扑。
    
    Args:
        G: 原始图
        node_num: 节点数
        edge_num: 边数
        cover_rate: 覆盖率
        num_hubs: 双峰(Bimodal) 中 hub 节点数量，None 时使用默认策略
        seed: 双峰构造随机种子
        use_adaptive: 是否使用自适应 hub_num 搜索（优化 R_weight）
    """
    # 1. BA Network
    logger.info(f"  构建 BA Network...")
    G_BA = construct_ba(node_num, edge_num)
    logger.info(f"  BA Network 构建完成: {G_BA.number_of_nodes()} 节点, {G_BA.number_of_edges()} 边")
    # 2. RA Network (Random)
    logger.info(f"  构建 RA Network (Random)...")
    G_RA = construct_random(node_num, edge_num)
    logger.info(f"  RA Network (Random) 构建完成: {G_RA.number_of_nodes()} 节点, {G_RA.number_of_edges()} 边")
    # 3. ONION
    logger.info(f"  构建 ONION 优化网络...")
    G_ONION = nc.construct(G, algorithm='Onion', max_iter=100)
    logger.info(f"  ONION 构建完成: {G_ONION.number_of_nodes()} 节点, {G_ONION.number_of_edges()} 边")
    # 4. ROMEN
    logger.info(f"  构建 ROMEN 优化网络...")
    G_ROMEM = nc.construct(G, algorithm='ROMEN')
    logger.info(f"  ROMEN 构建完成: {G_ROMEM.number_of_nodes()} 节点, {G_ROMEM.number_of_edges()} 边")
    # 5. UNITY
    logger.info(f"  构建 UNITY 优化网络...")
    G_UNITY = nc.construct(G, algorithm='UNITY')
    logger.info(f"  UNITY 构建完成: {G_UNITY.number_of_nodes()} 节点, {G_UNITY.number_of_edges()} 边")
    # 6. FRED-ABL
    logger.info(f"  构建 FRED-ABL 优化网络...")
    G_FRED_ABL = nc.construct(G, algorithm='FRED_ABL', iterations=15, initial_samples=3, verbose=False)
    logger.info(f"  FRED-ABL 构建完成: {G_FRED_ABL.number_of_nodes()} 节点, {G_FRED_ABL.number_of_edges()} 边")
    # 7. QDLM
    logger.info(f"  构建 QDLM 优化网络...")
    G_QDLM = nc.construct(G, algorithm='QDLM', pop_size=20, max_iterations=25, verbose=False)
    logger.info(f"  QDLM 构建完成: {G_QDLM.number_of_nodes()} 节点, {G_QDLM.number_of_edges()} 边")
    
    # 8. Bimodal（支持自适应搜索最佳 hub_num）
    if use_adaptive:
        logger.info(f"  构建 Bimodal 网络（自适应搜索最佳 hub_num）...")
        G_BiT = create_bimodal_adaptive(G, seed=seed, verbose=True)
        logger.info(f"  Bimodal 自适应结果: {G_BiT.number_of_nodes()} 节点, {G_BiT.number_of_edges()} 边")
    elif num_hubs is not None:
        logger.info(f"  构建 Bimodal 网络 (num_hubs={num_hubs})...")
        G_BiT = create_bimodal_with_num_hubs(node_num, edge_num, num_hubs, seed=seed)
    else:
        logger.info(f"  构建 Bimodal 网络（理论分布）...")
        G_BiT = create_bimodal_theoretical(node_num, edge_num, seed=seed)
    
    # 将拓扑组织成字典（按指定顺序）
    all_G = {
        "Original": G,
        "BA Network": G_BA,                    # 1. BA Network
        "RA Network": G_RA,            # 2. RA Network (Random)
        "ONION": G_ONION,              # 3. ONION
        "ROMEN": G_ROMEM,              # 4. ROMEN
        "UNITY": G_UNITY,               # 5. UNITY
        "FRED-ABL": G_FRED_ABL,        # 6. FRED-ABL
        "QDLM": G_QDLM,                # 7. QDLM
        "Bimodal": G_BiT,              # 8. Bimodal
    }
    return all_G


def get_attack_sequence(G, attack_mode):
    """获取攻击序列"""
    if attack_mode == "degree":
        return node_attack_degree(G)
    elif attack_mode == "random":
        return node_attack_random(G)
    else:
        raise ValueError(f"Invalid attack mode: {attack_mode}")


def simulate(all_G, attack_mode):
    """模拟攻击过程"""
    x_results = {}
    y_results = {}
    
    # #region agent log - 开始模拟攻击
    import json, os, time as time_module
    debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
    try:
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "id": f"log_{int(time_module.time() * 1000)}_simulate_start",
                "timestamp": int(time_module.time() * 1000),
                "location": "compare_single_layer.py:simulate",
                "message": f"Starting attack simulation: {attack_mode}",
                "hypothesisId": "H5",
                "sessionId": "debug-session",
                "runId": "run1",
                "data": {
                    "attack_mode": attack_mode,
                    "networks": [name for name, g in all_G.items() if g is not None]
                }
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception:
        pass
    # #endregion
    
    for name, G in all_G.items():
        if G is None:
            logger.warning(f"  跳过 {name}（图为空）")
            continue
        logger.info(f"  模拟攻击 {name}...")
        
        # #region agent log - 开始攻击特定网络
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                log_entry = {
                    "id": f"log_{int(time_module.time() * 1000)}_attack_{name}",
                    "timestamp": int(time_module.time() * 1000),
                    "location": "compare_single_layer.py:simulate",
                    "message": f"Attacking network: {name}",
                    "hypothesisId": "H5",
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "data": {
                        "network_name": name,
                        "attack_mode": attack_mode,
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "is_connected": nx.is_connected(G),
                        "is_bimodal": name == "Bimodal"
                    }
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception:
            pass
        # #endregion
        
        sequence = get_attack_sequence(G, attack_mode)
        x_results[name], y_results[name] = simulate_attack(G, sequence)
        
    return x_results, y_results


def simulate_attack(G, sequence):
    """
    模拟攻击过程
    
    Args:
        G: NetworkX 图
        sequence: 攻击节点序列 (节点列表)
        
    Returns:
        x_results: 攻击步数列表
        y_results: GCC 比例列表
    """
    # #region agent log - 攻击模拟开始
    import json, os, time as time_module
    debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
    try:
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "id": f"log_{int(time_module.time() * 1000)}_attack_start",
                "timestamp": int(time_module.time() * 1000),
                "location": "compare_single_layer.py:simulate_attack",
                "message": "Attack simulation started",
                "hypothesisId": "H3",
                "sessionId": "debug-session",
                "runId": "run1",
                "data": {
                    "initial_nodes": G.number_of_nodes(),
                    "initial_edges": G.number_of_edges(),
                    "sequence_length": len(sequence),
                    "is_connected": nx.is_connected(G)
                }
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception:
        pass
    # #endregion
    
    G = G.copy()
    initial_nodes = G.number_of_nodes()
    
    x_results = [0.0]  # 初始状态，attack_step/initial_nodes = 0
    y_results = [1.0]  # 初始 GCC 比例为 1
    
    attack_step = 0
    for node in sequence:
        attack_step += 1  # 攻击步数递增，对应 sequence 的索引位置
        
        if node not in G.nodes():
            # 如果节点不在图中，保持之前的 y 值
            x = attack_step / initial_nodes
            y = y_results[-1]  # 保持之前的 GCC 值
            x_results.append(x)
            y_results.append(y)
        else:
            # 移除节点及其所有相邻的边
            # 注意：NetworkX 的 remove_node() 会自动移除节点和所有与其相连的边
            G.remove_node(node)
            
            # 计算指标
            x, y = calculate_metrics(G, initial_nodes, attack_step)
            x_results.append(x)
            y_results.append(y)
        
        # #region agent log - 攻击步骤详情（每10步记录一次，或关键步骤）
        if attack_step <= 5 or attack_step % 10 == 0 or y <= 0.1:
            try:
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    components = list(nx.connected_components(G))
                    gcc_size = max(len(c) for c in components) if components else 0
                    log_entry = {
                        "id": f"log_{int(time_module.time() * 1000)}_step{attack_step}",
                        "timestamp": int(time_module.time() * 1000),
                        "location": "compare_single_layer.py:simulate_attack",
                        "message": f"Attack step {attack_step}",
                        "hypothesisId": "H3",
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "data": {
                            "attack_step": attack_step,
                            "remaining_nodes": G.number_of_nodes(),
                            "remaining_edges": G.number_of_edges(),
                            "gcc_size": gcc_size,
                            "gcc_ratio": y,
                            "num_components": len(components),
                            "prev_gcc_ratio": y_results[-2] if len(y_results) > 1 else 1.0,
                            "gcc_decreasing": y <= (y_results[-2] if len(y_results) > 1 else 1.0)
                        }
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception:
                pass
        # #endregion
        
        # 如果 GCC 已经为 0，停止
        if y <= 0:
            break
    
    # #region agent log - 攻击模拟结束
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "id": f"log_{int(time_module.time() * 1000)}_attack_end",
                "timestamp": int(time_module.time() * 1000),
                "location": "compare_single_layer.py:simulate_attack",
                "message": "Attack simulation ended",
                "hypothesisId": "H3",
                "sessionId": "debug-session",
                "runId": "run1",
                "data": {
                    "total_steps": attack_step,
                    "final_nodes": G.number_of_nodes(),
                    "x_results_length": len(x_results),
                    "y_results_length": len(y_results),
                    "final_gcc": y_results[-1] if y_results else 0,
                    "monotonic_decrease": all(y_results[i] >= y_results[i+1] for i in range(len(y_results)-1))
                }
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception:
        pass
    # #endregion
    
    return x_results, y_results


def calculate_metrics(G, initial_nodes, attack_step):
    """计算攻击后的指标"""
    if G.number_of_nodes() == 0:
        return attack_step / initial_nodes, 0.0
    
    gcc = calculate_gcc(G, initial_nodes)
    x = attack_step / initial_nodes  # 横坐标为 attack_step/initial_nodes
    return x, gcc


def calculate_gcc(G, initial_nodes):
    """
    计算 GCC 比例
    
    Returns:
        GCC 大小 / 初始节点数
    """
    if G.number_of_nodes() == 0:
        return 0.0
    
    components = list(nx.connected_components(G))
    if not components:
        return 0.0
    
    max_component_size = max(len(c) for c in components)
    return max_component_size / initial_nodes


def calculate_r_value(x_results, y_results, initial_nodes=None):
    """
    计算 R 值（曲线下面积）
    
    使用梯形法则计算 AUC
    
    Args:
        x_results: x 坐标列表（attack_step/initial_nodes，已归一化）
        y_results: y 坐标列表（GCC 比例）
        initial_nodes: 初始节点数（保留参数以兼容旧代码，但不再使用）
        
    Returns:
        float: R 值（由于 x 坐标已归一化，直接返回 AUC）
    """
    if len(x_results) < 2:
        return 0.0
    
    r_value = 0.0
    for i in range(len(x_results) - 1):
        dx = x_results[i + 1] - x_results[i]
        avg_y = (y_results[i] + y_results[i + 1]) / 2
        r_value += dx * avg_y
    
    # x 坐标已经是归一化的（attack_step/initial_nodes），所以不需要再次归一化
    return r_value


def node_attack_degree(G):
    """
    按度数降序返回攻击节点序列
    
    Returns:
        节点列表（按度数从高到低排序）
    """
    # #region agent log - 检查 degree attack 序列
    import json, os, time as time_module
    debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
    try:
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        degree_list = list(G.degree())
        degree_list.sort(key=lambda x: x[1], reverse=True)
        top5_degrees = [(n, d) for n, d in degree_list[:5]]
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "id": f"log_{int(time_module.time() * 1000)}_degree_seq",
                "timestamp": int(time_module.time() * 1000),
                "location": "compare_single_layer.py:node_attack_degree",
                "message": "Degree attack sequence generated",
                "hypothesisId": "H4",
                "sessionId": "debug-session",
                "runId": "run1",
                "data": {
                    "total_nodes": len(degree_list),
                    "top5_degrees": top5_degrees,
                    "max_degree": degree_list[0][1] if degree_list else 0,
                    "min_degree": degree_list[-1][1] if degree_list else 0,
                    "avg_degree": sum(d for _, d in degree_list) / len(degree_list) if degree_list else 0
                }
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        return [node for node, degree in degree_list]
    except Exception:
        degree_list = list(G.degree())
        degree_list.sort(key=lambda x: x[1], reverse=True)
        return [node for node, degree in degree_list]
    # #endregion


def node_attack_random(G):
    """随机攻击序列"""
    nodes = list(G.nodes())
    random.shuffle(nodes)
    return nodes


def plot_results(dataset_name, x_results, y_results, attack_mode="degree", save_dir=None, initial_nodes=None):
    """
    绘制攻击曲线并计算 R 值。
    save_dir: 保存目录，若为 None 则使用 results。
    initial_nodes: 初始节点数，用于 R 值归一化
    """
    if save_dir is None:
        save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算每个拓扑的 R 值（归一化）
    r_values = {}
    for name in x_results.keys():
        r_values[name] = calculate_r_value(x_results[name], y_results[name], initial_nodes=initial_nodes)
    
    # 按 R 值排序
    sorted_names = sorted(r_values.keys(), key=lambda x: r_values[x], reverse=True)
    
    # 颜色映射（优化后的配色方案）
    colors = {
        'BA': '#2E86AB',           # 深蓝色
        'RA Network': '#A23B72',   # 深紫红色
        'ONION': '#F18F01',        # 橙色
        'ROMEN': '#C73E1D',        # 深红色
        'UNITY': '#6A994E',        # 绿色
        'FRED-ABL': '#BC4749',     # 红棕色
        'QDLM': '#7209B7',         # 紫色
        'Bimodal': '#06A77D',      # 青绿色
        # 其它方法
        'Original': '#495057',     # 深灰色
        'GA': '#D62828',           # 红色
    }
    
    # 线型映射
    linestyles = {
        'BA': '-',
        'RA Network': '--',
        'ONION': '-.',
        'ROMEN': ':',
        'UNITY': '-',
        'FRED-ABL': '--',
        'QDLM': '-.',
        'Bimodal': '-',
        # 其它方法
        'Original': ':',
        'GA': '--',
    }
    
    # 绘图
    plt.figure(figsize=(12, 7))
    
    # 确定 x 轴范围（最大攻击步数）
    max_attack_step = 0
    for name in sorted_names:
        if x_results[name]:
            max_attack_step = max(max_attack_step, max(x_results[name]))
    
    for name in sorted_names:
        x_result = x_results[name]
        y_result = y_results[name]
        r_val = r_values[name]
        
        color = colors.get(name, '#888888')
        linestyle = linestyles.get(name, '-')
        linewidth = 2.5 if name == 'Bimodal' else 2.0
        
        label = f"{name} (R={r_val:.4f})"
        plt.plot(x_result, y_result, label=label,
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
    
    plt.xlabel('Attack Step / Initial Nodes', fontsize=16, fontweight='bold')
    plt.ylabel('Relative Size of GCC', fontsize=16, fontweight='bold')
    plt.title(f'Robustness Comparison under {attack_mode.capitalize()} Attack\n({dataset_name})', 
              fontsize=17, fontweight='bold', pad=15)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linewidth=0.8)
    plt.xlim(0, max_attack_step if max_attack_step > 0 else 1)
    plt.ylim(0, 1.05)
    # 加粗加大刻度
    plt.tick_params(axis='both', which='major', labelsize=14, width=2.0, length=8)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.0)
    
    save_path = os.path.join(save_dir, f"comparison_single_layer_{dataset_name}_{attack_mode}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"图片已保存: {save_path}")
    plt.close()
    
    # 打印 R 值排名
    print("\n" + "="*50)
    print(f"R 值排名 ({dataset_name}, {attack_mode} attack)")
    print("="*50)
    for i, name in enumerate(sorted_names, 1):
        print(f"  {i}. {name:<12} R = {r_values[name]:.4f}")
    print("="*50 + "\n")
    
    return r_values


def main(dataset_name="BA-100", attack_mode="degree", cover_rate=0.1, hub_num=None, use_adaptive=False):
    """
    主函数

    Args:
        dataset_name: 数据集名称
        attack_mode: 攻击模式 ('degree' 或 'random')
        cover_rate: 覆盖率
        hub_num: 双峰(Bimodal) 的 num_hubs（hub 节点数），传入 create_bimodal_with_num_hubs；None 时默认 1
        use_adaptive: 是否启用自适应 hub_num 搜索
    """
    run_dir, run_id = get_next_run_dir()
    print("\n" + "="*60)
    print(f"单层网络拓扑对比实验   [运行编号: {run_id}]")
    print(f"结果目录: {run_dir}")
    print(f"攻击模式: {attack_mode}")
    if use_adaptive:
        print(f"Bimodal 模式: 自适应搜索最佳 hub_num")
    elif hub_num is not None:
        print(f"BiT-HyRL hub 节点数: {hub_num}")
    print("="*60)
    
    # 加载图
    logger.info(f"加载图: {dataset_name}")
    G = load_or_generate_graph(dataset_name)
    logger.info(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 构建对比拓扑
    logger.info(f"构建对比拓扑...")
    all_G = construct_graph(G, G.number_of_nodes(), G.number_of_edges(), cover_rate, num_hubs=hub_num, use_adaptive=use_adaptive)
    
    # 打印各拓扑信息
    print("\n拓扑信息:")
    for name, g in all_G.items():
        if g is not None:
            print(f"  {name:<12}: {g.number_of_nodes():>4} 节点, {g.number_of_edges():>5} 边")
            degrees = [d for _, d in g.degree()]
            print(f"               度: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/len(degrees):.1f}")
    
    # 各算法度分布特征（简单概括）
    print_degree_distribution_features(all_G)
    
    # 模拟攻击
    logger.info(f"模拟 {attack_mode} 攻击...")
    x_results, y_results = simulate(all_G, attack_mode)
    
    # 获取初始节点数（从第一个非空图）
    initial_nodes = None
    for g in all_G.values():
        if g is not None:
            initial_nodes = g.number_of_nodes()
            break
    
    # 绘制结果（保存到 run_dir）
    logger.info(f"绘制结果...")
    r_values = plot_results(dataset_name, x_results, y_results, attack_mode, save_dir=run_dir, initial_nodes=initial_nodes)
    
    # 所有方法的度分布图
    plot_all_degree_distribution(all_G, run_dir, dataset_name)
    
    print("完成!")
    return r_values


def run_both_attacks(dataset_name="BA-100", cover_rate=0.1, hub_num=None, use_adaptive=False, 
                     batch_size=None, seed=None):
    """
    运行 degree 和 random 两种攻击模式，并计算加权指标 R_res

    R_res = 0.5 * R_random + 0.5 * R_degree (综合鲁棒性指标)
    hub_num: 双峰(Bimodal) 的 num_hubs（hub 节点数），传入 create_bimodal_with_num_hubs；None 时默认 1。
    use_adaptive: 是否启用自适应 hub_num 搜索
    batch_size: 如果提供，运行多轮 batch 实验并收集 R_res 数据用于箱线图
    seed: 随机种子（用于 batch 模式）
    """
    run_dir, run_id = get_next_run_dir()
    print("\n" + "="*70)
    print(f"  对比 Degree Attack 和 Random Attack   [运行编号: {run_id}]")
    print(f"  结果目录: {run_dir}")
    if use_adaptive:
        print(f"  Bimodal 模式: 自适应搜索最佳 hub_num")
    elif hub_num is not None:
        print(f"  BiT-HyRL hub 节点数: {hub_num}")
    print("="*70)

    # 加载图并构建拓扑（只构建一次）
    logger.info(f"加载图: {dataset_name}")
    G = load_or_generate_graph(dataset_name)
    logger.info(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    logger.info(f"构建对比拓扑...")
    all_G = construct_graph(G, G.number_of_nodes(), G.number_of_edges(), cover_rate, num_hubs=hub_num, use_adaptive=use_adaptive)
    
    # 打印各拓扑信息
    print("\n拓扑信息:")
    for name, g in all_G.items():
        if g is not None:
            degrees = [d for _, d in g.degree()]
            print(f"  {name:<12}: {g.number_of_nodes():>4} 节点, {g.number_of_edges():>5} 边, "
                  f"度: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/len(degrees):.1f}")
    
    # 各算法度分布特征（简单概括）
    print_degree_distribution_features(all_G)
    
    # 获取初始节点数（从第一个非空图）
    initial_nodes = None
    for g in all_G.values():
        if g is not None:
            initial_nodes = g.number_of_nodes()
            break
    
    # Degree attack
    print("\n" + "-"*50)
    logger.info("模拟 Degree Attack...")
    x_degree, y_degree = simulate(all_G, "degree")
    r_degree = {}
    for name in x_degree.keys():
        r_degree[name] = calculate_r_value(x_degree[name], y_degree[name], initial_nodes=initial_nodes)
    
    # Random attack
    logger.info("模拟 Random Attack...")
    x_random, y_random = simulate(all_G, "random")
    r_random = {}
    for name in x_random.keys():
        r_random[name] = calculate_r_value(x_random[name], y_random[name], initial_nodes=initial_nodes)
    
    # 计算加权指标 R_res = 0.5 * R_random + 0.5 * R_degree
    r_res = {}
    for name in r_degree.keys():
        r_res[name] = 0.5 * r_random.get(name, 0) + 0.5 * r_degree.get(name, 0)
    
    # 如果启用 batch 模式，运行多轮实验收集 R_res 数据和曲线数据
    batch_r_res_data = None
    batch_curves_degree = None  # {name: [x_list1, x_list2, ...], [y_list1, y_list2, ...]}
    batch_curves_random = None
    batch_r_degree_data = None  # {name: [r1, r2, ...]}
    batch_r_random_data = None
    
    if batch_size is not None and batch_size > 1:
        print(f"\n运行 {batch_size} 轮 batch 实验收集数据...")
        batch_r_res_data = {}
        batch_curves_degree = {}
        batch_curves_random = {}
        batch_r_degree_data = {}
        batch_r_random_data = {}
        
        for batch_idx in range(batch_size):
            if seed is not None:
                random.seed(seed + batch_idx)
                np.random.seed(seed + batch_idx)
            
            # 构建拓扑（每次可能因为随机性略有不同）
            bimodal_seed = (seed + batch_idx) if seed is not None else None
            all_G_batch = construct_graph(G, G.number_of_nodes(), G.number_of_edges(), 
                                          cover_rate, num_hubs=hub_num, seed=bimodal_seed, 
                                          use_adaptive=use_adaptive)
            
            # 获取初始节点数
            initial_nodes_batch = None
            for g in all_G_batch.values():
                if g is not None:
                    initial_nodes_batch = g.number_of_nodes()
                    break
            
            # Degree attack
            x_degree_batch, y_degree_batch = simulate(all_G_batch, "degree")
            r_degree_batch = {}
            for name in x_degree_batch.keys():
                r_degree_batch[name] = calculate_r_value(x_degree_batch[name], y_degree_batch[name], 
                                                       initial_nodes=initial_nodes_batch)
                # 收集曲线数据
                if name not in batch_curves_degree:
                    batch_curves_degree[name] = {'x': [], 'y': []}
                batch_curves_degree[name]['x'].append(x_degree_batch[name])
                batch_curves_degree[name]['y'].append(y_degree_batch[name])
                # 收集R值数据
                if name not in batch_r_degree_data:
                    batch_r_degree_data[name] = []
                batch_r_degree_data[name].append(r_degree_batch[name])
            
            # Random attack
            x_random_batch, y_random_batch = simulate(all_G_batch, "random")
            r_random_batch = {}
            for name in x_random_batch.keys():
                r_random_batch[name] = calculate_r_value(x_random_batch[name], y_random_batch[name], 
                                                        initial_nodes=initial_nodes_batch)
                # 收集曲线数据
                if name not in batch_curves_random:
                    batch_curves_random[name] = {'x': [], 'y': []}
                batch_curves_random[name]['x'].append(x_random_batch[name])
                batch_curves_random[name]['y'].append(y_random_batch[name])
                # 收集R值数据
                if name not in batch_r_random_data:
                    batch_r_random_data[name] = []
                batch_r_random_data[name].append(r_random_batch[name])
            
            # 计算每轮的 R_res
            for name in r_degree_batch.keys():
                if name not in batch_r_res_data:
                    batch_r_res_data[name] = []
                r_res_batch = 0.5 * r_random_batch.get(name, 0) + 0.5 * r_degree_batch.get(name, 0)
                batch_r_res_data[name].append(r_res_batch)
        
        print(f"Batch 数据收集完成，每个算法有 {batch_size} 个 R_res 值")
    
    # 按 R_res 排序
    sorted_names = sorted(r_res.keys(), key=lambda x: r_res[x], reverse=True)
    
    # 打印对比总结
    print("\n" + "="*70)
    print("  攻击模式对比总结")
    print("="*70)
    print(f"\n{'排名':<4} {'拓扑':<12} {'R_tar(Degree)':<14} {'R_ran(Random)':<14} {'R_res(加权)':<12}")
    print("-"*60)
    
    for i, name in enumerate(sorted_names, 1):
        d_r = r_degree[name]
        r_r = r_random.get(name, 0)
        res = r_res[name]
        print(f"  {i:<2}  {name:<12} {d_r:<14.4f} {r_r:<14.4f} {res:<12.4f}")
    
    print("-"*60)
    print(f"\n公式: R_res = 0.5 × R_tar + 0.5 × R_ran")
    print("      R_tar = Targeted Attack (Degree Attack) R值")
    print("      R_ran = Random Attack R值")
    if batch_r_res_data:
        print(f"      Batch 模式: {batch_size} 轮实验")
    
    # 绘制综合对比图（保存到 run_dir），并绘制所有方法度分布图
    plot_comparison_results(dataset_name, r_degree, r_random, r_res,
                           x_degree, y_degree, x_random, y_random, sorted_names,
                           all_G=all_G, save_dir=run_dir, 
                           batch_r_res_data=batch_r_res_data,
                           batch_curves_degree=batch_curves_degree,
                           batch_curves_random=batch_curves_random,
                           batch_r_degree_data=batch_r_degree_data,
                           batch_r_random_data=batch_r_random_data,
                           batch_size=batch_size)
    
    return r_degree, r_random, r_res


def _compute_average_curve(x_lists, y_lists):
    """
    计算多条曲线的平均曲线
    
    Args:
        x_lists: x坐标列表的列表 [[x1, x2, ...], [x1, x2, ...], ...]
        y_lists: y坐标列表的列表 [[y1, y2, ...], [y1, y2, ...], ...]
    
    Returns:
        x_avg, y_avg: 平均曲线的x和y坐标
    """
    if not x_lists or not y_lists:
        return [], []
    
    # 找到所有唯一的x值并排序
    all_x = set()
    for x_list in x_lists:
        all_x.update(x_list)
    x_avg = sorted(all_x)
    
    # 对每个x值，计算所有曲线在该x值的y值的平均值
    y_avg = []
    for x_val in x_avg:
        y_values = []
        for i, x_list in enumerate(x_lists):
            if x_val in x_list:
                idx = x_list.index(x_val)
                y_values.append(y_lists[i][idx])
        if y_values:
            y_avg.append(np.mean(y_values))
        else:
            y_avg.append(0.0)
    
    return x_avg, y_avg


def plot_comparison_results(dataset_name, r_degree, r_random, r_res,
                           x_degree, y_degree, x_random, y_random, sorted_names,
                           all_G=None, save_dir=None, batch_r_res_data=None,
                           batch_curves_degree=None, batch_curves_random=None,
                           batch_r_degree_data=None, batch_r_random_data=None,
                           batch_size=None):
    """
    绘制综合对比图。
    all_G: 各算法图字典，用于绘制前3名度分布图；若为 None 则不画度分布图。
    save_dir: 保存目录，若为 None 则使用 results。
    batch_r_res_data: 如果提供，子图3将显示 R_res 的箱线图而不是柱状图。
                      格式: {algorithm_name: [r_res_value1, r_res_value2, ...]}
    batch_curves_degree: batch模式下的degree攻击曲线数据
    batch_curves_random: batch模式下的random攻击曲线数据
    batch_r_degree_data: batch模式下的R_degree数据
    batch_r_random_data: batch模式下的R_random数据
    batch_size: batch大小
    """
    if save_dir is None:
        save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 颜色映射（优化后的配色方案）
    colors = {
        'BA': '#2E86AB',           # 深蓝色
        'RA Network': '#A23B72',   # 深紫红色
        'ONION': '#F18F01',        # 橙色
        'ROMEN': '#C73E1D',        # 深红色
        'UNITY': '#6A994E',        # 绿色
        'FRED-ABL': '#BC4749',     # 红棕色
        'QDLM': '#7209B7',         # 紫色
        'Bimodal': '#06A77D',      # 青绿色
        # 其它方法
        'Original': '#495057',     # 深灰色
        'GA': '#D62828',           # 红色
    }
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== 子图1: Degree Attack 曲线 =====
    ax1 = axes[0, 0]
    max_attack_step_degree = 0
    
    # 如果有batch数据，计算平均曲线；否则使用单次结果
    if batch_curves_degree and batch_size:
        for name in sorted_names:
            if name in batch_curves_degree:
                x_lists = batch_curves_degree[name]['x']
                y_lists = batch_curves_degree[name]['y']
                x_avg, y_avg = _compute_average_curve(x_lists, y_lists)
                
                if x_avg and y_avg:
                    color = colors.get(name, '#888888')
                    lw = 2.5 if name == 'Bimodal' else 2.0
                    r_avg = np.mean(batch_r_degree_data.get(name, [r_degree.get(name, 0)]))
                    label = f"{name} (R={r_avg:.4f})"
                    ax1.plot(x_avg, y_avg, label=label, color=color, linewidth=lw, alpha=0.9)
                    if x_avg:
                        max_attack_step_degree = max(max_attack_step_degree, max(x_avg))
    else:
        # 单次结果
        for name in sorted_names:
            if name in x_degree:
                color = colors.get(name, '#888888')
                lw = 2.5 if name == 'Bimodal' else 2.0
                label = f"{name} (R={r_degree[name]:.4f})"
                ax1.plot(x_degree[name], y_degree[name], label=label, color=color, linewidth=lw, alpha=0.9)
                if x_degree[name]:
                    max_attack_step_degree = max(max_attack_step_degree, max(x_degree[name]))
    
    ax1.set_xlabel('Attack Step / Initial Nodes', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Relative Size of GCC', fontsize=15, fontweight='bold')
    title_a = '(a) Targeted Attack' if batch_size else '(a) Degree Attack (Targeted)'
    ax1.set_title(title_a, fontsize=15, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_xlim(0, max_attack_step_degree if max_attack_step_degree > 0 else 1)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='both', which='major', labelsize=13, width=2.0, length=8)
    ax1.tick_params(axis='both', which='minor', labelsize=11)
    for spine in ax1.spines.values():
        spine.set_linewidth(2.0)
    
    # ===== 子图2: Random Attack 曲线 =====
    ax2 = axes[0, 1]
    max_attack_step_random = 0
    
    # 如果有batch数据，计算平均曲线；否则使用单次结果
    if batch_curves_random and batch_size:
        for name in sorted_names:
            if name in batch_curves_random:
                x_lists = batch_curves_random[name]['x']
                y_lists = batch_curves_random[name]['y']
                x_avg, y_avg = _compute_average_curve(x_lists, y_lists)
                
                if x_avg and y_avg:
                    color = colors.get(name, '#888888')
                    lw = 2.5 if name == 'Bimodal' else 2.0
                    r_avg = np.mean(batch_r_random_data.get(name, [r_random.get(name, 0)]))
                    label = f"{name} (R={r_avg:.4f})"
                    ax2.plot(x_avg, y_avg, label=label, color=color, linewidth=lw, alpha=0.9)
                    if x_avg:
                        max_attack_step_random = max(max_attack_step_random, max(x_avg))
    else:
        # 单次结果
        for name in sorted_names:
            if name in x_random:
                color = colors.get(name, '#888888')
                lw = 2.5 if name == 'Bimodal' else 2.0
                label = f"{name} (R={r_random[name]:.4f})"
                ax2.plot(x_random[name], y_random[name], label=label, color=color, linewidth=lw, alpha=0.9)
                if x_random[name]:
                    max_attack_step_random = max(max_attack_step_random, max(x_random[name]))
    
    ax2.set_xlabel('Attack Step / Initial Nodes', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Relative Size of GCC', fontsize=15, fontweight='bold')
    ax2.set_title('(b) Random Attack', fontsize=15, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linewidth=0.8)
    ax2.set_xlim(0, max_attack_step_random if max_attack_step_random > 0 else 1)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='both', which='major', labelsize=13, width=2.0, length=8)
    ax2.tick_params(axis='both', which='minor', labelsize=11)
    for spine in ax2.spines.values():
        spine.set_linewidth(2.0)
    
    # ===== 子图3: R值对比（batch模式为箱线图，否则为柱状图）=====
    ax3 = axes[1, 0]
    x_pos = range(len(sorted_names))
    
    if batch_r_res_data and batch_size:
        # Batch模式：显示R_res的箱线图
        box_data = [batch_r_res_data.get(name, []) for name in sorted_names]
        bp = ax3.boxplot(box_data, positions=x_pos, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=False,
                        boxprops=dict(facecolor='#06A77D', alpha=0.7, linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        ax3.set_xlabel('Topology', fontsize=15, fontweight='bold')
        ax3.set_ylabel('R_res (Weighted)', fontsize=15, fontweight='bold')
        ax3.set_title(f'(c) R_res (Weighted) Boxplot (batch_size={batch_size})', fontsize=15, fontweight='bold', pad=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y', linewidth=0.8)
        
        # 设置y轴范围
        all_values = []
        for data in box_data:
            all_values.extend(data)
        if all_values:
            ax3.set_ylim(0, max(all_values) * 1.15)
    else:
        # 单次模式：显示柱状图
        width = 0.25
        bars1 = ax3.bar([p - width for p in x_pos], [r_degree[n] for n in sorted_names], 
                        width, label='R_tar (Degree)', color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax3.bar(x_pos, [r_random[n] for n in sorted_names], 
                        width, label='R_ran (Random)', color='#F18F01', alpha=0.85, edgecolor='black', linewidth=1.2)
        bars3 = ax3.bar([p + width for p in x_pos], [r_res[n] for n in sorted_names], 
                        width, label='R_res (Weighted)', color='#06A77D', alpha=0.85, edgecolor='black', linewidth=1.2)
        
        ax3.set_xlabel('Topology', fontsize=15, fontweight='bold')
        ax3.set_ylabel('R Value', fontsize=15, fontweight='bold')
        ax3.set_title('(c) R Value Comparison', fontsize=15, fontweight='bold', pad=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3, axis='y', linewidth=0.8)
        ax3.set_ylim(0, max(max(r_random.values()), max(r_degree.values())) * 1.15)
    
    ax3.tick_params(axis='both', which='major', labelsize=13, width=2.0, length=8)
    ax3.tick_params(axis='both', which='minor', labelsize=11)
    for spine in ax3.spines.values():
        spine.set_linewidth(2.0)
    
    # ===== 子图4: R_res 折线图排名 =====
    ax4 = axes[1, 1]
    
    # 如果有batch数据，计算平均R值；否则使用单次结果
    if batch_r_res_data and batch_size and batch_r_degree_data and batch_r_random_data:
        # Batch模式：使用平均R值
        r_res_sorted = [np.mean(batch_r_res_data.get(n, [r_res.get(n, 0)])) for n in sorted_names]
        r_tar_sorted = [np.mean(batch_r_degree_data.get(n, [r_degree.get(n, 0)])) for n in sorted_names]
        r_ran_sorted = [np.mean(batch_r_random_data.get(n, [r_random.get(n, 0)])) for n in sorted_names]
    else:
        # 单次模式：使用单次R值
        r_res_sorted = [r_res[n] for n in sorted_names]
        r_tar_sorted = [r_degree[n] for n in sorted_names]
        r_ran_sorted = [r_random[n] for n in sorted_names]
    
    x_line = range(len(sorted_names))
    
    ax4.plot(x_line, r_tar_sorted, 'o-', label='R_tar (Degree)', color='#2E86AB', 
             linewidth=2.5, markersize=9, alpha=0.9, markerfacecolor='white', markeredgewidth=2)
    ax4.plot(x_line, r_ran_sorted, 's-', label='R_ran (Random)', color='#F18F01', 
             linewidth=2.5, markersize=9, alpha=0.9, markerfacecolor='white', markeredgewidth=2)
    ax4.plot(x_line, r_res_sorted, '^-', label='R_res (Weighted)', color='#06A77D', 
             linewidth=3, markersize=11, alpha=0.9, markerfacecolor='white', markeredgewidth=2)
    
    # 标注 R_res 值
    for i, (name, val) in enumerate(zip(sorted_names, r_res_sorted)):
        ax4.annotate(f'{val:.3f}', (i, val), textcoords="offset points", 
                    xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Topology (Sorted by R_res)', fontsize=15, fontweight='bold')
    ax4.set_ylabel('R Value', fontsize=15, fontweight='bold')
    title_d = f'(d) R Value Trend (Sorted by R_res, batch_size={batch_size})' if batch_size else '(d) R Value Trend (Sorted by R_res)'
    ax4.set_title(title_d, fontsize=15, fontweight='bold', pad=10)
    ax4.set_xticks(x_line)
    ax4.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, linewidth=0.8)
    max_r_val = max(max(r_tar_sorted), max(r_ran_sorted), max(r_res_sorted)) if r_res_sorted else 1.0
    ax4.set_ylim(0, max_r_val * 1.2)
    ax4.tick_params(axis='both', which='major', labelsize=13, width=2.0, length=8)
    ax4.tick_params(axis='both', which='minor', labelsize=11)
    for spine in ax4.spines.values():
        spine.set_linewidth(2.0)
    
    # 调整布局
    plt.suptitle(f'Robustness Comparison: {dataset_name}\n'
                 f'R_res = 0.5 x R_tar + 0.5 x R_ran',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"comparison_single_layer_{dataset_name}_combined.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"综合对比图已保存: {save_path}")
    plt.close()
    
    # 获取初始节点数（从 all_G 中）
    initial_nodes = None
    if all_G is not None:
        for g in all_G.values():
            if g is not None:
                initial_nodes = g.number_of_nodes()
                break
    
    # 同时保存单独的攻击曲线图到 save_dir
    plot_results(dataset_name, x_degree, y_degree, "degree", save_dir=save_dir, initial_nodes=initial_nodes)
    plot_results(dataset_name, x_random, y_random, "random", save_dir=save_dir, initial_nodes=initial_nodes)
    
    # 所有方法的度分布图
    if all_G is not None:
        plot_all_degree_distribution(all_G, save_dir, dataset_name)


def run_batch_comparison(dataset_name="BA-100", batch_size=10, cover_rate=0.1, seed=None, hub_num=None, use_adaptive=False):
    """
    运行多轮 batch_size 求平均值的对比实验
    
    对每种算法，运行 batch_size 次实验，计算：
    1. 每轮实验的 R 值（degree 和 random 两种攻击）
    2. 均值和方差
    3. 加权值（R_res = 0.5 * R_degree + 0.5 * R_random）
    
    Args:
        dataset_name: 数据集名称
        batch_size: 批次数（运行次数）
        cover_rate: 覆盖率
        seed: 随机种子（如果为None，则每次使用不同种子）
        hub_num: 双峰(Bimodal) 的 num_hubs（hub 节点数），传入 create_bimodal_with_num_hubs；None 时默认 1
        use_adaptive: 是否启用自适应 hub_num 搜索
        
    Returns:
        dict: 包含统计结果的字典
    """
    print("\n" + "="*70)
    print(f"  多轮 Batch 对比实验 (batch_size={batch_size})")
    if use_adaptive:
        print(f"  Bimodal 模式: 自适应搜索最佳 hub_num")
    elif hub_num is not None:
        print(f"  BiT-HyRL hub 节点数: {hub_num}")
    print("="*70)
    
    # 加载图（只加载一次）
    logger.info(f"加载图: {dataset_name}")
    G = load_or_generate_graph(dataset_name)
    logger.info(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 存储所有批次的 R 值
    # 结构: {algorithm_name: {'degree': [r1, r2, ...], 'random': [r1, r2, ...]}}
    all_r_values = {}
    
    # 运行 batch_size 次实验
    for batch_idx in range(batch_size):
        print(f"\n{'='*70}")
        print(f"  批次 {batch_idx + 1}/{batch_size}")
        print(f"{'='*70}")
        
        # 设置随机种子（如果提供了基础种子）
        if seed is not None:
            random.seed(seed + batch_idx)
            np.random.seed(seed + batch_idx)
        
        # 构建对比拓扑（每次可能因为随机性略有不同）
        bimodal_seed = (seed + batch_idx) if seed is not None else 42
        logger.info(f"构建对比拓扑 (批次 {batch_idx + 1})...")
        all_G = construct_graph(G, G.number_of_nodes(), G.number_of_edges(), cover_rate, num_hubs=hub_num, seed=bimodal_seed, use_adaptive=use_adaptive)
        
        # 获取初始节点数（从第一个非空图）
        initial_nodes = None
        for g in all_G.values():
            if g is not None:
                initial_nodes = g.number_of_nodes()
                break
        
        # Degree attack
        logger.info(f"模拟 Degree Attack (批次 {batch_idx + 1})...")
        x_degree, y_degree = simulate(all_G, "degree")
        r_degree_batch = {}
        for name in x_degree.keys():
            r_degree_batch[name] = calculate_r_value(x_degree[name], y_degree[name], initial_nodes=initial_nodes)
        
        # Random attack
        logger.info(f"模拟 Random Attack (批次 {batch_idx + 1})...")
        x_random, y_random = simulate(all_G, "random")
        r_random_batch = {}
        for name in x_random.keys():
            r_random_batch[name] = calculate_r_value(x_random[name], y_random[name], initial_nodes=initial_nodes)
        
        # 存储到总结果中
        for name in r_degree_batch.keys():
            if name not in all_r_values:
                all_r_values[name] = {'degree': [], 'random': []}
            all_r_values[name]['degree'].append(r_degree_batch[name])
            all_r_values[name]['random'].append(r_random_batch.get(name, 0))
    
    # 计算统计量
    statistics = {}
    for name in all_r_values.keys():
        degree_values = np.array(all_r_values[name]['degree'])
        random_values = np.array(all_r_values[name]['random'])
        
        # 计算均值和方差
        degree_mean = np.mean(degree_values)
        degree_std = np.std(degree_values, ddof=1)  # 样本标准差
        degree_var = np.var(degree_values, ddof=1)  # 样本方差
        
        random_mean = np.mean(random_values)
        random_std = np.std(random_values, ddof=1)
        random_var = np.var(random_values, ddof=1)
        
        # 计算加权值（每轮的加权值，然后求平均）
        weighted_values = 0.5 * degree_values + 0.5 * random_values
        weighted_mean = np.mean(weighted_values)
        weighted_std = np.std(weighted_values, ddof=1)
        weighted_var = np.var(weighted_values, ddof=1)
        
        statistics[name] = {
            'degree': {
                'values': degree_values.tolist(),
                'mean': degree_mean,
                'std': degree_std,
                'var': degree_var
            },
            'random': {
                'values': random_values.tolist(),
                'mean': random_mean,
                'std': random_std,
                'var': random_var
            },
            'weighted': {
                'values': weighted_values.tolist(),
                'mean': weighted_mean,
                'std': weighted_std,
                'var': weighted_var
            }
        }
    
    # 打印统计结果
    print("\n" + "="*100)
    print(f"  统计结果汇总 (batch_size={batch_size}, dataset={dataset_name})")
    print("="*100)
    
    # 按加权值均值排序
    sorted_names = sorted(statistics.keys(), 
                         key=lambda x: statistics[x]['weighted']['mean'], 
                         reverse=True)
    
    # 打印表头
    print(f"\n{'排名':<4} {'算法':<12} {'R_degree均值':<14} {'R_degree方差':<14} {'R_random均值':<14} "
          f"{'R_random方差':<14} {'R_weighted均值':<16} {'R_weighted方差':<16}")
    print("-"*100)
    
    for rank, name in enumerate(sorted_names, 1):
        stats = statistics[name]
        print(f"  {rank:<4} {name:<12} "
              f"{stats['degree']['mean']:<14.6f} {stats['degree']['var']:<14.6f} "
              f"{stats['random']['mean']:<14.6f} {stats['random']['var']:<14.6f} "
              f"{stats['weighted']['mean']:<16.6f} {stats['weighted']['var']:<16.6f}")
    
    print("-"*100)
    print(f"\n公式: R_weighted = 0.5 × R_degree + 0.5 × R_random")
    print(f"      所有统计量基于 {batch_size} 次独立实验")
    
    # 打印详细统计（包含标准差）
    print("\n" + "="*100)
    print(f"  详细统计信息（包含标准差）")
    print("="*100)
    print(f"\n{'算法':<12} {'R_degree (均值±标准差)':<25} {'R_random (均值±标准差)':<25} "
          f"{'R_weighted (均值±标准差)':<28}")
    print("-"*100)
    
    for name in sorted_names:
        stats = statistics[name]
        degree_str = f"{stats['degree']['mean']:.6f}±{stats['degree']['std']:.6f}"
        random_str = f"{stats['random']['mean']:.6f}±{stats['random']['std']:.6f}"
        weighted_str = f"{stats['weighted']['mean']:.6f}±{stats['weighted']['std']:.6f}"
        print(f"  {name:<12} {degree_str:<25} {random_str:<25} {weighted_str:<28}")
    
    print("-"*100)
    
    # 绘制统计结果图
    plot_batch_statistics(dataset_name, statistics, sorted_names, batch_size)
    
    return statistics


def plot_batch_statistics(dataset_name, statistics, sorted_names, batch_size):
    """
    绘制多轮 batch 统计结果图
    
    Args:
        dataset_name: 数据集名称
        statistics: 统计结果字典
        sorted_names: 按加权值排序的算法名称列表
        batch_size: 批次数
    """
    # 颜色映射
    colors = {
        'Original': '#1f77b4',
        'Random': '#ff7f0e', 
        'BA': '#2ca02c',
        'GA': '#d62728',
        'ONION': '#9467bd',
        'ROMEN': '#8c564b',
        'UNITY': '#e377c2',
        'Bimodal': '#17becf',
    }
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== 子图1: R_degree 均值和误差棒 =====
    ax1 = axes[0, 0]
    x_pos = range(len(sorted_names))
    degree_means = [statistics[name]['degree']['mean'] for name in sorted_names]
    degree_stds = [statistics[name]['degree']['std'] for name in sorted_names]
    
    bars1 = ax1.bar(x_pos, degree_means, yerr=degree_stds, capsize=5, 
                    color=[colors.get(name, '#888888') for name in sorted_names],
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R_degree Value', fontsize=14, fontweight='bold')
    ax1.set_title(f'(a) R_degree Mean ± Std (batch_size={batch_size})', fontsize=15, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=13, width=2.0, length=8)
    ax1.tick_params(axis='both', which='minor', labelsize=11)
    for spine in ax1.spines.values():
        spine.set_linewidth(2.0)
    
    # 标注数值
    for i, (mean, std) in enumerate(zip(degree_means, degree_stds)):
        ax1.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ===== 子图2: R_random 均值和误差棒 =====
    ax2 = axes[0, 1]
    random_means = [statistics[name]['random']['mean'] for name in sorted_names]
    random_stds = [statistics[name]['random']['std'] for name in sorted_names]
    
    bars2 = ax2.bar(x_pos, random_means, yerr=random_stds, capsize=5,
                    color=[colors.get(name, '#888888') for name in sorted_names],
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax2.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R_random Value', fontsize=14, fontweight='bold')
    ax2.set_title(f'(b) R_random Mean ± Std (batch_size={batch_size})', fontsize=15, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', which='major', labelsize=13, width=2.0, length=8)
    ax2.tick_params(axis='both', which='minor', labelsize=11)
    for spine in ax2.spines.values():
        spine.set_linewidth(2.0)
    
    # 标注数值
    for i, (mean, std) in enumerate(zip(random_means, random_stds)):
        ax2.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ===== 子图3: R_weighted 均值和误差棒 =====
    ax3 = axes[1, 0]
    weighted_means = [statistics[name]['weighted']['mean'] for name in sorted_names]
    weighted_stds = [statistics[name]['weighted']['std'] for name in sorted_names]
    
    bars3 = ax3.bar(x_pos, weighted_means, yerr=weighted_stds, capsize=5,
                    color=[colors.get(name, '#888888') for name in sorted_names],
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax3.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax3.set_ylabel('R_weighted Value', fontsize=14, fontweight='bold')
    ax3.set_title(f'(c) R_weighted Mean ± Std (batch_size={batch_size})', fontsize=15, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='both', which='major', labelsize=13, width=2.0, length=8)
    ax3.tick_params(axis='both', which='minor', labelsize=11)
    for spine in ax3.spines.values():
        spine.set_linewidth(2.0)
    
    # 标注数值
    for i, (mean, std) in enumerate(zip(weighted_means, weighted_stds)):
        ax3.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ===== 子图4: 三种 R 值对比折线图 =====
    ax4 = axes[1, 1]
    
    ax4.plot(x_pos, degree_means, 'o-', label='R_degree', color='#d62728', 
             linewidth=2, markersize=8, alpha=0.8)
    ax4.plot(x_pos, random_means, 's-', label='R_random', color='#2ca02c', 
             linewidth=2, markersize=8, alpha=0.8)
    ax4.plot(x_pos, weighted_means, '^-', label='R_weighted', color='#1f77b4', 
             linewidth=2.5, markersize=10, alpha=0.8)
    
    # 添加误差棒
    ax4.errorbar(x_pos, degree_means, yerr=degree_stds, fmt='none', 
                color='#d62728', alpha=0.5, capsize=3)
    ax4.errorbar(x_pos, random_means, yerr=random_stds, fmt='none', 
                color='#2ca02c', alpha=0.5, capsize=3)
    ax4.errorbar(x_pos, weighted_means, yerr=weighted_stds, fmt='none', 
                color='#1f77b4', alpha=0.5, capsize=3)
    
    ax4.set_xlabel('Algorithm (Sorted by R_weighted)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('R Value', fontsize=14, fontweight='bold')
    ax4.set_title(f'(d) R Values Comparison (batch_size={batch_size})', fontsize=15, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=13, width=2.0, length=8)
    ax4.tick_params(axis='both', which='minor', labelsize=11)
    for spine in ax4.spines.values():
        spine.set_linewidth(2.0)
    
    # 调整布局
    plt.suptitle(f'Batch Statistics Comparison: {dataset_name}\n'
                 f'R_weighted = 0.5 × R_degree + 0.5 × R_random (batch_size={batch_size})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs("results", exist_ok=True)
    save_path = f"results/batch_comparison_{dataset_name}_batch{batch_size}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"批次统计图已保存: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='单层网络拓扑对比实验')
    parser.add_argument('--dataset', type=str, default='BA-100', help='数据集名称')
    parser.add_argument('--attack', type=str, default=None, help='攻击模式 (degree/random)')
    parser.add_argument('--both', action='store_true', help='运行两种攻击模式对比')
    parser.add_argument('--batch', action='store_true', help='运行多轮 batch 对比实验')
    parser.add_argument('--batch_size', type=int, default=10, help='批次数（用于 --batch 模式）')
    parser.add_argument('--cover_rate', type=float, default=0.1, help='覆盖率')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（用于 --batch 模式）')
    parser.add_argument('--hub_num', '--num_hubs', type=int, default=None, dest='hub_num',
                    help='双峰(Bimodal) 的 num_hubs（hub 节点数），传入 create_bimodal_with_num_hubs；不指定时默认 1')
    parser.add_argument('--adaptive', action='store_true',
                    help='启用自适应 hub_num 搜索，自动找到使 R_weight 最大的 hub_num')
    
    args = parser.parse_args()
    
    if args.batch and args.both:
        # 运行两种攻击模式的batch对比实验（combined图）
        run_both_attacks(args.dataset, args.cover_rate, hub_num=args.hub_num, 
                        use_adaptive=args.adaptive, batch_size=args.batch_size, seed=args.seed)
    elif args.batch:
        # 运行多轮 batch 对比实验（单独的batch统计图）
        run_batch_comparison(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            cover_rate=args.cover_rate,
            seed=args.seed,
            hub_num=args.hub_num,
            use_adaptive=args.adaptive
        )
    elif args.both:
        # 运行两种攻击模式对比
        run_both_attacks(args.dataset, args.cover_rate, hub_num=args.hub_num, use_adaptive=args.adaptive)
    elif args.attack:
        # 指定了攻击模式，只运行单一模式
        main(args.dataset, args.attack, args.cover_rate, hub_num=args.hub_num, use_adaptive=args.adaptive)
    else:
        # 默认运行两种攻击的综合对比
        run_both_attacks(args.dataset, args.cover_rate, hub_num=args.hub_num, use_adaptive=args.adaptive)
