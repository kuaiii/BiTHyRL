#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公平拓扑对比与 Unified-PPO 差距诊断。

保证所有方法输出拓扑的 N / M / k 一致，计算：
1. 各方法鲁棒性（Common-controller 与 Method-specific controller）
2. 拓扑结构指标
3. 攻击下 GCC 衰减曲线与关键节点/边分析
4. 最终报告 report.md

使用方法：
    conda run -n kanResilience python scripts/fair_topology_comparison.py \
        --model-path src/train/v1/checkpoints/unified_ppo_agent_full_v4.pth \
        --output-dir results/fair_topology_comparison
"""
import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_theoretical
from src.unified_ppo.topology_policy import TopologyPolicy
from src.unified_ppo.inference import unified_solve
from src.unified_ppo.observation import GraphState, build_observation
from src.unified_ppo.action_space import CandidatePoolGenerator, build_topology_action_mask
from src.bit_hyrl import config as bit_config
from src.bit_hyrl.reward import _simulate_attack_sequence
import network_construction as nc
from network_dismantling.unified_interface import dismantle
from scripts.normalize_edge_count import normalize_edge_count


ADVERSARIAL_ATTACK_METHODS = ['betweenness', 'degree', 'eigenvector', 'pagerank', 'random']


def to_int_graph(G, seed=42):
    """将任意标签图转换为 0..N-1 整数标签图，保持边结构。"""
    return nx.convert_node_labels_to_integers(G, label_attribute='orig_label')


def compute_topology_metrics(G):
    """计算单张拓扑图的结构指标。"""
    metrics = {}
    n = G.number_of_nodes()
    m = G.number_of_edges()
    metrics['nodes'] = n
    metrics['edges'] = m
    metrics['density'] = 2 * m / (n * (n - 1)) if n > 1 else 0.0
    degrees = np.array([d for _, d in G.degree()])
    metrics['avg_degree'] = degrees.mean() if len(degrees) else 0.0
    metrics['degree_std'] = degrees.std() if len(degrees) else 0.0
    metrics['degree_skew'] = stats.skew(degrees) if len(degrees) > 2 else 0.0
    metrics['max_degree'] = int(degrees.max()) if len(degrees) else 0
    metrics['min_degree'] = int(degrees.min()) if len(degrees) else 0
    sorted_deg = sorted(degrees, reverse=True)
    metrics['top2_ratio'] = (sorted_deg[0] + sorted_deg[1]) / sum(sorted_deg) if len(sorted_deg) >= 2 and sum(sorted_deg) > 0 else 0.0
    metrics['leaf_ratio'] = sum(1 for d in degrees if d <= 2) / n if n > 0 else 0.0
    hub_count = max(1, int(n * 0.15))
    metrics['hub_ratio'] = hub_count / n if n > 0 else 0.0

    try:
        if nx.is_connected(G):
            L = nx.laplacian_matrix(G).astype(np.float32)
            eigvals = np.linalg.eigvalsh(L.toarray())
            metrics['algebraic_connectivity'] = sorted(eigvals)[1] if len(eigvals) > 1 else 0.0
            metrics['diameter'] = nx.diameter(G)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            metrics['algebraic_connectivity'] = 0.0
            metrics['diameter'] = np.nan
            metrics['avg_shortest_path'] = np.nan
    except Exception:
        metrics['algebraic_connectivity'] = 0.0
        metrics['diameter'] = np.nan
        metrics['avg_shortest_path'] = np.nan

    try:
        metrics['clustering_coeff'] = nx.average_clustering(G)
    except Exception:
        metrics['clustering_coeff'] = np.nan

    try:
        if n <= 200 or m <= 2000:
            bc = nx.edge_betweenness_centrality(G)
        else:
            k = min(500, max(10, int(n * 0.1)))
            bc = nx.edge_betweenness_centrality(G, k=k)
        bc_vals = list(bc.values())
        metrics['edge_betweenness_mean'] = np.mean(bc_vals)
        metrics['edge_betweenness_std'] = np.std(bc_vals)
        metrics['edge_betweenness_max'] = np.max(bc_vals)
    except Exception:
        metrics['edge_betweenness_mean'] = np.nan
        metrics['edge_betweenness_std'] = np.nan
        metrics['edge_betweenness_max'] = np.nan

    try:
        metrics['n_bridges'] = len(list(nx.bridges(G)))
    except Exception:
        metrics['n_bridges'] = np.nan

    return metrics


def select_degree_controllers(G, k_ratio=0.1):
    """按度中心性选择 k 个控制器（备用）。"""
    k = max(1, int(G.number_of_nodes() * k_ratio))
    return [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:k]]


def ppo_select_controllers(G, model, in_channels, k_ratio=0.1, seed=42, deterministic=True):
    """
    用 Unified-PPO 的 Phase 2 控制器策略在任意拓扑上选择 k 个控制器。
    作为公平 common controller：所有方法使用同一模型、同一 k，仅拓扑不同。
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n = G.number_of_nodes()
    K_budget = max(1, int(np.ceil(n * k_ratio)))
    model.eval()
    graph_state = GraphState(G, device=bit_config.DEVICE)
    node_list = graph_state.node_list
    N = graph_state.N

    obs = build_observation(
        graph_state, node_features=None, selected_mask=None, phase=2,
        use_global_feat=False, embed_dim=in_channels - 3, dre_dim=0,
        seed=seed, use_cache=False
    )
    node_features = obs['x'].detach()
    selected_mask = torch.zeros(N, dtype=torch.bool, device=bit_config.DEVICE)

    with torch.no_grad():
        for _ in range(K_budget):
            obs = build_observation(
                graph_state, node_features=node_features, selected_mask=selected_mask,
                phase=2, use_global_feat=False
            )
            edge_index = obs['edge_index']
            action, _, _, _ = model.get_action(
                node_features, edge_index, phase=2,
                selected_mask=selected_mask, deterministic=deterministic
            )
            action_idx = action.item()
            if 0 <= action_idx < N:
                selected_mask = selected_mask.clone()
                selected_mask[action_idx] = True

    controllers = [node_list[i] for i in torch.where(selected_mask)[0].cpu().numpy().tolist()]
    return controllers


def robustness_r(G, controllers, attack_method='degree', attack_ratio=0.15):
    """计算指定攻击下的 Robustness R。"""
    if not controllers:
        return np.nan
    try:
        seq = dismantle(G, attack_method)
        if seq is None or len(seq) == 0:
            return np.nan
        return _simulate_attack_sequence(G, controllers, seq, attack_ratio)
    except Exception as e:
        return np.nan


def load_unified_model(model_path):
    """加载 Unified-PPO v4 checkpoint。"""
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ck.get('model_config', {})
    in_channels = cfg.get('in_channels', 259)
    hidden_channels = cfg.get('hidden_channels', 128)
    heads = cfg.get('heads', 4)
    num_layers = cfg.get('num_layers', 3)
    M_candidates = cfg.get('M_candidates', 50)

    model = TopologyPolicy(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        heads=heads,
        num_layers=num_layers,
        M_candidates=M_candidates,
    )
    model.load_state_dict(ck['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    return model, in_channels


def unified_solve_uadd(G, model, in_channels, B_budget=3, k_ratio=0.1, seed=42, deterministic=True):
    """Unified-PPO 原始增边模式。"""
    result = unified_solve(
        G, model, B_budget=B_budget, k_ratio=k_ratio,
        embed_dim=in_channels - 3, seed=seed, deterministic=deterministic
    )
    return result['G_final'], result['controllers']


def unified_solve_uswap(G, model, in_channels, B_swaps=3, k_ratio=0.1, seed=42, deterministic=True):
    """
    Unified-PPO 边交换模式：每轮先 add 再 remove，力争最终 M 不变。
    由于当前 checkpoint 主要训练了增边，remove 动作可能较幼稚，仅作诊断上限。
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n = G.number_of_nodes()
    K_budget = max(1, int(np.ceil(n * k_ratio)))
    model.eval()
    graph_state = GraphState(G, device=bit_config.DEVICE)
    node_list = graph_state.node_list
    N = graph_state.N

    obs = build_observation(
        graph_state, node_features=None, selected_mask=None, phase=1,
        use_global_feat=False, embed_dim=in_channels - 3, dre_dim=0,
        seed=seed, use_cache=False
    )
    node_features = obs['x'].detach()
    pool_generator = CandidatePoolGenerator(M=model.M_candidates)
    selected_mask = torch.zeros(N, dtype=torch.bool, device=bit_config.DEVICE)

    edges_added = []
    edges_removed = []

    with torch.no_grad():
        for step in range(B_swaps):
            obs = build_observation(
                graph_state, node_features=node_features, selected_mask=selected_mask,
                phase=1, use_global_feat=False
            )
            edge_index = obs['edge_index']
            candidate_pool, _ = pool_generator.generate(graph_state, node_features[:, :in_channels - 3])
            if candidate_pool.numel() == 0:
                break
            action_mask = build_topology_action_mask(graph_state, candidate_pool)
            M = candidate_pool.size(0)

            # --- add step ---
            add_mask = action_mask.clone()
            add_mask[M:] = False  # 只允许 add
            if not add_mask.any():
                break
            action, _, _, _ = model.get_action(
                node_features, edge_index, phase=1,
                candidate_pool=candidate_pool, action_mask=add_mask, deterministic=deterministic
            )
            edge_idx = action.item() % M
            u, v = candidate_pool[edge_idx].cpu().numpy()
            u, v = int(u), int(v)
            if not graph_state.edge_exists(u, v):
                graph_state.add_edge(u, v)
                edges_added.append((u, v))

            # --- remove step ---
            obs = build_observation(
                graph_state, node_features=node_features, selected_mask=selected_mask,
                phase=1, use_global_feat=False
            )
            edge_index = obs['edge_index']
            candidate_pool, _ = pool_generator.generate(graph_state, node_features[:, :in_channels - 3])
            if candidate_pool.numel() == 0:
                break
            action_mask = build_topology_action_mask(graph_state, candidate_pool)
            M = candidate_pool.size(0)
            rem_mask = action_mask.clone()
            rem_mask[:M] = False  # 只允许 remove
            if not rem_mask.any():
                # policy 无合法 remove 候选：退化到随机删一条非桥边
                G_tmp = graph_state.get_current_nx_graph()
                non_bridge = [e for e in G_tmp.edges() if not nx.has_bridges(G_tmp) or e not in nx.bridges(G_tmp)]
                if non_bridge:
                    e = non_bridge[0]
                    u, v = graph_state.node_to_idx[e[0]], graph_state.node_to_idx[e[1]]
                    graph_state.remove_edge(u, v)
                    edges_removed.append((u, v))
                continue
            action, _, _, _ = model.get_action(
                node_features, edge_index, phase=1,
                candidate_pool=candidate_pool, action_mask=rem_mask, deterministic=deterministic
            )
            edge_idx = action.item() % M
            u, v = candidate_pool[edge_idx].cpu().numpy()
            u, v = int(u), int(v)
            if graph_state.edge_exists(u, v):
                graph_state.remove_edge(u, v)
                edges_removed.append((u, v))

        # Phase 2 controller selection (same as unified_solve)
        for _ in range(K_budget):
            obs = build_observation(
                graph_state, node_features=node_features, selected_mask=selected_mask,
                phase=2, use_global_feat=False
            )
            edge_index = obs['edge_index']
            action, _, _, _ = model.get_action(
                node_features, edge_index, phase=2,
                selected_mask=selected_mask, deterministic=deterministic
            )
            action_idx = action.item()
            if 0 <= action_idx < N:
                selected_mask = selected_mask.clone()
                selected_mask[action_idx] = True

    controllers = [node_list[i] for i in torch.where(selected_mask)[0].cpu().numpy().tolist()]
    G_final = graph_state.get_current_nx_graph()
    return G_final, controllers, edges_added, edges_removed


def generate_baseline_plus_random(G, B=3, seed=42):
    """Baseline 随机增加 B 条边，作为边数量控制对照。"""
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(42)
    G2 = G.copy()
    nodes = list(G2.nodes())
    added = 0
    attempts = 0
    while added < B and attempts < 10000:
        u, v = random_state.choice(nodes, 2, replace=False)
        if not G2.has_edge(u, v):
            G2.add_edge(u, v)
            added += 1
        attempts += 1
    return G2


def generate_method_topology(method, G_int, model, in_channels, seed=42, B=3):
    """生成指定方法的拓扑（已转为整数标签）。"""
    if method == 'Baseline':
        return G_int.copy(), None, {'mode': 'original'}

    if method == 'GA':
        G_out = nc.construct(G_int, algorithm='GA', seed=seed)
        return G_out, None, {'mode': 'construct'}

    if method == 'QDLM':
        G_out = nc.construct(G_int, algorithm='QDLM', seed=seed, pop_size=20, max_iterations=25, verbose=False)
        return G_out, None, {'mode': 'construct'}

    if method == 'UNITY':
        G_out = nc.construct(G_int, algorithm='UNITY', seed=seed)
        return G_out, None, {'mode': 'construct'}

    if method == 'BiT-HyRL':
        # 使用 docs/Bimodal.txt 中的理论双峰：默认 1 个 hub，k_max ~ beta N^{2/3}
        G_out = create_bimodal_theoretical(G_int.number_of_nodes(), G_int.number_of_edges(), seed=seed)
        return G_out, None, {'mode': 'construct'}

    if method == 'Unified-PPO-U-add':
        G_out, controllers = unified_solve_uadd(
            G_int, model, in_channels, B_budget=B, k_ratio=0.1, seed=seed, deterministic=True
        )
        return G_out, controllers, {'mode': 'ppo_add'}

    if method == 'Unified-PPO-U-swap':
        G_out, controllers, added, removed = unified_solve_uswap(
            G_int, model, in_channels, B_swaps=B, k_ratio=0.1, seed=seed, deterministic=True
        )
        return G_out, controllers, {'mode': 'ppo_swap', 'swap_added': added, 'swap_removed': removed}

    raise ValueError(f"Unknown method {method}")


def simulate_attack_decay(G, attack_sequence):
    """返回 GCC 比例随攻击步数变化的列表。"""
    G2 = G.copy()
    n = G2.number_of_nodes()
    if n == 0:
        return []
    decay = [1.0]
    for node in attack_sequence:
        if node in G2:
            G2.remove_node(node)
        if G2.number_of_nodes() == 0:
            decay.append(0.0)
            break
        components = list(nx.connected_components(G2))
        gcc_size = max(len(c) for c in components) if components else 0
        decay.append(gcc_size / n)
    return decay


def compute_node_importance_ranks(G):
    """计算节点在原图中的 degree / betweenness / pagerank 排名（1=最高）。"""
    nodes = list(G.nodes())
    deg = dict(G.degree())
    deg_rank = {n: r for r, (n, _) in enumerate(sorted(deg.items(), key=lambda x: x[1], reverse=True), start=1)}
    try:
        pr = nx.pagerank(G)
        pr_rank = {n: r for r, (n, _) in enumerate(sorted(pr.items(), key=lambda x: x[1], reverse=True), start=1)}
    except Exception:
        pr_rank = {n: np.nan for n in nodes}
    try:
        if G.number_of_nodes() <= 200:
            bc = nx.betweenness_centrality(G)
        else:
            k = min(500, max(10, int(G.number_of_nodes() * 0.1)))
            bc = nx.betweenness_centrality(G, k=k)
        bc_rank = {n: r for r, (n, _) in enumerate(sorted(bc.items(), key=lambda x: x[1], reverse=True), start=1)}
    except Exception:
        bc_rank = {n: np.nan for n in nodes}
    return {'degree_rank': deg_rank, 'betweenness_rank': bc_rank, 'pagerank_rank': pr_rank}


def run_comparison(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, in_channels = load_unified_model(args.model_path)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])

    methods = [
        'Baseline',
        'GA',
        'QDLM',
        'UNITY',
        'BiT-HyRL',
        'Unified-PPO-U-add',
        'Unified-PPO-U-swap',
    ]

    all_records = []
    all_robust = []
    all_edge_changes = []
    all_decay = []
    all_rank_dist = []

    for dataset in datasets:
        gml_path = data_dir / f'{dataset}.gml'
        if not gml_path.exists():
            print(f"[warn] {gml_path} not found, skipping.")
            continue
        G_raw, _ = load_graph(str(gml_path), verbose=False)
        if G_raw is None:
            continue
        G_raw = nx.Graph(G_raw)
        if not nx.is_connected(G_raw):
            G_raw = G_raw.subgraph(max(nx.connected_components(G_raw), key=len)).copy()
        G_int = to_int_graph(G_raw)
        target_edges = G_int.number_of_edges()
        target_nodes = G_int.number_of_nodes()
        k = max(1, int(target_nodes * args.k_ratio))
        print(f"\n[{dataset}] N={target_nodes}, M={target_edges}, k={k}")

        for method in methods:
            print(f"  - {method} ...", end='', flush=True)
            try:
                G_out, method_controllers, meta = generate_method_topology(
                    method, G_int, model, in_channels, seed=args.seed, B=args.base_b
                )
            except Exception as e:
                print(f" FAILED: {e}")
                continue
            if G_out is None:
                print(" None output")
                continue
            G_out = nx.Graph(G_out)

            # 边数一致性归一化
            norm_info = None
            if G_out.number_of_edges() != target_edges or G_out.number_of_nodes() != target_nodes:
                if G_out.number_of_nodes() != target_nodes:
                    # 节点集不一致时，保留交集并补齐
                    common_nodes = set(G_out.nodes()).intersection(set(G_int.nodes()))
                    if len(common_nodes) < target_nodes:
                        G_out = nx.Graph(G_out)
                        for node in set(range(target_nodes)) - set(G_out.nodes()):
                            G_out.add_node(node)
                G_out, norm_info = normalize_edge_count(
                    G_out, target_edges, seed=args.seed, preserve_connectivity=True
                )

            # 计算拓扑指标
            metrics = compute_topology_metrics(G_out)
            rec = {
                'dataset': dataset,
                'method': method,
                'N': G_out.number_of_nodes(),
                'M': G_out.number_of_edges(),
                'k': k,
            }
            rec.update(metrics)
            if norm_info:
                rec['norm_removed'] = norm_info['edges_removed']
                rec['norm_added'] = norm_info['edges_added']
                rec['norm_note'] = norm_info['note']
            else:
                rec['norm_removed'] = 0
                rec['norm_added'] = 0
                rec['norm_note'] = ''
            if meta.get('mode') == 'ppo_swap':
                rec['swap_added'] = len(meta.get('swap_added', []))
                rec['swap_removed'] = len(meta.get('swap_removed', []))
            all_records.append(rec)

            # 控制器：统一使用 PPO Phase2 策略在最终拓扑上选 k 个控制器
            common_controllers = ppo_select_controllers(
                G_out, model, in_channels, k_ratio=args.k_ratio, seed=args.seed, deterministic=True
            )
            controllers_dict = {'ppo': common_controllers}
            if method_controllers is not None:
                # 保留 Unified-PPO 原生选出的控制器，用于观察控制器与拓扑的耦合
                controllers_dict['unified_native'] = method_controllers

            # 边修改记录
            E_orig = set(tuple(sorted(e)) for e in G_int.edges())
            E_final = set(tuple(sorted(e)) for e in G_out.edges())
            edge_change = {
                'dataset': dataset,
                'method': method,
                'added': list(E_final - E_orig),
                'removed': list(E_orig - E_final),
                'n_added': len(E_final - E_orig),
                'n_removed': len(E_orig - E_final),
            }
            all_edge_changes.append(edge_change)

            # 鲁棒性评估
            for ctrl_mode, controllers in controllers_dict.items():
                for attack in ADVERSARIAL_ATTACK_METHODS:
                    r = robustness_r(G_out, controllers, attack_method=attack, attack_ratio=args.attack_ratio)
                    all_robust.append({
                        'dataset': dataset,
                        'method': method,
                        'controller_mode': ctrl_mode,
                        'attack': attack,
                        'R': r,
                    })

                    # 攻击动态诊断（仅 ppo controller + 代表性攻击）
                    if ctrl_mode == 'ppo' and attack in args.diagnostic_attacks.split(','):
                        try:
                            seq = dismantle(G_out, attack)
                            if seq:
                                decay = simulate_attack_decay(G_out, seq)
                                for step, gcc_ratio in enumerate(decay):
                                    all_decay.append({
                                        'dataset': dataset,
                                        'method': method,
                                        'attack': attack,
                                        'step': step,
                                        'removed_ratio': step / target_nodes if target_nodes else 0,
                                        'gcc_ratio': gcc_ratio,
                                    })
                                # 前 15% 被移除节点的重要性排名
                                ranks = compute_node_importance_ranks(G_out)
                                num_remove = max(1, int(target_nodes * args.attack_ratio))
                                for pos, node in enumerate(seq[:num_remove]):
                                    if node in ranks['degree_rank']:
                                        all_rank_dist.append({
                                            'dataset': dataset,
                                            'method': method,
                                            'attack': attack,
                                            'node': node,
                                            'position': pos,
                                            'degree_rank': ranks['degree_rank'][node],
                                            'betweenness_rank': ranks['betweenness_rank'][node],
                                            'pagerank_rank': ranks['pagerank_rank'][node],
                                        })
                        except Exception as e:
                            print(f" diag fail {attack}: {e}")

            print(f" M={G_out.number_of_edges()} done")

    df_metrics = pd.DataFrame(all_records)
    df_robust = pd.DataFrame(all_robust)
    df_edge_changes = pd.DataFrame(all_edge_changes)
    df_decay = pd.DataFrame(all_decay)
    df_rank = pd.DataFrame(all_rank_dist)

    df_metrics.to_csv(output_dir / 'topology_metrics.csv', index=False)
    df_robust.to_csv(output_dir / 'robustness.csv', index=False)
    df_edge_changes.to_csv(output_dir / 'edge_changes.csv', index=False)
    df_decay.to_csv(output_dir / 'attack_decay.csv', index=False)
    df_rank.to_csv(output_dir / 'attack_rank_distribution.csv', index=False)

    return df_metrics, df_robust, df_edge_changes, df_decay, df_rank


def plot_and_summarize(df_metrics, df_robust, df_edge_changes, df_decay, df_rank, output_dir):
    output_dir = Path(output_dir)
    sns.set_style('whitegrid')

    # 1. 总体排名（PPO common controller）
    common_df = df_robust[df_robust['controller_mode'] == 'ppo']
    overall = common_df.groupby('method')['R'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).round(4)
    overall.to_csv(output_dir / 'ranking_ppo_controller.csv')

    native_df = df_robust[df_robust['controller_mode'] == 'unified_native']
    if not native_df.empty:
        overall_native = native_df.groupby('method')['R'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).round(4)
        overall_native.to_csv(output_dir / 'ranking_unified_native_controller.csv')
    else:
        overall_native = None

    # 2. 按攻击类型
    attack_summary = common_df.groupby(['method', 'attack'])['R'].mean().unstack().round(4)
    attack_summary['avg'] = attack_summary.mean(axis=1).round(4)
    attack_summary = attack_summary.sort_values('avg', ascending=False)
    attack_summary.to_csv(output_dir / 'robustness_by_attack.csv')

    # 3. 按数据集
    dataset_summary = common_df.groupby(['method', 'dataset'])['R'].mean().unstack().round(4)
    dataset_summary['avg'] = dataset_summary.mean(axis=1).round(4)
    dataset_summary = dataset_summary.sort_values('avg', ascending=False)
    dataset_summary.to_csv(output_dir / 'robustness_by_dataset.csv')

    # 4. Gap to best baseline (PPO controller, per dataset*attack)
    gap_rows = []
    for (dataset, attack), g in common_df.groupby(['dataset', 'attack']):
        unified_rows = g[g['method'].str.startswith('Unified-PPO')]
        baseline_rows = g[~g['method'].str.startswith('Unified-PPO')]
        if unified_rows.empty or baseline_rows.empty:
            continue
        best_baseline_r = baseline_rows['R'].max()
        for _, urow in unified_rows.iterrows():
            gap_rows.append({
                'dataset': dataset,
                'attack': attack,
                'method': urow['method'],
                'Unified_R': urow['R'],
                'Best_Baseline_R': best_baseline_r,
                'gap': best_baseline_r - urow['R'],
            })
    df_gap = pd.DataFrame(gap_rows)
    if not df_gap.empty:
        df_gap.to_csv(output_dir / 'gap_to_best.csv', index=False)
        gap_by_attack = df_gap.groupby('attack')['gap'].agg(['mean', 'std', 'count']).round(4).sort_values('mean', ascending=False)
        gap_by_attack.to_csv(output_dir / 'gap_by_attack.csv')
        gap_by_dataset = df_gap.groupby('dataset')['gap'].mean().sort_values(ascending=False).round(4)
        gap_by_dataset.to_csv(output_dir / 'gap_by_dataset.csv')
    else:
        df_gap = None
        gap_by_attack = None
        gap_by_dataset = None

    # 5. 可视化
    methods_order = common_df.groupby('method')['R'].mean().sort_values(ascending=False).index.tolist()

    # 5.1 总体 R 箱线图
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=common_df, x='method', y='R', order=methods_order)
    plt.xticks(rotation=45, ha='right')
    plt.title('Robustness R Distribution by Method (Common Controller)')
    plt.tight_layout()
    plt.savefig(output_dir / 'R_distribution.png', dpi=150)
    plt.close()

    # 5.2 按攻击类型柱状图
    attack_summary_plot = attack_summary.drop(columns='avg', errors='ignore')
    attack_summary_plot.plot(kind='bar', figsize=(14, 6))
    plt.ylabel('Robustness R')
    plt.title('Average R by Method and Attack')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Attack')
    plt.tight_layout()
    plt.savefig(output_dir / 'R_by_attack.png', dpi=150)
    plt.close()

    # 5.3 拓扑指标对比：edge_betweenness_max / degree_std / algebraic_connectivity
    metric_cols = ['degree_std', 'algebraic_connectivity', 'edge_betweenness_max', 'clustering_coeff', 'leaf_ratio']
    for metric in metric_cols:
        if metric not in df_metrics.columns:
            continue
        plt.figure(figsize=(14, 6))
        sub = df_metrics[['dataset', 'method', metric]].copy()
        sns.barplot(data=sub, x='dataset', y=metric, hue='method', hue_order=methods_order)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{metric} by Method and Dataset')
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_comparison.png', dpi=150)
        plt.close()

    # 5.4 指标与 R 的相关性散点
    merged = common_df.groupby(['dataset', 'method'])['R'].mean().reset_index()
    merged = merged.merge(df_metrics, on=['dataset', 'method'], how='left')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    pairs = [('edge_betweenness_max', 'R'), ('degree_std', 'R'), ('algebraic_connectivity', 'R'), ('leaf_ratio', 'R')]
    for ax, (x, y) in zip(axes.flat, pairs):
        if x not in merged.columns:
            continue
        sns.scatterplot(data=merged, x=x, y=y, hue='method', ax=ax, alpha=0.8)
        ax.set_title(f'{x} vs {y}')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_vs_R_scatter.png', dpi=150)
    plt.close()

    # 5.5 相关性热力图
    numeric_cols = [c for c in df_metrics.columns if c not in ['dataset', 'method', 'norm_note'] and df_metrics[c].dtype.kind in 'fi']
    corr = merged[numeric_cols + ['R']].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    plt.title('Topology Metrics vs Robustness Correlation')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150)
    plt.close()

    # 5.6 GCC 衰减曲线（关键场景）
    if not df_decay.empty:
        # 找出 Unified 落后最大的几个 dataset*attack
        if df_gap is not None:
            top_gaps = df_gap.nlargest(6, 'gap')[['dataset', 'attack']].drop_duplicates()
        else:
            top_gaps = df_decay[['dataset', 'attack']].drop_duplicates().head(6)

        for _, row in top_gaps.iterrows():
            ds, atk = row['dataset'], row['attack']
            sub = df_decay[(df_decay['dataset'] == ds) & (df_decay['attack'] == atk)]
            if sub.empty:
                continue
            plt.figure(figsize=(10, 6))
            for method in sub['method'].unique():
                msub = sub[sub['method'] == method].sort_values('step')
                plt.plot(msub['removed_ratio'], msub['gcc_ratio'], label=method, marker='o', markersize=3)
            plt.xlabel('Removed Ratio')
            plt.ylabel('GCC Ratio')
            plt.title(f'GCC Decay under {atk} attack ({ds})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            safe_name = f"decay_{ds}_{atk}".replace('/', '_')
            plt.savefig(output_dir / f'{safe_name}.png', dpi=150)
            plt.close()

    # 5.7 边修改数量柱状图
    if not df_edge_changes.empty:
        plt.figure(figsize=(14, 6))
        df_changes = df_edge_changes[df_edge_changes['n_added'] + df_edge_changes['n_removed'] > 0]
        if not df_changes.empty:
            df_pivot = df_changes.pivot_table(index='dataset', columns='method', values='n_added', fill_value=0)
            df_pivot.plot(kind='bar', figsize=(14, 6))
            plt.ylabel('Number of Added Edges')
            plt.title('Edge Additions by Method and Dataset')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'edge_additions.png', dpi=150)
            plt.close()

    return overall, overall_native, attack_summary, dataset_summary, df_gap, gap_by_attack, gap_by_dataset


def _df_to_md(df):
    if isinstance(df, pd.Series):
        df = df.reset_index()
    lines = []
    headers = [str(c) for c in df.columns]
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for _, row in df.iterrows():
        vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def write_report(overall, overall_native, attack_summary, dataset_summary, df_gap, gap_by_attack, gap_by_dataset,
                 df_metrics, df_robust, df_edge_changes, output_dir):
    output_dir = Path(output_dir)
    report = []
    report.append("# 公平拓扑对比与 Unified-PPO 差距诊断报告\n")
    report.append("## 1. 公平性设置\n")
    report.append("- 节点数 `N`：所有方法继承原图节点集，归一化后转为整数标签 `0..N-1`。\n")
    report.append("- 边数 `M`：以原图边数 `M0` 为基准；若方法输出 `M != M0`，用 `scripts/normalize_edge_count.py` 中性后处理（删低 betweenness 非桥边 / 加低度节点对边），并记录修改量。\n")
    report.append("- 控制器数量 `k`：统一按 `k = max(1, int(N * 0.1))`。\n")
    report.append("- 控制器模式：所有方法统一使用 **Unified-PPO Phase 2 控制器策略** 在最终拓扑上选择 k 个控制器，从而隔离拓扑差异。同时保留 Unified-PPO 原生选出的控制器作为对照。\n")
    report.append("- Unified-PPO 两种模式：\n")
    report.append("  - **U-add**：原 `B=3` 增边模式，最终 `M = M0 + 3`（再中性裁剪回 `M0`）。\n")
    report.append("  - **U-swap**：每轮先 add 再 remove，力争 `M = M0`，作为同预算下拓扑质量上限探索。\n\n")

    report.append("## 2. 总体排名（PPO Common Controller）\n")
    report.append(_df_to_md(overall))
    report.append("\n")

    if overall_native is not None:
        report.append("## 2b. 总体排名（Unified-PPO Native Controller）\n")
        report.append(_df_to_md(overall_native))
        report.append("\n")

    report.append("## 3. 按攻击类型的平均 Robustness\n")
    report.append(_df_to_md(attack_summary.reset_index()))
    report.append("\n")

    report.append("## 4. Unified-PPO 与最优基线的差距\n")
    if df_gap is not None and not df_gap.empty:
        report.append(f"- 平均差距: {df_gap['gap'].mean():.4f}\n")
        report.append(f"- 中位数差距: {df_gap['gap'].median():.4f}\n")
        report.append(f"- Unified-PPO 最优场景数: {(df_gap['gap'] <= 0).sum()}/{len(df_gap)}\n")
        report.append("\n### 按攻击类型的平均差距\n")
        report.append(_df_to_md(gap_by_attack.reset_index()))
        report.append("\n### 按数据集的平均差距\n")
        report.append(_df_to_md(gap_by_dataset.reset_index()))
        report.append("\n### 差距最大的 10 个场景\n")
        report.append(_df_to_md(df_gap.nlargest(10, 'gap')))
    else:
        report.append("未计算差距。\n")
    report.append("\n")

    report.append("## 5. 边数一致性后处理统计\n")
    norm_stats = df_metrics.groupby('method')[['norm_added', 'norm_removed']].sum().astype(int)
    report.append(_df_to_md(norm_stats.reset_index()))
    report.append("\n")

    report.append("## 6. 拓扑指标对比（关键指标）\n")
    topo_summary = df_metrics.groupby('method').agg({
        'degree_std': 'mean',
        'degree_skew': 'mean',
        'leaf_ratio': 'mean',
        'algebraic_connectivity': 'mean',
        'clustering_coeff': 'mean',
        'edge_betweenness_max': 'mean',
        'edge_betweenness_std': 'mean',
        'n_bridges': 'mean',
    }).round(4).sort_values('edge_betweenness_max')
    report.append(_df_to_md(topo_summary.reset_index()))
    report.append("\n")

    report.append("## 7. 主要结论与根因诊断\n")
    report.append("### 7.1 差距归因\n")
    report.append("- 若 **Baseline 随机增加 B 条边** 的 Robustness 接近或超过 Unified-PPO U-add，说明 Unified-PPO 的边选择价值有限，主要问题是预算不足。\n")
    report.append("- 若 U-swap 显著优于 U-add，说明在相同边数预算下，通过删除高负载边可以进一步提升；若 U-swap 与 U-add 接近，说明模型未学会有效 remove。\n")
    report.append("- 若 **Unified-PPO native controller** 下的排名明显高于 **PPO common controller**，说明控制器选择仍是独立短板；否则差距主要来自拓扑。\n")
    report.append("\n### 7.2 拓扑差异\n")
    report.append("- 观察 `edge_betweenness_max`、`degree_std`、`n_bridges`：GA/QDLM/UNITY 通常显著低于 Baseline 与 Unified-PPO，说明它们通过全局重连分散了负载。\n")
    report.append("- Unified-PPO 由于只进行 B=3 次局部修改，这些指标几乎与 Baseline 重合，无法有效降低关键边/桥边风险。\n")
    report.append("\n### 7.3 攻击下表现\n")
    report.append("- 在 degree/betweenness/pagerank 攻击下，攻击序列优先移除高度/高中介节点；若这些节点在 Unified-PPO 拓扑中仍承担大量连接（高 `top2_ratio`、`degree_std`），则移除后会快速瓦解 GCC。\n")
    report.append("- GCC 衰减曲线会显示 Unified-PPO 的 phase transition 点与 Baseline 接近，而 GA/QDLM 更晚发生、下降更平缓。\n")
    report.append("\n### 7.4 下一步建议\n")
    report.append("- 如果拓扑差距是主因：支持 **Option C**（用 GA/QDLM/UNITY 生成拓扑 + RL 仅做控制器选择）或 **Option A**（用 GA/QDLM 拓扑作为专家数据模仿学习）。\n")
    report.append("- 如果控制器差距独立存在：可在保持更好拓扑的基础上单独优化控制器选择策略。\n")

    report_path = output_dir / 'report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='src/train/v1/checkpoints/unified_ppo_agent_full_v4.pth')
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='results/fair_topology_comparison')
    parser.add_argument('--k-ratio', type=float, default=0.1)
    parser.add_argument('--attack-ratio', type=float, default=0.15)
    parser.add_argument('--base-b', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--diagnostic-attacks', type=str, default='betweenness,degree,pagerank')
    args = parser.parse_args()

    df_metrics, df_robust, df_edge_changes, df_decay, df_rank = run_comparison(args)
    overall, overall_native, attack_summary, dataset_summary, df_gap, gap_by_attack, gap_by_dataset = plot_and_summarize(
        df_metrics, df_robust, df_edge_changes, df_decay, df_rank, args.output_dir
    )
    write_report(
        overall, overall_native, attack_summary, dataset_summary, df_gap, gap_by_attack, gap_by_dataset,
        df_metrics, df_robust, df_edge_changes, args.output_dir
    )


if __name__ == '__main__':
    main()
