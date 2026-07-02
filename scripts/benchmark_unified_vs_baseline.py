#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速 A/B 验证：统一 PPO (Topology Refinement + Controller Selection) vs
现有基线 (create_bimodal_network_exact + hybrid_gnn_select)。

用法：
  python scripts/benchmark_unified_vs_baseline.py --num-graphs 20 --max-nodes 100
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import csv
import time
import random
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_network_exact
from src.bit_hyrl import config
from src.bit_hyrl.gnn_model import get_gnn_node_features, graph_to_pyg_data
from src.bit_hyrl.selection import hybrid_gnn_select, gnn_predict, load_gnn_model
from src.bit_hyrl.reward import calculate_gcc_focused_reward
from src.unified_ppo.topology_policy import TopologyPolicy
from src.unified_ppo.reward_unified import calculate_unified_reward
from src.unified_ppo.inference import unified_solve


def load_small_graphs(data_dir, num_graphs=20, min_nodes=50, max_nodes=100, seed=42):
    if data_dir is None:
        data_dir = os.path.join(ROOT, 'dataset', 'all', 'syn')
    files = [f for f in os.listdir(data_dir) if f.endswith('.gml')]
    random.seed(seed)
    random.shuffle(files)
    graphs = []
    for f in files:
        if len(graphs) >= num_graphs:
            break
        try:
            G, _ = load_graph(os.path.join(data_dir, f), verbose=False)
            if G is None:
                continue
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            n = G.number_of_nodes()
            if min_nodes <= n <= max_nodes:
                graphs.append((f, G))
        except Exception:
            continue
    return graphs


def baseline_solve(G, k_ratio=0.1, model_path=None):
    """基线：双峰重构 + GNN 选择。"""
    start = time.time()
    n = G.number_of_nodes()
    k = max(1, int(n * k_ratio))
    hub_num = max(1, int(n * 0.15))
    G_BiT = create_bimodal_network_exact(G, hub_num=hub_num)

    # 加载基线模型并适配特征维度
    if model_path and os.path.exists(model_path):
        model, checkpoint, model_type = load_gnn_model(model_path, prefer_curriculum=False)
        if model is not None:
            embed_dim = checkpoint.get('embed_dim', 256)
            dre_dim = checkpoint.get('dre_dim', 0)
            centers = gnn_predict(G_BiT, k, model=model, model_type=model_type,
                                  deterministic=True, embed_dim=embed_dim, dre_dim=dre_dim)
        else:
            centers = hybrid_gnn_select(G_BiT, k, model_path=model_path, prefer_curriculum=False)
    else:
        centers = hybrid_gnn_select(G_BiT, k, model_path=model_path, prefer_curriculum=False)

    elapsed = time.time() - start
    reward = calculate_unified_reward(G_BiT, centers, G_original=G)
    gcc_reward = calculate_gcc_focused_reward(G_BiT, centers)
    edge_diff = len(set(tuple(sorted(e)) for e in G.edges()).symmetric_difference(
        set(tuple(sorted(e)) for e in G_BiT.edges())))
    return {
        'G_final': G_BiT,
        'controllers': centers,
        'unified_reward': reward,
        'gcc_reward': gcc_reward,
        'edge_diff': edge_diff,
        'time': elapsed,
    }


def unified_solve_and_eval(G, model, B_budget=3, k_ratio=0.1,
                            embed_dim=256, dre_dim=0, seed=42):
    """统一 PPO：在原始图上直接拓扑微调 + 选择控制器，并计算指标。"""
    result = unified_solve(G, model, B_budget=B_budget, k_ratio=k_ratio,
                           embed_dim=embed_dim, dre_dim=dre_dim, seed=seed,
                           deterministic=True)
    G_final = result['G_final']
    centers = result['controllers']
    reward = calculate_unified_reward(G_final, centers, G_original=G)
    gcc_reward = calculate_gcc_focused_reward(G_final, centers)
    edge_diff = len(set(tuple(sorted(e)) for e in G.edges()).symmetric_difference(
        set(tuple(sorted(e)) for e in G_final.edges())))
    return {
        'G_final': G_final,
        'controllers': centers,
        'unified_reward': reward,
        'gcc_reward': gcc_reward,
        'edge_diff': edge_diff,
        'time': result['time'],
    }


def main():
    parser = argparse.ArgumentParser(description='Unified PPO vs BiT-HyRL Baseline Benchmark')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--num-graphs', type=int, default=20)
    parser.add_argument('--min-nodes', type=int, default=50)
    parser.add_argument('--max-nodes', type=int, default=100)
    parser.add_argument('--B-budget', type=int, default=3)
    parser.add_argument('--K-ratio', type=float, default=0.1)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--dre-dim', type=int, default=0)
    parser.add_argument('--M-candidates', type=int, default=30)
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--model-path', type=str, default=None,
                        help='统一 PPO 模型路径，None 则随机初始化')
    parser.add_argument('--baseline-model-path', type=str, default=None,
                        help='基线 GNN 模型路径，None 则自动搜索')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Unified PPO vs Baseline Benchmark")
    print("=" * 60)
    print(f"Graphs: {args.num_graphs}, nodes: {args.min_nodes}-{args.max_nodes}")
    print(f"B={args.B_budget}, K-ratio={args.K_ratio}, M={args.M_candidates}")
    print("=" * 60)

    graphs = load_small_graphs(args.data_dir, args.num_graphs, args.min_nodes,
                               args.max_nodes, args.seed)
    if not graphs:
        print("未加载到图，退出。")
        return

    in_channels = args.embed_dim + 3
    model = TopologyPolicy(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        M_candidates=args.M_candidates,
    ).to(config.DEVICE)

    if args.model_path and os.path.exists(args.model_path):
        ck = torch.load(args.model_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        print(f"加载统一模型: {args.model_path}")
    else:
        print("未提供统一模型或模型不存在，使用随机初始化模型进行对比。")

    rows = []
    baseline_rewards = []
    unified_rewards = []

    for name, G in tqdm(graphs, desc='Benchmarking'):
        try:
            base = baseline_solve(G, k_ratio=args.K_ratio, model_path=args.baseline_model_path)
            uni = unified_solve_and_eval(G, model, B_budget=args.B_budget, k_ratio=args.K_ratio,
                                         embed_dim=args.embed_dim, dre_dim=args.dre_dim, seed=args.seed)
            rows.append({
                'name': name,
                'n': G.number_of_nodes(),
                'm': G.number_of_edges(),
                'baseline_reward': base['unified_reward'],
                'unified_reward': uni['unified_reward'],
                'baseline_gcc': base['gcc_reward'],
                'unified_gcc': uni['gcc_reward'],
                'baseline_edge_diff': base['edge_diff'],
                'unified_edge_diff': uni['edge_diff'],
                'baseline_time': base['time'],
                'unified_time': uni['time'],
            })
            baseline_rewards.append(base['unified_reward'])
            unified_rewards.append(uni['unified_reward'])
        except Exception as e:
            print(f"图 {name} 对比失败: {e}")
            continue

    if not rows:
        print("没有成功对比任何图。")
        return

    # 保存 CSV
    if args.output is None:
        args.output = os.path.join(ROOT, 'results', 'unified_ppo_benchmark.csv')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果已保存: {args.output}")

    # 统计
    def avg(key):
        return np.mean([r[key] for r in rows])

    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"平均统一奖励 | 基线: {avg('baseline_reward'):.4f}  统一PPO: {avg('unified_reward'):.4f}")
    print(f"平均GCC奖励  | 基线: {avg('baseline_gcc'):.4f}  统一PPO: {avg('unified_gcc'):.4f}")
    print(f"平均边差异   | 基线: {avg('baseline_edge_diff'):.1f}  统一PPO: {avg('unified_edge_diff'):.1f}")
    print(f"平均时间(s)  | 基线: {avg('baseline_time'):.3f}  统一PPO: {avg('unified_time'):.3f}")
    wins = sum(1 for r in rows if r['unified_reward'] > r['baseline_reward'])
    print(f"统一PPO获胜次数: {wins}/{len(rows)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
