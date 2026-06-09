# -*- coding: utf-8 -*-
"""
真实网络数据增强：根据 dataset/all/real 中的网络统计特征生成合成网络

为每个真实网络（或按节点/边数量分组）生成多种类型的合成网络：
- BA (Barabási-Albert): 无标度网络
- ER (Erdős-Rényi): 随机网络  
- WS (Watts-Strogatz): 小世界网络
- Bimodal: 双峰度分布网络

用法:
    python scripts/generate_real_network_augmentation.py --output dataset/all/real_augmented
    python scripts/generate_real_network_augmentation.py --output dataset/all/real_augmented --per-graph 3
"""

import os
import sys
import argparse
import glob
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.topology.generators import load_graph as _load_graph_file
from src.topology.generators import construct_random, construct_ba
from src.topology.reconstruction import create_bimodal_theoretical


def load_real_network_stats(data_dir, min_nodes=10, max_nodes=500):
    """
    加载真实网络并提取 (n, m) 统计
    
    Returns:
        list of (n, m, name) 去重后的网络规格
    """
    gml_files = glob.glob(os.path.join(data_dir, '**', '*.gml'), recursive=True)
    graphml_files = glob.glob(os.path.join(data_dir, '**', '*.graphml'), recursive=True)
    all_files = list(set(gml_files + graphml_files))
    
    # 只保留 .gml 或 .graphml，同一网络可能有两种格式，取其一
    seen_nm = set()
    stats = []
    
    for fpath in tqdm(all_files, desc="Loading real networks"):
        try:
            G, _ = _load_graph_file(fpath, verbose=False)
            if G is None:
                continue
            
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
            n = G.number_of_nodes()
            m = G.number_of_edges()
            
            if min_nodes <= n <= max_nodes and (n, m) not in seen_nm:
                seen_nm.add((n, m))
                name = os.path.basename(fpath).replace('.gml', '').replace('.graphml', '')
                stats.append((n, m, name))
        except Exception:
            continue
    
    return stats


def generate_ws_network(n, m, seed=None):
    """生成 Watts-Strogatz 小世界网络，目标边数 m"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if n < 3 or m < n - 1:
        return None
    
    # WS 需要 k (每个节点邻居数)，总边数约 n*k/2
    k = max(2, min(n - 1, int(round(2 * m / n))))
    p = 0.1  # 重连概率
    
    try:
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        if not nx.is_connected(G):
            return None
        current_m = G.number_of_edges()
        if current_m < m:
            attempts = 0
            while G.number_of_edges() < m and attempts < 10000:
                u, v = random.sample(range(n), 2)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                attempts += 1
        elif current_m > m:
            edges = list(G.edges())
            random.shuffle(edges)
            to_remove = min(current_m - m, len(edges) - (n - 1))  # 保持连通
            for e in edges[:to_remove]:
                G.remove_edge(*e)
                if not nx.is_connected(G):
                    G.add_edge(*e)
                    break
        return G if nx.is_connected(G) else None
    except Exception:
        return None


def generate_synthetic_for_spec(n, m, spec_type, seed):
    """根据规格 (n, m) 生成指定类型的合成网络"""
    if spec_type == 'BA':
        try:
            G = construct_ba(n, m)
            if G and nx.is_connected(G):
                return G
        except Exception:
            pass
    elif spec_type == 'ER':
        try:
            G = construct_random(n, m)
            if G and nx.is_connected(G):
                return G
        except Exception:
            pass
    elif spec_type == 'WS':
        try:
            G = generate_ws_network(n, m, seed)
            if G and nx.is_connected(G):
                return G
        except Exception:
            pass
    elif spec_type == 'Bimodal':
        try:
            G = create_bimodal_theoretical(n, m, seed=seed)
            if G and nx.is_connected(G):
                return G
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Generate augmented synthetic networks for real network training')
    parser.add_argument('--data-dir', type=str, default='dataset/all/real',
                        help='Real network directory')
    parser.add_argument('--output', type=str, default='dataset/all/real_augmented',
                        help='Output directory for augmented graphs')
    parser.add_argument('--per-graph', type=int, default=2,
                        help='Number of synthetic graphs per (n,m) spec per type (default: 2)')
    parser.add_argument('--types', type=str, default='BA,ER,Bimodal',
                        help='Comma-separated types: BA,ER,WS,Bimodal')
    parser.add_argument('--min-nodes', type=int, default=10)
    parser.add_argument('--max-nodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("Real Network Data Augmentation")
    print("="*60)
    
    stats = load_real_network_stats(args.data_dir, args.min_nodes, args.max_nodes)
    print(f"\nLoaded {len(stats)} unique (n, m) specs from real networks")
    
    types = [t.strip() for t in args.types.split(',')]
    print(f"Generating types: {types}")
    print(f"Per spec per type: {args.per_graph} graphs")
    
    total_generated = 0
    
    for n, m, name in tqdm(stats, desc="Generating"):
        for spec_type in types:
            for i in range(args.per_graph):
                seed = args.seed + hash((n, m, spec_type, i)) % (2**31)
                G = generate_synthetic_for_spec(n, m, spec_type, seed)
                if G is not None:
                    out_name = f"aug_{name}_{spec_type}_{i}.gml"
                    out_path = os.path.join(args.output, out_name)
                    try:
                        nx.write_gml(G, out_path)
                        total_generated += 1
                    except Exception:
                        pass
    
    print(f"\nGenerated {total_generated} synthetic graphs in {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
