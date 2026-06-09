# -*- coding: utf-8 -*-
"""
生成训练数据：为特定节点规模范围生成合成网络

用法:
  # 生成 50-150 节点的 BA 网络（默认100个）
  python scripts/generate_training_data.py --min_nodes 50 --max_nodes 150 --count 100

  # 按类型各生成 200 个（BA/WS/RA 各 200，共 600 个，20-100 节点）
  python scripts/generate_training_data.py --min_nodes 20 --max_nodes 100 --count_per_type 200 --types ba ws er
"""
import os
import sys
import argparse
import random
import numpy as np
import networkx as nx
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'dataset', 'all', 'syn')


def generate_ba_graph(n, m=3, seed=None):
    """生成 BA 无标度网络"""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G


def generate_er_graph(n, p=None, seed=None):
    """生成 ER 随机网络"""
    if p is None:
        # 确保连通，设置 p 略高于连通阈值
        p = 2 * np.log(n) / n
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    # 确保连通
    if not nx.is_connected(G):
        # 连接所有连通分量
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            G.add_edge(u, v)
    
    return G


def generate_ws_graph(n, k=4, p=0.3, seed=None):
    """生成 WS 小世界网络"""
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G


def generate_powerlaw_cluster_graph(n, m=3, p=0.5, seed=None):
    """生成幂律聚类网络"""
    try:
        G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
        return G
    except:
        return generate_ba_graph(n, m, seed)


def ensure_connected(G):
    """确保图连通"""
    if nx.is_connected(G):
        return G
    
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        u = random.choice(list(components[i]))
        v = random.choice(list(components[i + 1]))
        G.add_edge(u, v)
    
    return G


def generate_graphs(min_nodes, max_nodes, count=None, graph_types=['ba'], output_dir=OUTPUT_DIR, count_per_type=None):
    """
    生成指定规模范围的网络

    Args:
        min_nodes: 最小节点数
        max_nodes: 最大节点数
        count: 总生成数量（与 count_per_type 二选一）
        graph_types: 网络类型列表 ['ba', 'er', 'ws', 'plc']，er 即 RA 随机图
        output_dir: 输出目录
        count_per_type: 每种类型各生成数量；若指定则总数量 = count_per_type * len(graph_types)
    """
    os.makedirs(output_dir, exist_ok=True)
    if count_per_type is not None:
        total = count_per_type * len(graph_types)
        print(f"每种类型生成 {count_per_type} 个，共 {total} 个网络 (节点数: {min_nodes}-{max_nodes})")
    else:
        total = count or 100
        print(f"生成 {total} 个网络 (节点数: {min_nodes}-{max_nodes})")
    print(f"网络类型: {graph_types}")
    print(f"输出目录: {output_dir}")

    # 统计已有的同类型图数量
    existing_counts = {}
    for t in graph_types:
        prefix = f"{t.upper()}_gen_"
        existing = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
        existing_counts[t] = len(existing)

    generated = 0
    failed = 0
    pbar = tqdm(total=total, desc="Generating")

    def try_generate_one(graph_type, n, seed):
        if graph_type == 'ba':
            m = random.randint(2, min(4, n // 2))
            G = generate_ba_graph(n, m, seed)
        elif graph_type == 'er':
            G = generate_er_graph(n, seed=seed)
        elif graph_type == 'ws':
            k = random.randint(4, min(8, n // 2))
            p = random.uniform(0.1, 0.5)
            G = generate_ws_graph(n, k, p, seed)
        elif graph_type == 'plc':
            m = random.randint(2, min(4, n // 2))
            p = random.uniform(0.3, 0.7)
            G = generate_powerlaw_cluster_graph(n, m, p, seed)
        else:
            return None
        G = ensure_connected(G)
        return G if nx.is_connected(G) else None

    if count_per_type is not None:
        # 按类型各生成 count_per_type 个
        for graph_type in graph_types:
            type_generated = 0
            while type_generated < count_per_type:
                n = random.randint(min_nodes, max_nodes)
                seed = random.randint(0, 1000000)
                try:
                    G = try_generate_one(graph_type, n, seed)
                    if G is None:
                        failed += 1
                        continue
                    idx = existing_counts[graph_type] + type_generated
                    filename = f"{graph_type.upper()}_gen_{n}n_{idx}.gml"
                    filepath = os.path.join(output_dir, filename)
                    nx.write_gml(G, filepath)
                    type_generated += 1
                    generated += 1
                    pbar.update(1)
                except Exception:
                    failed += 1
    else:
        type_generated = {t: 0 for t in graph_types}
        while generated < total:
            n = random.randint(min_nodes, max_nodes)
            graph_type = random.choice(graph_types)
            seed = random.randint(0, 1000000)
            try:
                G = try_generate_one(graph_type, n, seed)
                if G is None:
                    failed += 1
                    continue
                type_generated[graph_type] += 1
                idx = existing_counts[graph_type] + type_generated[graph_type]
                filename = f"{graph_type.upper()}_gen_{n}n_{idx}.gml"
                filepath = os.path.join(output_dir, filename)
                nx.write_gml(G, filepath)
                generated += 1
                pbar.update(1)
            except Exception:
                failed += 1

    pbar.close()
    print(f"\n生成完成: {generated} 个网络")
    if failed > 0:
        print(f"失败: {failed} 次尝试")
    return generated


def analyze_distribution(data_dir):
    """分析训练数据的节点数分布"""
    sizes = []
    
    for f in os.listdir(data_dir):
        if f.endswith('.gml'):
            try:
                G = nx.read_gml(os.path.join(data_dir, f))
                sizes.append(G.number_of_nodes())
            except:
                pass
    
    if not sizes:
        print("没有找到训练数据")
        return
    
    print(f"\n训练数据分布分析")
    print(f"{'='*50}")
    print(f"总图数: {len(sizes)}")
    print(f"节点数范围: {min(sizes)} - {max(sizes)}")
    print(f"平均节点数: {sum(sizes)/len(sizes):.1f}")
    print()
    
    # 按区间统计
    ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 1000)]
    print("节点数分布:")
    for lo, hi in ranges:
        count = sum(1 for s in sizes if lo <= s < hi)
        pct = count / len(sizes) * 100
        bar = '█' * int(pct / 2)
        print(f"  {lo:4d}-{hi:4d}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='生成训练数据')
    parser.add_argument('--min_nodes', type=int, default=50, help='最小节点数')
    parser.add_argument('--max_nodes', type=int, default=150, help='最大节点数')
    parser.add_argument('--count', type=int, default=None, help='总生成数量（与 --count_per_type 二选一）')
    parser.add_argument('--count_per_type', type=int, default=None,
                        help='每种类型各生成数量，如 200 表示 BA/WS/ER 各 200 个')
    parser.add_argument('--types', nargs='+', default=['ba'],
                        choices=['ba', 'er', 'ws', 'plc'],
                        help='网络类型: ba(无标度), er/RA(随机), ws(小世界), plc(幂律聚类)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--analyze', action='store_true', help='仅分析现有数据分布')

    args = parser.parse_args()

    if args.analyze:
        analyze_distribution(args.output_dir)
        return

    count = args.count
    if args.count_per_type is not None:
        count = None
    elif count is None:
        count = 100

    generate_graphs(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        count=count,
        graph_types=args.types,
        output_dir=args.output_dir,
        count_per_type=args.count_per_type,
    )
    
    # 分析最终分布
    analyze_distribution(args.output_dir)


if __name__ == '__main__':
    main()