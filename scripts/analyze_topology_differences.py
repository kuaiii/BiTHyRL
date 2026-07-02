#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 Unified-PPO v4 重构后的拓扑与 QDLM/GA 等强基线构造的拓扑差异。
输出结构指标、鲁棒性指标、边修改模式及诊断报告。
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
import network_construction as nc
from network_dismantling.unified_interface import dismantle
from src.bit_hyrl.reward import _simulate_attack_sequence


ADVERSARIAL_ATTACK_METHODS = ['degree', 'betweenness', 'pagerank', 'eigenvector', 'random']


def robustness_r(G, controllers, attack_method='degree', attack_ratio=0.15):
    """计算指定攻击下的 Robustness R（含控制器的最大连通分量保留率）。"""
    try:
        seq = dismantle(G, attack_method)
        if seq is None or len(seq) == 0:
            return np.nan
        retention = _simulate_attack_sequence(G, controllers, seq, attack_ratio)
        return retention
    except Exception as e:
        return np.nan


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
    metrics['max_degree'] = degrees.max() if len(degrees) else 0
    metrics['min_degree'] = degrees.min() if len(degrees) else 0
    sorted_deg = sorted(degrees, reverse=True)
    hub_count = max(1, int(n * 0.15))
    metrics['hub_ratio'] = hub_count / n if n > 0 else 0.0
    metrics['top2_ratio'] = (sorted_deg[0] + sorted_deg[1]) / sum(sorted_deg) if len(sorted_deg) >= 2 and sum(sorted_deg) > 0 else 0.0
    metrics['leaf_ratio'] = sum(1 for d in degrees if d <= 2) / n if n > 0 else 0.0

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
        bc = nx.edge_betweenness_centrality(G)
        bc_vals = list(bc.values())
        metrics['edge_betweenness_mean'] = np.mean(bc_vals)
        metrics['edge_betweenness_std'] = np.std(bc_vals)
        metrics['edge_betweenness_max'] = np.max(bc_vals)
    except Exception:
        metrics['edge_betweenness_mean'] = np.nan
        metrics['edge_betweenness_std'] = np.nan
        metrics['edge_betweenness_max'] = np.nan

    return metrics


def compute_controller_placement(G, k_ratio=0.1):
    """简单的度中心性控制器选择，仅用于计算拓扑鲁棒性。"""
    k = max(1, int(G.number_of_nodes() * k_ratio))
    deg = dict(G.degree())
    controllers = sorted(deg, key=deg.get, reverse=True)[:k]
    return controllers


def generate_unified_ppo_topology(G, model_path, B_budget=3, k_ratio=0.1, seed=42):
    """用 Unified-PPO v4 checkpoint 生成最终拓扑。"""
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

    result = unified_solve(
        G, model, B_budget=B_budget, k_ratio=k_ratio,
        embed_dim=in_channels - 3, seed=seed, deterministic=True
    )
    return result['G_final'], result['controllers']


def generate_topologies(G, dataset_name, model_path, B_budget=3, k_ratio=0.1, seed=42):
    """生成所有待对比拓扑。"""
    topologies = {'Baseline': G.copy()}

    # GA
    try:
        G_GA = nc.construct(G, algorithm='GA', seed=seed)
        if G_GA is not None:
            topologies['GA'] = G_GA
    except Exception as e:
        print(f"[{dataset_name}] GA failed: {e}")

    # QDLM
    try:
        G_QDLM = nc.construct(G, algorithm='QDLM', seed=seed, pop_size=20, max_iterations=25, verbose=False)
        if G_QDLM is not None:
            topologies['QDLM'] = G_QDLM
    except Exception as e:
        print(f"[{dataset_name}] QDLM failed: {e}")

    # BiT-HyRL bimodal theoretical
    try:
        G_BiT = create_bimodal_theoretical(G.number_of_nodes(), G.number_of_edges(), seed=seed)
        if G_BiT is not None:
            topologies['BiT-HyRL'] = G_BiT
    except Exception as e:
        print(f"[{dataset_name}] BiT-HyRL failed: {e}")

    # Unified-PPO
    try:
        G_ppo, controllers = generate_unified_ppo_topology(G, model_path, B_budget=B_budget, k_ratio=k_ratio, seed=seed)
        if G_ppo is not None:
            topologies['Unified-PPO'] = G_ppo
            return topologies, controllers
    except Exception as e:
        print(f"[{dataset_name}] Unified-PPO failed: {e}")

    return topologies, None


def edge_modifications(G_orig, G_final):
    """返回 Unified-PPO 的边修改信息。"""
    E_orig = set(tuple(sorted((u, v))) for u, v in G_orig.edges())
    E_final = set(tuple(sorted((u, v))) for u, v in G_final.edges())
    added = E_final - E_orig
    removed = E_orig - E_final
    return added, removed


def analyze_dataset(G, dataset_name, model_path, k_ratio=0.1, seed=42, attack_ratio=0.15):
    """分析单个数据集的所有拓扑。"""
    print(f"\n[{dataset_name}] Generating topologies...")
    topologies, ppo_controllers = generate_topologies(
        G, dataset_name, model_path, B_budget=3, k_ratio=k_ratio, seed=seed
    )

    records = []
    edge_changes = {}

    for method, G_method in topologies.items():
        if G_method is None:
            continue
        rec = {'dataset': dataset_name, 'method': method}
        rec.update(compute_topology_metrics(G_method))

        # 用简单控制器计算鲁棒性（统一控制器选择，避免控制器差异干扰拓扑对比）
        controllers = compute_controller_placement(G_method, k_ratio=k_ratio)
        for attack in ADVERSARIAL_ATTACK_METHODS:
            rec[f'R_{attack}'] = robustness_r(G_method, controllers, attack_method=attack, attack_ratio=attack_ratio)

        # Unified-PPO 边修改
        if method == 'Unified-PPO':
            added, removed = edge_modifications(G, G_method)
            edge_changes['added'] = list(added)
            edge_changes['removed'] = list(removed)
            edge_changes['n_added'] = len(added)
            edge_changes['n_removed'] = len(removed)
            if ppo_controllers:
                edge_changes['controllers'] = list(ppo_controllers)

        records.append(rec)

    return records, edge_changes


def _df_to_md(df):
    """将 DataFrame / Series 转为简单 Markdown 表格，无需 tabulate。"""
    if isinstance(df, pd.Series):
        df = df.reset_index()
    lines = []
    headers = [str(c) for c in df.columns]
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for _, row in df.iterrows():
        vals = [str(v) for v in row.values]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def compute_gap_to_best(df):
    """计算每个场景 Unified-PPO 与最优基线的差距。"""
    rows = []
    for (dataset, attack), g in df.groupby(['dataset', 'attack']):
        target = g[g['method'] == 'Unified-PPO']
        if len(target) == 0:
            continue
        target_r = target['Robustness'].values[0]
        best_baseline_r = g[g['method'] != 'Unified-PPO']['Robustness'].max()
        rows.append({
            'dataset': dataset,
            'attack': attack,
            'Unified-PPO_R': target_r,
            'Best_Baseline_R': best_baseline_r,
            'gap': best_baseline_r - target_r,
        })
    return pd.DataFrame(rows)


def plot_metric_comparison(df, output_dir, metric):
    """绘制各方法在某指标上的对比图。"""
    plt.figure(figsize=(14, 6))
    methods = sorted(df['method'].unique())
    datasets = sorted(df['dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.12
    for i, method in enumerate(methods):
        sub = df[df['method'] == method].set_index('dataset').reindex(datasets)
        plt.bar(x + i * width, sub[metric].values, width, label=method)
    plt.xlabel('Dataset')
    plt.ylabel(metric)
    plt.title(f'{metric} by Method and Dataset')
    plt.xticks(x + width * (len(methods) - 1) / 2, datasets, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{metric}_comparison.png', dpi=150)
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    """绘制结构指标与鲁棒性差距的相关性热力图。"""
    numeric_cols = [c for c in df.columns if c not in ['dataset', 'method']]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    plt.title('Topology Metrics Correlation')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='src/train/v1/checkpoints/unified_ppo_agent_full_v4.pth')
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='results/topology_comparison_v4')
    parser.add_argument('--k-ratio', type=float, default=0.1)
    parser.add_argument('--attack-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])

    all_records = []
    all_edge_changes = {}

    for dataset in datasets:
        gml_path = data_dir / f'{dataset}.gml'
        if not gml_path.exists():
            print(f"[warn] {gml_path} not found, skipping.")
            continue
        G, _ = load_graph(str(gml_path), verbose=False)
        if G is None:
            continue

        records, edge_changes = analyze_dataset(
            G, dataset, args.model_path,
            k_ratio=args.k_ratio, seed=args.seed, attack_ratio=args.attack_ratio
        )
        all_records.extend(records)
        all_edge_changes[dataset] = edge_changes

    # 保存长表
    df_long = pd.DataFrame(all_records)
    df_long.to_csv(output_dir / 'topology_metrics_long.csv', index=False)

    # 保存边修改信息
    with open(output_dir / 'edge_changes.json', 'w') as f:
        json.dump(all_edge_changes, f, indent=2)

    # 宽表：每个数据集 × 方法
    df_wide = df_long.copy()

    # 按攻击类型拆分鲁棒性
    robust_rows = []
    for _, row in df_long.iterrows():
        for attack in ADVERSARIAL_ATTACK_METHODS:
            robust_rows.append({
                'dataset': row['dataset'],
                'method': row['method'],
                'attack': attack,
                'Robustness': row[f'R_{attack}'],
            })
    df_robust = pd.DataFrame(robust_rows)
    df_robust.to_csv(output_dir / 'robustness_by_attack.csv', index=False)

    # 计算差距
    gap_df = compute_gap_to_best(df_robust)
    gap_df.to_csv(output_dir / 'gap_to_best.csv', index=False)

    # 平均指标
    summary = df_long.groupby('method').agg({
        'density': 'mean',
        'avg_degree': 'mean',
        'degree_std': 'mean',
        'degree_skew': 'mean',
        'hub_ratio': 'mean',
        'top2_ratio': 'mean',
        'leaf_ratio': 'mean',
        'algebraic_connectivity': 'mean',
        'diameter': 'mean',
        'avg_shortest_path': 'mean',
        'clustering_coeff': 'mean',
        'edge_betweenness_mean': 'mean',
        'edge_betweenness_std': 'mean',
        'edge_betweenness_max': 'mean',
    }).round(4)
    summary.to_csv(output_dir / 'topology_summary.csv')

    # 平均鲁棒性
    robust_summary = df_robust.groupby('method')['Robustness'].agg(['mean', 'std']).round(4)
    robust_summary.to_csv(output_dir / 'robustness_summary.csv')

    # 按攻击类型
    attack_summary = df_robust.groupby(['method', 'attack'])['Robustness'].mean().unstack().round(4)
    attack_summary.to_csv(output_dir / 'robustness_by_attack_summary.csv')

    # 差距统计
    gap_summary = gap_df.groupby('attack')['gap'].agg(['mean', 'std', 'count']).round(4)
    gap_summary.to_csv(output_dir / 'gap_by_attack_summary.csv')

    # 可视化
    for metric in ['density', 'avg_degree', 'degree_std', 'degree_skew',
                   'algebraic_connectivity', 'clustering_coeff',
                   'edge_betweenness_mean', 'edge_betweenness_max']:
        if metric in df_long.columns:
            plot_metric_comparison(df_long, output_dir, metric)

    plot_correlation_heatmap(df_long, output_dir)

    # 生成报告
    report = []
    report.append("# Unified-PPO 拓扑对比分析报告\n")
    report.append("## 1. 平均拓扑指标\n")
    report.append(_df_to_md(summary))
    report.append("\n## 2. 平均鲁棒性（统一度中心性控制器）\n")
    report.append(_df_to_md(robust_summary))
    report.append("\n## 3. 按攻击类型的鲁棒性\n")
    report.append(_df_to_md(attack_summary))
    report.append("\n## 4. Unified-PPO 与最优基线差距\n")
    report.append(f"\n- 平均差距: {gap_df['gap'].mean():.4f}\n")
    report.append(f"- 中位数差距: {gap_df['gap'].median():.4f}\n")
    report.append(f"- Unified-PPO 最优场景数: {(gap_df['gap'] <= 0).sum()}/{len(gap_df)}\n")
    report.append("\n### 按攻击类型的平均差距\n")
    report.append(_df_to_md(gap_summary))
    report.append("\n### 差距最大的 10 个场景\n")
    report.append(_df_to_md(gap_df.nlargest(10, 'gap')))
    report.append("\n### 各数据集平均差距\n")
    report.append(_df_to_md(gap_df.groupby('dataset')['gap'].mean().sort_values(ascending=False).round(4).reset_index()))

    report.append("\n## 5. Unified-PPO 边修改统计\n")
    for dataset, changes in all_edge_changes.items():
        if changes:
            report.append(f"\n- **{dataset}**: 增加 {changes.get('n_added', 0)} 条边，删除 {changes.get('n_removed', 0)} 条边")

    report.append("\n## 6. 假设诊断（H1-H4）\n")
    report.append("- **H1 预算不足**: 观察 Unified-PPO 边修改数是否远少于 QDLM/GA。")
    report.append("- **H2 边选择错误**: 比较 Unified-PPO 增加/删除边的 betweenness 与 QDLM/GA 的差异。")
    report.append("- **H3 控制器不匹配**: 本报告使用统一度中心性控制器，可单独对比 PPO 控制器与度中心性控制器的差异。")
    report.append("- **H4 奖励不对齐**: 对比训练奖励与实际 R 的相关性，检查 degree_std/algebraic_connectivity 等指标是否与 gap 相关。")

    report_path = output_dir / 'report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n分析完成。报告: {report_path}")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
