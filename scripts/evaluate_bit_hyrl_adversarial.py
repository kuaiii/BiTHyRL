#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiT-HyRL 对抗式多攻击验证脚本。

在 dataset/testdata/ 上评估训练好的 GNN 模型，
使用 network_dismantling 中多种攻击方法进行验证，
输出 GCC AUC (R-value)、崩溃点、CSA/CCE/WCP 等综合指标。

用法:
  python scripts/evaluate_bit_hyrl_adversarial.py \
      --model src/train/v1/checkpoints/gnn_ppo_agent_adversarial.pth \
      --output results/adversarial_eval.csv
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import csv
import numpy as np
import networkx as nx
from tqdm import tqdm

from src.topology.generators import load_graph
from src.bit_hyrl import config
from src.bit_hyrl.selection import gnn_predict, load_gnn_model
from src.utils.logger import get_logger
import network_dismantling as nd
import network_metrics as nm

logger = get_logger(__name__)
DEVICE = config.DEVICE


# 默认攻击方法池（兼顾效率与多样性）
DEFAULT_ATTACK_METHODS = [
    'degree', 'betweenness', 'pagerank', 'eigenvector',
    'random', 'CI_L1', 'CI_L2', 'CoreHD', 'GND',
]


def load_test_graphs(data_dir=None, min_nodes=10):
    """从 dataset/testdata/ 加载测试图。"""
    if data_dir is None:
        data_dir = os.path.join(ROOT, 'dataset', 'testdata')

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"测试数据目录不存在: {data_dir}")

    gml_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.gml')])
    graphs = []
    names = []

    for filename in gml_files:
        filepath = os.path.join(data_dir, filename)
        name = os.path.splitext(filename)[0]
        try:
            G, _ = load_graph(filepath, verbose=False)
            if G is None or G.number_of_nodes() < min_nodes:
                continue
            if not nx.is_connected(G):
                continue
            graphs.append(G)
            names.append(name)
        except Exception as e:
            logger.warning(f"加载 {name} 失败: {e}")

    return graphs, names


def _load_model(model_path):
    """加载 GNN 模型（自动检测模型类型）。"""
    model, checkpoint, model_type = load_gnn_model(model_path)
    if model is None:
        raise RuntimeError(f"无法加载模型: {model_path}")
    in_channels = checkpoint.get('in_channels', 128)
    return model, in_channels, model_type


def evaluate_under_attack(G, centers, attack_method, attack_ratio=0.5):
    """
    使用指定攻击方法评估控制器部署。
    
    Returns:
        dict: {'gcc_auc': float, 'collapse_ratio': float, 'csa': float, 'cce': float, 'wcp': float}
    """
    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        return None

    try:
        sequence = nd.dismantle(G, method=attack_method)
    except Exception as e:
        logger.warning(f"攻击方法 {attack_method} 失败: {e}")
        return None

    num_remove = max(1, int(total_nodes * attack_ratio))
    attack_sequence = sequence[:num_remove]

    # 使用 network_metrics 计算攻击轨迹和指标
    try:
        # GCC 轨迹
        gcc_traj = nm.compute('attack_trajectory', G, attack_sequence=attack_sequence)
        x_curve = [p[1] for p in gcc_traj]
        y_curve = [p[2] / total_nodes for p in gcc_traj]
        gcc_auc = nm.compute('r_value_interpolated', x_curve, y_curve, num_points=101)

        # 崩溃点: GCC 降至 20% 时的移除比例
        collapse_ratio = None
        for p in gcc_traj:
            if p[2] <= total_nodes * 0.2:
                collapse_ratio = p[1]
                break
        if collapse_ratio is None and gcc_traj:
            collapse_ratio = gcc_traj[-1][1]

        # 功能指标（在攻击后网络上计算）
        G_attacked = G.copy()
        alive_centers = set(centers)
        for node in attack_sequence:
            if node in G_attacked:
                G_attacked.remove_node(node)
            if node in alive_centers:
                alive_centers.remove(node)

        if G_attacked.number_of_nodes() == 0 or not alive_centers:
            csa = cce = wcp = 0.0
        else:
            csa = nm.compute('csa', G_attacked, centers=list(alive_centers))
            cce = nm.compute('cce', G_attacked, centers=list(alive_centers))
            wcp = nm.compute('wcp', G_attacked, centers=list(alive_centers))

        return {
            'gcc_auc': gcc_auc,
            'collapse_ratio': collapse_ratio,
            'csa': csa,
            'cce': cce,
            'wcp': wcp,
        }
    except Exception as e:
        logger.warning(f"评估失败: {e}")
        return None


def evaluate_model(model_path, test_graphs, names, k_ratio=0.1, attack_methods=None, attack_ratio=0.5):
    """在测试图上对每种攻击方法评估模型。"""
    if attack_methods is None:
        attack_methods = DEFAULT_ATTACK_METHODS

    # 过滤不可用的方法
    available_methods = []
    for m in attack_methods:
        try:
            if m in nd.list_methods():
                available_methods.append(m)
        except Exception:
            pass

    if not available_methods:
        raise RuntimeError("没有可用的攻击方法")

    print(f"可用攻击方法 ({len(available_methods)}): {', '.join(available_methods)}")

    model, in_channels, model_type = _load_model(model_path)

    results = []

    for G, name in tqdm(list(zip(test_graphs, names)), desc='Evaluating datasets', ncols=100):
        n_nodes = G.number_of_nodes()
        k = max(1, int(n_nodes * k_ratio))

        try:
            # GNN 选择控制器
            centers = gnn_predict(G, k, model=model, deterministic=True, embed_dim=in_channels)
        except Exception as e:
            logger.warning(f"GNN 预测 {name} 失败: {e}")
            continue

        for method in available_methods:
            metrics = evaluate_under_attack(G, centers, method, attack_ratio=attack_ratio)
            if metrics is None:
                continue
            results.append({
                'dataset': name,
                'nodes': n_nodes,
                'controllers': k,
                'attack_method': method,
                **metrics,
            })

    return results


def aggregate_results(results):
    """按数据集和方法聚合结果。"""
    from collections import defaultdict

    by_method = defaultdict(list)
    by_dataset_method = defaultdict(list)

    for r in results:
        by_method[r['attack_method']].append(r)
        by_dataset_method[(r['dataset'], r['attack_method'])].append(r)

    return by_method, by_dataset_method


def print_summary(by_method):
    """打印按攻击方法聚合的汇总表。"""
    print("\n" + "=" * 90)
    print("按攻击方法汇总 (跨数据集平均)")
    print("=" * 90)
    print(f"{'Attack Method':<18} {'GCC AUC':>10} {'Collapse':>10} {'CSA':>10} {'CCE':>10} {'WCP':>10} {'Count':>6}")
    print("-" * 90)

    for method in sorted(by_method.keys()):
        rows = by_method[method]
        print(
            f"{method:<18} "
            f"{np.mean([r['gcc_auc'] for r in rows]):>10.4f} "
            f"{np.mean([r['collapse_ratio'] for r in rows]):>10.4f} "
            f"{np.mean([r['csa'] for r in rows]):>10.4f} "
            f"{np.mean([r['cce'] for r in rows]):>10.4f} "
            f"{np.mean([r['wcp'] for r in rows]):>10.4f} "
            f"{len(rows):>6d}"
        )
    print("=" * 90)


def save_csv(results, output_path):
    """保存详细结果到 CSV。"""
    if not results:
        return
    keys = results[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n详细结果已保存: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate BiT-HyRL under multiple dismantling attacks'
    )
    parser.add_argument('--model', type=str, default=None,
                        help='训练好的 GNN 模型路径 (与 --models 互斥)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='多个模型路径，用于横向对比 (例如: --models model1.pth model2.pth)')
    parser.add_argument('--model-names', type=str, nargs='+', default=None,
                        help='模型显示名称 (与 --models 一一对应)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='测试数据目录 (默认: dataset/testdata)')
    parser.add_argument('--output', type=str, default='results/adversarial_eval.csv',
                        help='输出 CSV 路径')
    parser.add_argument('--k-ratio', type=float, default=0.1,
                        help='控制器比例 (默认: 0.1)')
    parser.add_argument('--attack-ratio', type=float, default=0.5,
                        help='验证时攻击比例 (默认: 0.5)')
    parser.add_argument('--attack-methods', type=str, nargs='+', default=None,
                        help='攻击方法列表 (默认使用内置方法池)')
    parser.add_argument('--min-nodes', type=int, default=10,
                        help='最小节点数过滤 (默认: 10)')
    return parser.parse_args()


def evaluate_multiple_models(model_paths, model_names, test_graphs, names, k_ratio, attack_methods, attack_ratio):
    """评估多个模型并返回合并结果。"""
    all_results = []
    for path, label in zip(model_paths, model_names):
        print(f"\n>>> 评估模型: {label} ({path})")
        results = evaluate_model(
            path, test_graphs, names,
            k_ratio=k_ratio,
            attack_methods=attack_methods,
            attack_ratio=attack_ratio,
        )
        for r in results:
            r['model_name'] = label
        all_results.extend(results)
    return all_results


def print_multi_model_summary(all_results):
    """打印多模型对比汇总表。"""
    from collections import defaultdict
    import numpy as np

    # 按 (model_name, attack_method) 聚合
    agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        key = (r['model_name'], r['attack_method'])
        for metric in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']:
            agg[key][metric].append(r[metric])

    # 按 model 分组
    model_groups = defaultdict(list)
    for (model, method), metrics in agg.items():
        model_groups[model].append((method, metrics))

    print("\n" + "=" * 110)
    print("多模型横向对比：按模型 × 攻击方法（跨数据集平均）")
    print("=" * 110)
    header = f"{'Model':<22} {'Attack':<14} {'GCC AUC':>9} {'Collapse':>9} {'CSA':>9} {'CCE':>9} {'WCP':>9} {'Count':>6}"
    print(header)
    print("-" * 110)

    for model in sorted(model_groups.keys()):
        for method, metrics in sorted(model_groups[model]):
            count = len(metrics['gcc_auc'])
            vals = {m: np.mean(metrics[m]) for m in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']}
            print(f"{model:<22} {method:<14} {vals['gcc_auc']:>9.4f} {vals['collapse_ratio']:>9.4f} "
                  f"{vals['csa']:>9.4f} {vals['cce']:>9.4f} {vals['wcp']:>9.4f} {count:>6}")
    print("=" * 110)

    # 按模型打印综合平均
    print("\n" + "=" * 90)
    print("模型综合排名（跨攻击方法平均）")
    print("=" * 90)
    print(f"{'Model':<22} {'GCC AUC':>9} {'Collapse':>9} {'CSA':>9} {'CCE':>9} {'WCP':>9}")
    print("-" * 90)
    model_avg = {}
    for model in sorted(model_groups.keys()):
        all_vals = defaultdict(list)
        for _, metrics in model_groups[model]:
            for m in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']:
                all_vals[m].extend(metrics[m])
        avg = {m: np.mean(all_vals[m]) for m in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']}
        model_avg[model] = avg
        print(f"{model:<22} {avg['gcc_auc']:>9.4f} {avg['collapse_ratio']:>9.4f} "
              f"{avg['csa']:>9.4f} {avg['cce']:>9.4f} {avg['wcp']:>9.4f}")
    print("=" * 90)

    # 各指标最佳模型
    print("\n" + "=" * 60)
    print("各指标最佳模型")
    print("=" * 60)
    for metric in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']:
        best = max(model_avg, key=lambda m: model_avg[m][metric])
        print(f"  {metric:<18} -> {best:<22} ({model_avg[best][metric]:.4f})")
    print("=" * 60)


def main():
    args = parse_args()

    # 解析模型列表
    if args.models:
        model_paths = args.models
        if args.model_names:
            model_names = args.model_names
        else:
            model_names = [os.path.basename(p).replace('.pth', '') for p in model_paths]
    elif args.model:
        model_paths = [args.model]
        model_names = [os.path.basename(args.model).replace('.pth', '')]
    else:
        print("错误: 必须指定 --model 或 --models")
        return

    print("=" * 60)
    print("BiT-HyRL Adversarial Multi-Attack Evaluation")
    print("=" * 60)
    print(f"Models: {', '.join(model_names)}")
    print(f"Attack ratio: {args.attack_ratio}")
    print(f"K-ratio: {args.k_ratio}")
    print("=" * 60)

    # 加载测试数据
    test_graphs, names = load_test_graphs(args.data_dir, min_nodes=args.min_nodes)
    if not test_graphs:
        print("错误: 没有加载到有效的测试图")
        return

    print(f"加载 {len(test_graphs)} 个测试数据集: {', '.join(names)}")

    # 评估
    if len(model_paths) == 1:
        results = evaluate_model(
            model_paths[0], test_graphs, names,
            k_ratio=args.k_ratio,
            attack_methods=args.attack_methods,
            attack_ratio=args.attack_ratio,
        )
        for r in results:
            r['model_name'] = model_names[0]
    else:
        results = evaluate_multiple_models(
            model_paths, model_names, test_graphs, names,
            k_ratio=args.k_ratio,
            attack_methods=args.attack_methods,
            attack_ratio=args.attack_ratio,
        )

    if not results:
        print("没有生成有效的评估结果")
        return

    # 聚合并打印
    if len(model_paths) > 1:
        print_multi_model_summary(results)
    else:
        by_method, _ = aggregate_results(results)
        print_summary(by_method)

    # 保存
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_csv(results, args.output)


if __name__ == '__main__':
    main()
