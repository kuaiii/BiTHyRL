# -*- coding: utf-8 -*-
"""
BiT-HyRL 运行入口：重构 + 可选攻击仿真与绘图。

用法:
  python scripts/run_bit_hyrl.py --dataset GtsCe --rate 0.1
  python scripts/run_bit_hyrl.py --dataset GtsCe --rate 0.1 --run_attack --attack_mode degree
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import pickle
import time
from src.topology.generators import load_graph
from src.bit_hyrl import (
    bit_hyrl_reconstruct,
    simulate_attack_and_plot,
    config,
)


def _load_graph_from_args(args):
    if args.synthetic:
        from src.topology.generators import construct_ba
        return construct_ba(args.nodes, args.nodes * 2), None
    if args.dataset:
        for base in ('dataset/testdata', 'dataset/all'):
            p = os.path.join(ROOT, base, f'{args.dataset}.gml')
            if os.path.exists(p):
                G, pos = load_graph(p)
                return G, args.dataset
        raise FileNotFoundError(f"Dataset {args.dataset} not found")
    if args.input_graph:
        r = load_graph(args.input_graph)
        G = r[0] if isinstance(r, tuple) else r
        return G, os.path.splitext(os.path.basename(args.input_graph))[0]
    raise ValueError("Need --synthetic, --dataset, or --input_graph")


def main():
    ap = argparse.ArgumentParser(description='Run BiT-HyRL')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--input_graph', type=str)
    g.add_argument('--dataset', type=str)
    g.add_argument('--synthetic', action='store_true')
    ap.add_argument('--nodes', type=int, default=100)
    ap.add_argument('--rate', type=float, default=0.1)
    ap.add_argument('--hub_ratio', type=float, default=0.15)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--metric_type', type=str, default='robustness',
                    choices=['combined', 'robustness', 'csa', 'entropy', 'wcp'])
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--model_path', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--attack_mode', type=str, default='degree')
    ap.add_argument('--run_attack', action='store_true')
    ap.add_argument('--use_node2vec', action='store_true', default=True)
    ap.add_argument('--no_node2vec', dest='use_node2vec', action='store_false')
    ap.add_argument('--use_stepwise', action='store_true', default=True)
    ap.add_argument('--no_stepwise', dest='use_stepwise', action='store_false')
    ap.add_argument('--use_ci', action='store_true', default=True)
    ap.add_argument('--no_ci', dest='use_ci', action='store_false')
    args = ap.parse_args()

    print("=" * 60)
    print("BiT-HyRL 算法重构")
    print("=" * 60)
    G, name = _load_graph_from_args(args)
    print(f"网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    G_bimodal, controllers = bit_hyrl_reconstruct(
        G, controller_rate=args.rate, hub_ratio=args.hub_ratio,
        episodes=args.episodes, metric_type=args.metric_type, seed=args.seed,
        model_path=args.model_path, use_node2vec=args.use_node2vec,
        use_stepwise_reward=args.use_stepwise, use_ci=args.use_ci,
    )
    print(f"重构完成! 控制器: {sorted(controllers)}")

    out_dir = args.output_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"bit_hyrl_{name or 'run'}_{ts}"
    pkl_path = args.output or os.path.join(out_dir, f"{base}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'original_graph': G, 'bimodal_graph': G_bimodal, 'controllers': controllers,
            'parameters': {'rate': args.rate, 'hub_ratio': args.hub_ratio,
                          'episodes': args.episodes, 'metric_type': args.metric_type, 'seed': args.seed},
        }, f)
    print(f"结果已保存: {os.path.abspath(pkl_path)}")

    gml_path = pkl_path.replace('.pkl', '.gml')
    import networkx as nx
    nx.write_gml(G_bimodal, gml_path)
    print(f"GML 已保存: {os.path.abspath(gml_path)}")

    txt_path = pkl_path.replace('.pkl', '_controllers.txt')
    with open(txt_path, 'w') as f:
        f.write(f"BiT-HyRL 控制器\n")
        f.write(f"节点数: {G_bimodal.number_of_nodes()}, 边数: {G_bimodal.number_of_edges()}\n")
        f.write(f"控制器数: {len(controllers)}\n")
        f.write(f"控制器: {sorted(controllers)}\n")
    print(f"控制器列表: {os.path.abspath(txt_path)}")

    if args.run_attack:
        attack_dir = os.path.join(out_dir, f"attack_{args.attack_mode}")
        simulate_attack_and_plot(G_bimodal, controllers, args.attack_mode, attack_dir)
        print(f"攻击仿真结果: {os.path.abspath(attack_dir)}")

    print("=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
