# -*- coding: utf-8 -*-
"""
课程学习模型 (DynamicGATPolicy) 评估脚本

在测试集上评估 curriculum_final_dynamic_gat.pth 等课程模型。

Usage:
    # 默认评估 curriculum_final_dynamic_gat.pth
    python scripts/evaluate_curriculum.py

    # 指定模型路径
    python scripts/evaluate_curriculum.py --model src/train/v1/checkpoints/curriculum_phase3_dynamic_gat.pth

    # 指定测试数据目录
    python scripts/evaluate_curriculum.py --testdata dataset/testdata

    # 使用双峰拓扑进行测试（与 main.py 对比实验一致）
    python scripts/evaluate_curriculum.py --bimodal
"""

import os
import sys
import argparse
import glob

import torch
import numpy as np
import networkx as nx
from tqdm import tqdm

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.bit_hyrl import config
from src.bit_hyrl.gnn_model import (
    DynamicGATPolicy,
    get_unified_features,
    graph_to_pyg_data,
    compute_coverage_features,
    compute_dispersion_features,
)
from src.bit_hyrl.reward import (
    calculate_unified_gcc_reward,
    get_curriculum_reward_fn,
    CurriculumRewardPhase,
)
from src.topology.generators import load_graph as _load_graph
from src.topology.reconstruction import create_bimodal_theoretical

DEVICE = config.DEVICE


def load_checkpoint(model_path):
    """加载课程模型 checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")

    ck = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_type = ck.get('model_type', 'dynamic_gat')
    model_config = ck.get('model_config', {})

    if model_type == 'dynamic_gat':
        gat_config = {k: v for k, v in model_config.items() if k != 'num_heads'}
        if 'num_heads' in model_config:
            gat_config['heads'] = model_config['num_heads']
        model = DynamicGATPolicy(**gat_config).to(DEVICE)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    state_dict = ck.get('model_state_dict', ck)
    if not isinstance(state_dict, dict):
        state_dict = ck

    # 去掉 DataParallel 前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, ck


def select_controllers(model, G, k, use_dynamic_features=True):
    """
    使用 DynamicGATPolicy 贪婪选择 k 个控制器
    """
    node_features, scale_encoding, node_list, raw_dim = get_unified_features(G, device=DEVICE)
    x = model.feature_projector(node_features, raw_dim)
    _, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)

    num_nodes = len(node_list)
    selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    centers = []

    with torch.no_grad():
        for step in range(k):
            if use_dynamic_features and centers:
                coverage_feat = compute_coverage_features(G, centers, node_list, device=DEVICE)
                dispersion_feat = compute_dispersion_features(G, centers, node_list, device=DEVICE)
                dynamic_features = torch.cat([coverage_feat, dispersion_feat], dim=-1)
            else:
                dynamic_features = torch.zeros((num_nodes, 6), device=DEVICE)

            action, _, _, _ = model.get_action(
                x, edge_index, scale_encoding=scale_encoding,
                dynamic_features=dynamic_features,
                selected_mask=selected_mask, deterministic=True
            )
            action_idx = action.item()
            centers.append(node_list[action_idx])
            selected_mask = selected_mask.clone()
            selected_mask[action_idx] = True

    return centers


def load_test_graphs(testdata_dir, use_bimodal=False):
    """加载测试图"""
    test_graphs = []
    for ext in ['*.gml', '*.graphml']:
        for fpath in glob.glob(os.path.join(testdata_dir, ext)):
            try:
                G, _ = _load_graph(fpath)
                if G is None or G.number_of_nodes() < 10:
                    continue
                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                if G.number_of_nodes() >= 10:
                    name = os.path.basename(fpath)
                    if use_bimodal:
                        n, m = G.number_of_nodes(), G.number_of_edges()
                        G_bimodal = create_bimodal_theoretical(n, m, seed=42)
                        test_graphs.append((name, G_bimodal))
                    else:
                        test_graphs.append((name, G))
            except Exception:
                pass
    return test_graphs


def evaluate(model, test_graphs, k_ratio=0.1, reward_fn=None):
    """
    在测试集上评估模型
    """
    if reward_fn is None:
        reward_fn = calculate_unified_gcc_reward

    results = []
    scale_results = {'tiny (<50)': [], 'small (50-100)': [], 'medium (100-200)': [], 'large (>=200)': []}

    for name, G in tqdm(test_graphs, desc='评估', ncols=80):
        n = G.number_of_nodes()
        k = max(1, int(n * k_ratio))

        try:
            centers = select_controllers(model, G, k)
            reward = reward_fn(G, centers)
        except Exception as e:
            print(f"  {name} 失败: {e}")
            continue

        results.append({'name': name, 'nodes': n, 'k': k, 'reward': reward})

        if n < 50:
            scale_results['tiny (<50)'].append(reward)
        elif n < 100:
            scale_results['small (50-100)'].append(reward)
        elif n < 200:
            scale_results['medium (100-200)'].append(reward)
        else:
            scale_results['large (>=200)'].append(reward)

    return results, scale_results


def main():
    parser = argparse.ArgumentParser(description='评估课程学习模型')
    parser.add_argument('--model', type=str, default='src/train/v1/checkpoints/curriculum_final_dynamic_gat.pth',
                        help='模型路径')
    parser.add_argument('--testdata', type=str, default='dataset/testdata',
                        help='测试数据目录')
    parser.add_argument('--k-ratio', type=float, default=0.1,
                        help='控制器比例')
    parser.add_argument('--bimodal', action='store_true',
                        help='使用双峰拓扑测试（与 main.py 一致）')
    parser.add_argument('--reward', type=str, default='unified_gcc',
                        choices=['unified_gcc', 'phase1', 'phase2', 'phase3'],
                        help='奖励函数类型')
    args = parser.parse_args()

    # 加载模型
    print(f"加载模型: {args.model}")
    model, ck = load_checkpoint(args.model)
    best_reward = ck.get('best_reward', 0)
    print(f"  Checkpoint best reward: {best_reward:.4f}\n")

    # 加载测试图
    testdata_dir = os.path.join(PROJECT_ROOT, args.testdata) if not os.path.isabs(args.testdata) else args.testdata
    if not os.path.exists(testdata_dir):
        print(f"测试数据目录不存在: {testdata_dir}")
        return

    test_graphs = load_test_graphs(testdata_dir, use_bimodal=args.bimodal)
    if not test_graphs:
        print("未找到有效的测试图")
        return

    print(f"测试图数量: {len(test_graphs)}")
    if args.bimodal:
        print("拓扑模式: 双峰 (create_bimodal_theoretical)")
    print("")

    # 选择奖励函数
    if args.reward == 'unified_gcc':
        reward_fn = calculate_unified_gcc_reward
    else:
        phase_map = {'phase1': CurriculumRewardPhase.PHASE1, 'phase2': CurriculumRewardPhase.PHASE2, 'phase3': CurriculumRewardPhase.PHASE3}
        reward_fn = get_curriculum_reward_fn(phase_map[args.reward])

    # 评估
    results, scale_results = evaluate(model, test_graphs, k_ratio=args.k_ratio, reward_fn=reward_fn)

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"平均奖励: {np.mean([r['reward'] for r in results]):.4f}")
    print(f"最小奖励: {np.min([r['reward'] for r in results]):.4f}")
    print(f"最大奖励: {np.max([r['reward'] for r in results]):.4f}")
    print("")
    print("按规模:")
    for scale, rewards in scale_results.items():
        if rewards:
            print(f"  {scale}: 平均 {np.mean(rewards):.4f}, 数量 {len(rewards)}")
    print("")
    print("各图详情:")
    for r in sorted(results, key=lambda x: x['name']):
        print(f"  {r['name']:<25} N={r['nodes']:<4} k={r['k']:<3} reward={r['reward']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
