# -*- coding: utf-8 -*-
"""
评估不同阶段模型在真实网络上的表现

对比:
- Phase 3 模型
- Phase 4 模型  
- Phase 4 Enhanced 模型

在真实网络测试集上评估GCC保持率和R值。
"""

import os
import sys
import argparse
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.bit_hyrl import config
from src.bit_hyrl.gnn_model import DynamicGATPolicy
from src.bit_hyrl.selection import rl_select_controllers
from src.topology.generators import load_graph as _load_graph_file
from network_metrics import compute as _nm_compute
import glob

DEVICE = config.DEVICE


def load_model(model_path, model_type='dynamic_gat'):
    """加载训练好的模型"""
    model_config = {
        'in_channels': 64,
        'hidden_channels': 96,
        'scale_encoding_dim': 8,
        'dynamic_feature_dim': 6,
        'heads': 4,
        'num_layers': 3,
        'dropout': 0.15,
    }
    
    model = DynamicGATPolicy(**model_config).to(DEVICE)
    
    if os.path.exists(model_path):
        ck = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model_state_dict'], strict=False)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        return None
    
    return model


def evaluate_on_graph(G, model, k_ratio=0.1):
    """
    在单个图上评估模型
    
    Returns:
        dict: 包含R值、关键点GCC等指标
    """
    n = G.number_of_nodes()
    k = max(1, int(n * k_ratio))
    
    # 使用RL选择控制器
    try:
        centers = rl_select_controllers(
            G, k, model, 
            metric_type="robustness",
            use_dynamic_features=True
        )
    except Exception as e:
        print(f"Error selecting controllers: {e}")
        return None
    
    if not centers:
        return None
    
    centers_set = set(centers)
    
    # 计算R值（0-50%攻击范围）
    attack_ratios = np.linspace(0, 0.5, 21)
    gcc_values = []
    
    degrees = dict(G.degree())
    degree_targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    for ratio in attack_ratios:
        if ratio == 0:
            gcc_values.append(1.0)
            continue
        
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
        
        gcc_values.append(gcc_ratio)
    
    r_value = np.trapz(gcc_values, attack_ratios) / 0.5
    
    # 关键点GCC
    key_gcc_10 = gcc_values[2]  # 10%攻击
    key_gcc_20 = gcc_values[4]  # 20%攻击
    key_gcc_30 = gcc_values[6]  # 30%攻击
    
    return {
        'r_value': r_value,
        'gcc_10': key_gcc_10,
        'gcc_20': key_gcc_20,
        'gcc_30': key_gcc_30,
        'gcc_curve': gcc_values,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase Models on Real Networks')
    
    parser.add_argument('--test-dir', type=str, default='dataset/testdata',
                        help='Test data directory')
    parser.add_argument('--phase3-model', type=str, default='src/train/v1/checkpoints/curriculum_phase3_dynamic_gat.pth',
                        help='Phase 3 model path')
    parser.add_argument('--phase4-model', type=str, default='src/train/v1/checkpoints/curriculum_phase4_dynamic_gat.pth',
                        help='Phase 4 model path')
    parser.add_argument('--phase4-enhanced-model', type=str, default='src/train/v1/checkpoints/curriculum_phase4_enhanced_dynamic_gat.pth',
                        help='Phase 4 Enhanced model path')
    parser.add_argument('--k-ratio', type=float, default=0.1,
                        help='Controller ratio')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Evaluating Phase Models on Real Networks")
    print("="*60)
    
    # 加载测试图
    print(f"\nLoading test graphs from {args.test_dir}...")
    gml_files = glob.glob(os.path.join(args.test_dir, '*.gml'))
    
    test_graphs = []
    test_names = []
    
    for fpath in gml_files:
        try:
            G, _ = _load_graph_file(fpath, verbose=False)
            if G and nx.is_connected(G) and G.number_of_nodes() >= 10:
                test_graphs.append(G)
                test_names.append(os.path.basename(fpath).replace('.gml', ''))
        except:
            continue
    
    print(f"Loaded {len(test_graphs)} test graphs")
    
    # 加载模型
    print("\nLoading models...")
    models = {}
    
    if os.path.exists(args.phase3_model):
        models['Phase 3'] = load_model(args.phase3_model)
    
    if os.path.exists(args.phase4_model):
        models['Phase 4'] = load_model(args.phase4_model)
    
    if os.path.exists(args.phase4_enhanced_model):
        models['Phase 4 Enhanced'] = load_model(args.phase4_enhanced_model)
    
    if not models:
        print("Error: No models found!")
        return
    
    # 评估
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    results = {name: [] for name in models.keys()}
    
    for graph_name, G in zip(tqdm(test_names, desc="Evaluating"), test_graphs):
        print(f"\n{graph_name} (N={G.number_of_nodes()}, E={G.number_of_edges()})")
        print("-" * 60)
        
        for model_name, model in models.items():
            if model is None:
                continue
            
            result = evaluate_on_graph(G, model, args.k_ratio)
            
            if result:
                results[model_name].append(result)
                print(f"  {model_name:20s}: R={result['r_value']:.4f}, "
                      f"GCC@10%={result['gcc_10']:.3f}, "
                      f"GCC@20%={result['gcc_20']:.3f}, "
                      f"GCC@30%={result['gcc_30']:.3f}")
    
    # 汇总统计
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    for model_name in models.keys():
        if not results[model_name]:
            continue
        
        r_values = [r['r_value'] for r in results[model_name]]
        gcc_10_values = [r['gcc_10'] for r in results[model_name]]
        gcc_20_values = [r['gcc_20'] for r in results[model_name]]
        gcc_30_values = [r['gcc_30'] for r in results[model_name]]
        
        print(f"\n{model_name}:")
        print(f"  R-value:      {np.mean(r_values):.4f} ± {np.std(r_values):.4f}")
        print(f"  GCC @ 10%:    {np.mean(gcc_10_values):.4f} ± {np.std(gcc_10_values):.4f}")
        print(f"  GCC @ 20%:    {np.mean(gcc_20_values):.4f} ± {np.std(gcc_20_values):.4f}")
        print(f"  GCC @ 30%:    {np.mean(gcc_30_values):.4f} ± {np.std(gcc_30_values):.4f}")
    
    # 对比改进
    if 'Phase 4 Enhanced' in results and 'Phase 3' in results:
        if results['Phase 4 Enhanced'] and results['Phase 3']:
            r_improved = np.mean([r['r_value'] for r in results['Phase 4 Enhanced']]) - \
                        np.mean([r['r_value'] for r in results['Phase 3']])
            
            print(f"\n" + "="*60)
            print(f"Improvement (Phase 4 Enhanced vs Phase 3):")
            print(f"  R-value: {r_improved:+.4f} ({r_improved*100:+.2f}%)")
            print("="*60)


if __name__ == '__main__':
    main()
