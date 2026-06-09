#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试改进版训练效果

对比:
1. 原始 REINFORCE 训练
2. 改进版 Actor-Critic 训练
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import os
import sys
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 导入模块
from src.bit_hyrl import config
from src.bit_hyrl.improved_training import (
    train_improved,
    evaluate_improved_model,
    calculate_attack_aware_reward,
    select_with_improved_model,
)
from src.bit_hyrl.training import train_offline_optimized


def generate_test_graphs(num_graphs=30, nodes_range=(100, 200)):
    """生成测试图"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(nodes_range[0], nodes_range[1])
        m = random.choice([3, 4, 5])
        G = nx.barabasi_albert_graph(n, m)
        if nx.is_connected(G):
            graphs.append(G)
    return graphs


def evaluate_baseline(G, centers, attack_steps=50):
    """评估基线（高度数选择）"""
    return calculate_attack_aware_reward(G, centers, attack_steps=attack_steps)


def select_high_degree(G, k):
    """选择高度数节点（朴素基线）"""
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    return sorted_nodes[:k]


def select_safe_degree(G, k):
    """选择安全位置的高度数节点（改进基线）"""
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    # 跳过前 10% 的高度数节点（攻击目标）
    safe_start = max(1, G.number_of_nodes() // 10)
    return sorted_nodes[safe_start:safe_start + k]


def simulate_attack(G, centers, attack_steps=50):
    """模拟攻击，返回 GCC 比例序列"""
    centers_set = set(centers)
    total_nodes = G.number_of_nodes()
    
    G_sim = G.copy()
    remaining_centers = centers_set.copy()
    
    y_values = [1.0]
    
    for step in range(min(attack_steps, total_nodes - 1)):
        if G_sim.number_of_nodes() <= 1:
            y_values.append(0.0)
            continue
        
        # Degree attack
        degrees = dict(G_sim.degree())
        if not degrees:
            y_values.append(0.0)
            continue
        target = max(degrees, key=degrees.get)
        
        G_sim.remove_node(target)
        remaining_centers.discard(target)
        
        if G_sim.number_of_nodes() == 0 or len(remaining_centers) == 0:
            y_values.append(0.0)
        else:
            components = list(nx.connected_components(G_sim))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            if controlled_sizes:
                y_values.append(max(controlled_sizes) / total_nodes)
            else:
                y_values.append(0.0)
    
    return y_values


def main():
    print("="*60)
    print("BiT-HyRL 改进版训练测试")
    print("="*60)
    
    # 生成训练和测试图
    print("\n[1] 生成训练图...")
    train_graphs = generate_test_graphs(30, (100, 200))
    print(f"    生成 {len(train_graphs)} 个训练图")
    
    print("\n[2] 生成测试图...")
    test_graphs = generate_test_graphs(10, (150, 250))
    print(f"    生成 {len(test_graphs)} 个测试图")
    
    # 改进版训练
    print("\n[3] 改进版 Actor-Critic 训练...")
    improved_model_path = os.path.join(config.MODEL_DIR, 'rl_agent_improved_test.pth')
    
    result = train_improved(
        train_graphs,
        epochs=100,  # 快速测试用较少轮数
        save_path=improved_model_path,
        use_node2vec=True,
        lr=0.0003,
        attack_steps=30,
        verbose=True,
    )
    
    if result:
        print(f"    训练完成，最佳奖励: {result['best_reward']:.4f}")
    
    # 评估
    print("\n[4] 评估不同方法...")
    
    methods = {
        'High Degree (Naive)': select_high_degree,
        'Safe Degree (Improved Baseline)': select_safe_degree,
    }
    
    results = {name: [] for name in methods}
    results['Improved Actor-Critic'] = []
    
    for i, G in enumerate(test_graphs):
        n = G.number_of_nodes()
        k = max(1, int(n * 0.1))
        
        print(f"    测试图 {i+1}/{len(test_graphs)} (n={n}, k={k})")
        
        # 基线方法
        for name, select_fn in methods.items():
            centers = select_fn(G, k)
            r = calculate_attack_aware_reward(G, centers, attack_steps=50)
            results[name].append(r)
        
        # 改进版模型
        centers = select_with_improved_model(G, k, improved_model_path, use_node2vec=True)
        r = calculate_attack_aware_reward(G, centers, attack_steps=50)
        results['Improved Actor-Critic'].append(r)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果 (R值，越大越好)")
    print("="*60)
    print(f"{'方法':<30} {'平均R值':<10} {'最小':<10} {'最大':<10}")
    print("-"*60)
    
    for name, values in results.items():
        avg = np.mean(values)
        min_v = np.min(values)
        max_v = np.max(values)
        print(f"{name:<30} {avg:<10.4f} {min_v:<10.4f} {max_v:<10.4f}")
    
    # 可视化单个测试图的攻击过程
    print("\n[5] 可视化攻击过程...")
    
    test_G = test_graphs[0]
    n = test_G.number_of_nodes()
    k = max(1, int(n * 0.1))
    
    plt.figure(figsize=(10, 6))
    
    # High Degree
    centers = select_high_degree(test_G, k)
    y_values = simulate_attack(test_G, centers, attack_steps=50)
    plt.plot(y_values, label='High Degree (Naive)', linestyle='--', alpha=0.7)
    
    # Safe Degree
    centers = select_safe_degree(test_G, k)
    y_values = simulate_attack(test_G, centers, attack_steps=50)
    plt.plot(y_values, label='Safe Degree', linestyle='-.', alpha=0.7)
    
    # Improved
    centers = select_with_improved_model(test_G, k, improved_model_path, use_node2vec=True)
    y_values = simulate_attack(test_G, centers, attack_steps=50)
    plt.plot(y_values, label='Improved Actor-Critic', linewidth=2)
    
    plt.xlabel('Attack Step')
    plt.ylabel('GCC Ratio')
    plt.title(f'Attack Simulation (n={n}, k={k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = 'results/improved_training_test.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    图表已保存: {plot_path}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == '__main__':
    main()
