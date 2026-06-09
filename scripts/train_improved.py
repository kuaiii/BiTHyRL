#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BiT-HyRL 改进版训练脚本

使用改进的 Actor-Critic 架构和攻击感知奖励函数训练控制器选择策略

使用方法:
    python scripts/train_improved.py --epochs 300 --graphs 50
    python scripts/train_improved.py --epochs 500 --network ba --nodes 200
    python scripts/train_improved.py --resume src/train/v1/checkpoints/rl_agent_improved.pth --epochs 100
"""
import sys
import os
import argparse
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from src.bit_hyrl.improved_training import (
    train_improved,
    evaluate_improved_model,
    calculate_attack_aware_reward,
)
from src.topology.generators import generate_network
from src.bit_hyrl import config


def generate_training_graphs(num_graphs=50, network_type='ba', nodes_range=(100, 300)):
    """
    生成训练图
    
    Args:
        num_graphs: 图数量
        network_type: 网络类型 (ba, ws, random, er)
        nodes_range: 节点数范围
        
    Returns:
        list: NetworkX 图列表
    """
    graphs = []
    
    print(f"Generating {num_graphs} {network_type.upper()} graphs...")
    
    for i in range(num_graphs):
        n = random.randint(nodes_range[0], nodes_range[1])
        
        try:
            if network_type == 'ba':
                # Barabási-Albert 无标度网络
                m = random.choice([3, 4, 5])
                G = nx.barabasi_albert_graph(n, m)
            elif network_type == 'ws':
                # Watts-Strogatz 小世界网络
                k = random.choice([4, 6, 8])
                p = random.uniform(0.1, 0.3)
                G = nx.watts_strogatz_graph(n, k, p)
            elif network_type == 'er':
                # Erdős-Rényi 随机网络
                p = random.uniform(0.02, 0.1)
                G = nx.erdos_renyi_graph(n, p)
            else:
                # 默认 BA 网络
                G = nx.barabasi_albert_graph(n, 4)
            
            # 确保连通
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
            if G.number_of_nodes() >= 10:
                graphs.append(G)
                
        except Exception as e:
            print(f"  Warning: Failed to generate graph {i}: {e}")
            continue
    
    print(f"Generated {len(graphs)} valid graphs")
    return graphs


def load_training_graphs(data_dir):
    """
    从目录加载训练图
    
    Args:
        data_dir: 数据目录
        
    Returns:
        list: NetworkX 图列表
    """
    import glob
    
    graphs = []
    
    # 支持多种格式
    patterns = ['*.edgelist', '*.gml', '*.graphml', '*.adjlist']
    
    for pattern in patterns:
        files = glob.glob(os.path.join(data_dir, '**', pattern), recursive=True)
        
        for f in files:
            try:
                if f.endswith('.edgelist'):
                    G = nx.read_edgelist(f)
                elif f.endswith('.gml'):
                    G = nx.read_gml(f)
                elif f.endswith('.graphml'):
                    G = nx.read_graphml(f)
                elif f.endswith('.adjlist'):
                    G = nx.read_adjlist(f)
                else:
                    continue
                
                # 转换为无向图
                if G.is_directed():
                    G = G.to_undirected()
                
                # 确保连通
                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                
                if G.number_of_nodes() >= 10:
                    graphs.append(G)
                    
            except Exception as e:
                print(f"  Warning: Failed to load {f}: {e}")
                continue
    
    print(f"Loaded {len(graphs)} graphs from {data_dir}")
    return graphs


def main():
    parser = argparse.ArgumentParser(
        description='BiT-HyRL Improved Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default=None,
                        help='训练数据目录（如果指定，从文件加载图）')
    parser.add_argument('--graphs', type=int, default=50,
                        help='生成的训练图数量（当不指定 data-dir 时）')
    parser.add_argument('--network', type=str, default='ba',
                        choices=['ba', 'ws', 'er', 'random'],
                        help='网络类型')
    parser.add_argument('--nodes-min', type=int, default=100,
                        help='最小节点数')
    parser.add_argument('--nodes-max', type=int, default=300,
                        help='最大节点数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--entropy-coef', type=float, default=0.05,
                        help='初始熵系数')
    parser.add_argument('--attack-steps', type=int, default=30,
                        help='奖励计算时的攻击步数')
    
    # 模型参数
    parser.add_argument('--no-node2vec', action='store_true',
                        help='不使用 Node2Vec 特征')
    parser.add_argument('--resume', type=str, default=None,
                        help='继续训练的模型路径')
    parser.add_argument('--save-path', type=str, default=None,
                        help='模型保存路径')
    
    # 评估参数
    parser.add_argument('--eval-graphs', type=int, default=10,
                        help='评估图数量')
    parser.add_argument('--eval-only', action='store_true',
                        help='仅评估，不训练')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--quiet', action='store_true',
                        help='减少输出')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 确定保存路径
    if args.save_path is None:
        args.save_path = os.path.join(config.MODEL_DIR, 'rl_agent_improved.pth')
    
    # 仅评估模式
    if args.eval_only:
        if not os.path.exists(args.save_path):
            print(f"Error: Model not found: {args.save_path}")
            sys.exit(1)
        
        print("Generating evaluation graphs...")
        eval_graphs = generate_training_graphs(
            args.eval_graphs, args.network,
            (args.nodes_min, args.nodes_max)
        )
        
        print("\nEvaluating model...")
        results = evaluate_improved_model(
            args.save_path, eval_graphs,
            use_node2vec=not args.no_node2vec,
            attack_steps=50,
            verbose=not args.quiet
        )
        
        if results:
            print(f"\nFinal R-value: {results['avg_r_value']:.4f}")
        
        return
    
    # 加载或生成训练图
    if args.data_dir and os.path.exists(args.data_dir):
        graphs = load_training_graphs(args.data_dir)
    else:
        graphs = generate_training_graphs(
            args.graphs, args.network,
            (args.nodes_min, args.nodes_max)
        )
    
    if not graphs:
        print("Error: No training graphs available")
        sys.exit(1)
    
    # 划分训练/验证集
    random.shuffle(graphs)
    split_idx = max(1, int(len(graphs) * 0.9))
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:] if split_idx < len(graphs) else graphs[-5:]
    
    print(f"\nTraining graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    
    # 训练
    print("\n" + "="*60)
    print("Starting Improved Training")
    print("="*60)
    
    result = train_improved(
        train_graphs,
        epochs=args.epochs,
        save_path=args.save_path,
        use_node2vec=not args.no_node2vec,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        attack_steps=args.attack_steps,
        verbose=not args.quiet,
        resume_from=args.resume,
    )
    
    if result:
        print(f"\nTraining completed!")
        print(f"Best reward: {result['best_reward']:.4f}")
        print(f"Model saved: {result['model_path']}")
        
        # 验证集评估
        if val_graphs:
            print("\nEvaluating on validation set...")
            val_results = evaluate_improved_model(
                args.save_path, val_graphs,
                use_node2vec=not args.no_node2vec,
                attack_steps=50,
                verbose=not args.quiet
            )
            
            if val_results:
                print(f"Validation R-value: {val_results['avg_r_value']:.4f}")
    else:
        print("Training failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
