#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BiT-HyRL 小规模网络优化训练脚本

用法:
    # 训练小规模网络专用模型
    python scripts/train_small_network.py --mode small --epochs 200
    
    # 训练增强型RL控制器选择器
    python scripts/train_small_network.py --mode enhanced --epochs 200
    
    # 使用自定义数据集训练
    python scripts/train_small_network.py --mode small --data-dir dataset/all/syn --epochs 100
    
    # 评估模型
    python scripts/train_small_network.py --mode evaluate --model src/train/v1/checkpoints/small_network_agent.pth
"""
import os
import sys
import argparse
import random
import pickle
import numpy as np
import torch
import networkx as nx
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.topology.generators import generate_ba_network, generate_ws_network, generate_random_network

logger = get_logger(__name__)


def generate_training_graphs(
    num_graphs=200,
    size_range=(10, 100),
    graph_types=['ba', 'ws', 'random'],
    seed=42,
):
    """
    生成训练图集合
    
    Args:
        num_graphs: 图数量
        size_range: 节点数范围 (min, max)
        graph_types: 图类型列表
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    graphs = []
    
    for i in range(num_graphs):
        # 随机选择图类型
        graph_type = random.choice(graph_types)
        
        # 随机选择节点数（偏向小图）
        # 使用指数分布使小图更多
        min_size, max_size = size_range
        size = int(min_size + (max_size - min_size) * (1 - np.exp(-random.random() * 2)))
        size = max(min_size, min(max_size, size))
        
        # 生成图
        if graph_type == 'ba':
            m = random.randint(2, min(5, size // 3))
            G = generate_ba_network(size, m)
        elif graph_type == 'ws':
            k = random.randint(4, min(8, size // 2))
            p = random.uniform(0.1, 0.5)
            G = generate_ws_network(size, k, p)
        elif graph_type == 'random':
            p = random.uniform(0.05, 0.3)
            G = generate_random_network(size, p)
        else:
            m = random.randint(2, 4)
            G = generate_ba_network(size, m)
        
        # 确保图连通
        if not nx.is_connected(G):
            # 取最大连通分量
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            # 重新编号节点
            G = nx.convert_node_labels_to_integers(G)
        
        if G.number_of_nodes() >= 5:
            graphs.append(G)
    
    # 按大小排序（课程学习）
    graphs.sort(key=lambda g: g.number_of_nodes())
    
    logger.info(f"生成了 {len(graphs)} 个训练图")
    logger.info(f"节点数范围: {graphs[0].number_of_nodes()} - {graphs[-1].number_of_nodes()}")
    
    return graphs


def load_graphs_from_directory(data_dir, max_graphs=None):
    """
    从目录加载图
    
    支持格式:
    - .pkl: pickle序列化的NetworkX图
    - .gml: GML格式
    - .edgelist: 边列表格式
    """
    graphs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return graphs
    
    # 查找所有图文件
    for ext in ['*.pkl', '*.gml', '*.edgelist', '*.gpickle']:
        for filepath in data_path.glob(ext):
            try:
                if ext == '*.pkl' or ext == '*.gpickle':
                    with open(filepath, 'rb') as f:
                        G = pickle.load(f)
                elif ext == '*.gml':
                    G = nx.read_gml(filepath)
                elif ext == '*.edgelist':
                    G = nx.read_edgelist(filepath)
                else:
                    continue
                
                if isinstance(G, nx.Graph) and G.number_of_nodes() >= 5:
                    graphs.append(G)
                    
                    if max_graphs and len(graphs) >= max_graphs:
                        break
            except Exception as e:
                logger.warning(f"加载 {filepath} 失败: {e}")
                continue
        
        if max_graphs and len(graphs) >= max_graphs:
            break
    
    if graphs:
        graphs.sort(key=lambda g: g.number_of_nodes())
        logger.info(f"从 {data_dir} 加载了 {len(graphs)} 个图")
    else:
        logger.warning(f"未能从 {data_dir} 加载任何图")
    
    return graphs


def train_small_network_model(args):
    """训练小规模网络专用模型"""
    from src.bit_hyrl.small_network_optimizer import (
        train_small_network_model,
        SmallNetworkTrainer,
    )
    
    # 加载或生成训练图
    if args.data_dir:
        graphs = load_graphs_from_directory(args.data_dir, args.max_graphs)
    else:
        graphs = generate_training_graphs(
            num_graphs=args.num_graphs,
            size_range=(args.min_size, args.max_size),
            seed=args.seed,
        )
    
    if not graphs:
        logger.error("没有可用的训练图")
        return
    
    # 保存路径
    save_path = args.output or 'src/train/v1/checkpoints/small_network_agent.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 训练
    result = train_small_network_model(
        graphs=graphs,
        epochs=args.epochs,
        save_path=save_path,
        lr=args.lr,
        verbose=True,
    )
    
    if result:
        print(f"\n模型已保存到: {save_path}")
        print(f"最佳奖励: {result['best_reward']:.4f}")


def train_enhanced_rl_model(args):
    """训练增强型RL控制器选择器"""
    from src.controller.enhanced_rl_selector import train_enhanced_rl_selector
    
    # 加载或生成训练图
    if args.data_dir:
        graphs = load_graphs_from_directory(args.data_dir, args.max_graphs)
    else:
        graphs = generate_training_graphs(
            num_graphs=args.num_graphs,
            size_range=(args.min_size, args.max_size),
            seed=args.seed,
        )
    
    if not graphs:
        logger.error("没有可用的训练图")
        return
    
    # 保存路径
    save_path = args.output or 'src/train/v1/checkpoints/enhanced_rl_selector.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 训练
    result, trainer = train_enhanced_rl_selector(
        graphs=graphs,
        epochs=args.epochs,
        save_path=save_path,
        lr=args.lr,
        verbose=True,
    )
    
    if result:
        print(f"\n模型已保存到: {save_path}")
        print(f"最佳奖励: {result['best_reward']:.4f}")


def evaluate_model(args):
    """评估模型"""
    model_path = args.model
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 生成测试图
    test_graphs = generate_training_graphs(
        num_graphs=50,
        size_range=(args.min_size, args.max_size),
        seed=args.seed + 1000,  # 不同的种子
    )
    
    # 根据模型类型加载
    if 'small_network' in model_path:
        from src.bit_hyrl.small_network_optimizer import (
            load_small_graph_model,
            SmallNetworkTrainer,
            calculate_adaptive_reward,
        )
        
        model = load_small_graph_model(model_path)
        if model is None:
            return
        
        trainer = SmallNetworkTrainer(model=model)
        reward_fn = calculate_adaptive_reward
        
    elif 'enhanced_rl' in model_path:
        from src.controller.enhanced_rl_selector import (
            load_enhanced_rl_model,
            EnhancedRLTrainer,
            MultiObjectiveReward,
        )
        
        model = load_enhanced_rl_model(model_path)
        if model is None:
            return
        
        trainer = EnhancedRLTrainer()
        trainer.model = model
        reward_fn = MultiObjectiveReward()
    else:
        logger.error("无法识别模型类型")
        return
    
    # 评估
    print(f"\n{'='*60}")
    print(f"评估模型: {os.path.basename(model_path)}")
    print(f"测试图数量: {len(test_graphs)}")
    print(f"{'='*60}\n")
    
    rewards = []
    for i, G in enumerate(test_graphs):
        k = max(1, int(G.number_of_nodes() * 0.1))
        centers = trainer.select_controllers(G, k, deterministic=True)
        
        if callable(reward_fn):
            reward = reward_fn(G, centers)
        else:
            reward = reward_fn(G, centers)
        
        rewards.append(reward)
        
        if (i + 1) % 10 == 0:
            print(f"进度: {i+1}/{len(test_graphs)}, 平均奖励: {np.mean(rewards):.4f}")
    
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"  平均奖励: {np.mean(rewards):.4f}")
    print(f"  最小奖励: {np.min(rewards):.4f}")
    print(f"  最大奖励: {np.max(rewards):.4f}")
    print(f"  标准差: {np.std(rewards):.4f}")
    print(f"{'='*60}\n")


def compare_methods(args):
    """对比不同方法"""
    from src.bit_hyrl.selection import hybrid_rl_select, ci_select_subset
    from src.bit_hyrl.small_network_optimizer import (
        load_small_graph_model,
        SmallNetworkTrainer,
        calculate_adaptive_reward,
    )
    from src.controller.enhanced_rl_selector import (
        load_enhanced_rl_model,
        EnhancedRLTrainer,
        MultiObjectiveReward,
    )
    
    # 生成测试图
    test_graphs = generate_training_graphs(
        num_graphs=30,
        size_range=(args.min_size, args.max_size),
        seed=args.seed + 2000,
    )
    
    # 加载模型
    small_model = load_small_graph_model('src/train/v1/checkpoints/small_network_agent.pth')
    enhanced_model = load_enhanced_rl_model('src/train/v1/checkpoints/enhanced_rl_selector.pth')
    
    reward_fn = MultiObjectiveReward()
    
    results = {
        'CI': [],
        'BiT-HyRL (原始)': [],
        'Small Network': [],
        'Enhanced RL': [],
    }
    
    print(f"\n{'='*60}")
    print(f"方法对比")
    print(f"测试图数量: {len(test_graphs)}")
    print(f"{'='*60}\n")
    
    for G in test_graphs:
        k = max(1, int(G.number_of_nodes() * 0.1))
        
        # CI算法
        ci_centers = ci_select_subset(G, list(G.nodes()), k, [], radius=2)
        results['CI'].append(reward_fn(G, ci_centers))
        
        # 原始BiT-HyRL
        try:
            bit_centers = hybrid_rl_select(G, k, model_path='src/train/v1/checkpoints/rl_agent.pth')
            results['BiT-HyRL (原始)'].append(reward_fn(G, bit_centers))
        except:
            results['BiT-HyRL (原始)'].append(0.0)
        
        # Small Network模型
        if small_model:
            trainer = SmallNetworkTrainer(model=small_model)
            small_centers = trainer.select_controllers(G, k, deterministic=True)
            results['Small Network'].append(reward_fn(G, small_centers))
        else:
            results['Small Network'].append(0.0)
        
        # Enhanced RL模型
        if enhanced_model:
            trainer = EnhancedRLTrainer()
            trainer.model = enhanced_model
            enhanced_centers = trainer.select_controllers(G, k, deterministic=True)
            results['Enhanced RL'].append(reward_fn(G, enhanced_centers))
        else:
            results['Enhanced RL'].append(0.0)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"对比结果")
    print(f"{'='*60}")
    for method, rewards in results.items():
        if rewards:
            print(f"  {method:20s}: 平均={np.mean(rewards):.4f}, 标准差={np.std(rewards):.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='BiT-HyRL 小规模网络优化训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='small',
                       choices=['small', 'enhanced', 'evaluate', 'compare'],
                       help='训练模式: small=小规模网络专用, enhanced=增强型RL, evaluate=评估, compare=对比')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default=None,
                       help='训练数据目录')
    parser.add_argument('--num-graphs', type=int, default=200,
                       help='生成的训练图数量')
    parser.add_argument('--max-graphs', type=int, default=None,
                       help='最大加载图数量')
    parser.add_argument('--min-size', type=int, default=10,
                       help='最小节点数')
    parser.add_argument('--max-size', type=int, default=100,
                       help='最大节点数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 输出参数
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='模型输出路径')
    parser.add_argument('--model', type=str, default=None,
                       help='评估时的模型路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 执行
    if args.mode == 'small':
        train_small_network_model(args)
    elif args.mode == 'enhanced':
        train_enhanced_rl_model(args)
    elif args.mode == 'evaluate':
        if not args.model:
            print("请指定模型路径: --model <path>")
            return
        evaluate_model(args)
    elif args.mode == 'compare':
        compare_methods(args)


if __name__ == '__main__':
    main()
