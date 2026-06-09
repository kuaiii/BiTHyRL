# -*- coding: utf-8 -*-
"""
BiT-HyRL 统一模型训练脚本

训练支持所有规模网络(20-1000节点)的统一GNN+PPO模型。

特点:
1. 规模自适应Node2Vec特征提取
2. 规模编码融合
3. 统一的3层GAT架构
4. GCC导向奖励函数
5. 课程学习和规模平衡采样

用法:
  # 标准训练
  python scripts/train_unified_model.py --epochs 200
  
  # 使用课程学习
  python scripts/train_unified_model.py --epochs 200 --curriculum
  
  # 重点关注小网络
  python scripts/train_unified_model.py --epochs 200 --focus_min 50 --focus_max 150
  
  # 增量训练
  python scripts/train_unified_model.py --epochs 100 --resume
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import random
import time
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_network_exact
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def load_training_data(
    train_dir=None,
    min_nodes=20,
    max_nodes=1000,
    max_graphs=None,
    use_bimodal=True,
    hub_ratio=0.15,
    verbose=True,
):
    """
    加载训练数据
    
    Args:
        train_dir: 训练数据目录
        min_nodes: 最小节点数
        max_nodes: 最大节点数
        max_graphs: 最大图数量
        use_bimodal: 是否使用双峰拓扑重构
        hub_ratio: hub节点比例
        verbose: 是否显示进度
        
    Returns:
        graphs: 图列表
    """
    if train_dir is None:
        train_dir = os.path.join(ROOT, 'dataset', 'all', 'syn')
    
    if not os.path.exists(train_dir):
        print(f"错误: 训练数据目录不存在: {train_dir}")
        return []
    
    # 收集所有 .gml 文件（优先使用 .gml，因为 load_graph 对 .gml 支持更好）
    all_files = []
    gml_basenames = set()
    
    # 首先收集所有 .gml 文件
    for f in os.listdir(train_dir):
        if f.endswith('.gml'):
            all_files.append(os.path.join(train_dir, f))
            gml_basenames.add(os.path.splitext(f)[0])
    
    # 然后收集没有对应 .gml 的 .graphml 文件
    for f in os.listdir(train_dir):
        if f.endswith('.graphml'):
            basename = os.path.splitext(f)[0]
            if basename not in gml_basenames:
                all_files.append(os.path.join(train_dir, f))
    
    if verbose:
        print(f"从 {train_dir} 发现 {len(all_files)} 个图文件")
    
    # 随机打乱并限制数量
    if max_graphs and max_graphs < len(all_files):
        random.shuffle(all_files)
        all_files = all_files[:max_graphs]
        if verbose:
            print(f"随机采样 {max_graphs} 个图用于训练")
    
    if verbose:
        print(f"节点数过滤: {min_nodes} - {max_nodes}")
    
    graphs = []
    skipped = 0
    
    for filepath in tqdm(all_files, desc='加载数据', disable=not verbose, ncols=80):
        try:
            # load_graph returns (G, pos) tuple
            result = load_graph(filepath, verbose=False)
            
            # Handle both tuple return and direct graph return
            if isinstance(result, tuple):
                G, _ = result
            else:
                G = result
            
            if G is None or G.number_of_nodes() == 0:
                skipped += 1
                continue
            
            # 节点数过滤
            n = G.number_of_nodes()
            if n < min_nodes or n > max_nodes:
                skipped += 1
                continue
            
            # 确保连通
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
            # 双峰拓扑重构
            if use_bimodal:
                try:
                    G = create_bimodal_network_exact(G, hub_ratio=hub_ratio)
                except Exception as e:
                    logger.debug(f"双峰重构失败: {e}")
            
            graphs.append(G)
            
        except Exception as e:
            logger.debug(f"加载失败 {filepath}: {e}")
            skipped += 1
    
    if verbose:
        print(f"成功加载 {len(graphs)} 个图, 跳过 {skipped} 个")
        
        # 分析规模分布 (支持 20-1000 全范围)
        if graphs:
            scale_dist = {
                'tiny (20-50)': 0, 'small (50-100)': 0, 'medium (100-200)': 0,
                'large (200-500)': 0, 'xlarge (500-1000)': 0
            }
            for g in graphs:
                n = g.number_of_nodes()
                if n < 50:
                    scale_dist['tiny (20-50)'] += 1
                elif n < 100:
                    scale_dist['small (50-100)'] += 1
                elif n < 200:
                    scale_dist['medium (100-200)'] += 1
                elif n < 500:
                    scale_dist['large (200-500)'] += 1
                else:
                    scale_dist['xlarge (500-1000)'] += 1
            print(f"规模分布: {scale_dist}")
    
    return graphs


def generate_synthetic_data(
    num_graphs=200,
    min_nodes=30,
    max_nodes=300,
    use_bimodal=True,
    hub_ratio=0.15,
    verbose=True,
):
    """
    生成合成训练数据
    
    生成多种类型的图以覆盖不同规模和结构。
    
    Args:
        num_graphs: 生成的图数量
        min_nodes: 最小节点数
        max_nodes: 最大节点数
        use_bimodal: 是否使用双峰拓扑
        hub_ratio: hub节点比例
        verbose: 是否显示进度
        
    Returns:
        graphs: 图列表
    """
    graphs = []
    
    # 定义规模分布（更多小图）
    scale_weights = {
        'tiny': (20, 50, 0.25),      # 25% 极小图
        'small': (50, 100, 0.30),    # 30% 小图
        'medium': (100, 200, 0.25),  # 25% 中图
        'large': (200, max_nodes, 0.20),  # 20% 大图
    }
    
    # 图类型
    graph_types = ['ba', 'ws', 'er']
    
    if verbose:
        print(f"生成 {num_graphs} 个合成图 (节点范围: {min_nodes}-{max_nodes})")
    
    for i in tqdm(range(num_graphs), desc='生成数据', disable=not verbose, ncols=80):
        # 按权重选择规模
        r = random.random()
        cumsum = 0
        for scale_name, (low, high, weight) in scale_weights.items():
            cumsum += weight
            if r < cumsum:
                n = random.randint(max(min_nodes, low), min(max_nodes, high))
                break
        else:
            n = random.randint(min_nodes, max_nodes)
        
        # 随机选择图类型
        graph_type = random.choice(graph_types)
        
        try:
            if graph_type == 'ba':
                # BA无标度网络
                m = random.randint(2, min(5, n // 2))
                G = nx.barabasi_albert_graph(n, m)
            elif graph_type == 'ws':
                # WS小世界网络
                k = random.randint(4, min(10, n // 2))
                p = random.uniform(0.1, 0.5)
                G = nx.watts_strogatz_graph(n, k, p)
            else:
                # ER随机网络
                p = random.uniform(0.05, 0.15)
                G = nx.erdos_renyi_graph(n, p)
                # 确保连通
                while not nx.is_connected(G):
                    p += 0.01
                    G = nx.erdos_renyi_graph(n, p)
            
            # 双峰拓扑重构
            if use_bimodal:
                try:
                    G = create_bimodal_network_exact(G, hub_ratio=hub_ratio)
                except:
                    pass
            
            graphs.append(G)
            
        except Exception as e:
            logger.debug(f"生成失败: {e}")
    
    if verbose:
        print(f"成功生成 {len(graphs)} 个图")
    
    return graphs


def evaluate_on_testdata(model, testdata_dir=None, k_ratio=0.1, verbose=True):
    """
    在测试集上评估模型
    
    Args:
        model: 训练好的模型
        testdata_dir: 测试数据目录
        k_ratio: 控制器比例
        verbose: 是否显示详情
        
    Returns:
        dict: 评估结果
    """
    from src.bit_hyrl.gnn_model import (
        get_unified_features, 
        graph_to_pyg_data,
    )
    from src.bit_hyrl.reward import calculate_unified_gcc_reward
    from src.bit_hyrl import config
    
    if testdata_dir is None:
        testdata_dir = os.path.join(ROOT, 'dataset', 'testdata')
    
    if not os.path.exists(testdata_dir):
        print(f"测试数据目录不存在: {testdata_dir}")
        return {}
    
    # 加载测试数据
    test_graphs = []
    for f in os.listdir(testdata_dir):
        if f.endswith('.gml') or f.endswith('.graphml'):
            filepath = os.path.join(testdata_dir, f)
            try:
                G = load_graph(filepath)
                if G and G.number_of_nodes() >= 10:
                    test_graphs.append((f, G))
            except:
                pass
    
    if not test_graphs:
        print("未找到有效的测试图")
        return {}
    
    if verbose:
        print(f"\n在 {len(test_graphs)} 个测试图上评估...")
    
    model.eval()
    device = config.DEVICE
    
    results = []
    scale_results = {'tiny': [], 'small': [], 'medium': [], 'large': []}
    
    with torch.no_grad():
        for name, G in tqdm(test_graphs, desc='评估', disable=not verbose, ncols=80):
            n = G.number_of_nodes()
            k = max(1, int(n * k_ratio))
            
            # 获取特征
            try:
                node_features, scale_encoding, node_list, raw_dim = get_unified_features(G, device=device)
                x = model.feature_projector(node_features, raw_dim)
                _, edge_index, _ = graph_to_pyg_data(G, device=device)
            except Exception as e:
                logger.debug(f"特征提取失败 {name}: {e}")
                continue
            
            # 选择控制器
            selected_mask = torch.zeros(len(node_list), dtype=torch.bool, device=device)
            centers = []
            
            for _ in range(k):
                action, _, _, _ = model.get_action(
                    x, edge_index, scale_encoding=scale_encoding,
                    selected_mask=selected_mask, deterministic=True
                )
                action_idx = action.item()
                centers.append(node_list[action_idx])
                selected_mask[action_idx] = True
            
            # 计算奖励
            reward = calculate_unified_gcc_reward(G, centers)
            
            results.append({
                'name': name,
                'nodes': n,
                'controllers': k,
                'reward': reward,
            })
            
            # 按规模分组
            if n < 50:
                scale_results['tiny'].append(reward)
            elif n < 100:
                scale_results['small'].append(reward)
            elif n < 200:
                scale_results['medium'].append(reward)
            else:
                scale_results['large'].append(reward)
    
    # 汇总结果
    avg_reward = np.mean([r['reward'] for r in results]) if results else 0
    
    if verbose:
        print(f"\n评估结果:")
        print(f"  整体平均奖励: {avg_reward:.4f}")
        for scale, rewards in scale_results.items():
            if rewards:
                print(f"  {scale}: {np.mean(rewards):.4f} (n={len(rewards)})")
    
    return {
        'avg_reward': avg_reward,
        'scale_results': {k: np.mean(v) if v else 0 for k, v in scale_results.items()},
        'details': results,
    }


def main():
    parser = argparse.ArgumentParser(description='BiT-HyRL 统一模型训练')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default=None, help='训练数据目录')
    parser.add_argument('--min_nodes', type=int, default=20, help='最小节点数')
    parser.add_argument('--max_nodes', type=int, default=1000, help='最大节点数')
    parser.add_argument('--max_graphs', type=int, default=None, help='最大训练图数量')
    parser.add_argument('--use_synthetic', action='store_true', help='使用合成数据')
    parser.add_argument('--synthetic_num', type=int, default=300, help='合成数据数量')
    
    # 模型参数
    parser.add_argument('--in_channels', type=int, default=64, help='投影后的特征维度')
    parser.add_argument('--hidden_channels', type=int, default=96, help='隐藏层维度')
    parser.add_argument('--scale_encoding_dim', type=int, default=8, help='规模编码维度')
    parser.add_argument('--heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=3, help='GAT层数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--k_ratio', type=float, default=0.1, help='控制器比例')
    parser.add_argument('--collect_per_epoch', type=int, default=60, help='每轮采样数')
    
    # 训练策略 (默认: 规模平衡采样，20-1000 全范围均等对待)
    parser.add_argument('--curriculum', action='store_true', help='使用课程学习(前期小图)')
    parser.add_argument('--scale_balance', action='store_true', default=True, help='规模平衡采样，确保各规模均等')
    parser.add_argument('--no_scale_balance', dest='scale_balance', action='store_false', help='禁用规模平衡')
    parser.add_argument('--focus_min', type=int, default=None, help='重点关注的最小节点数(不设则全范围均等)')
    parser.add_argument('--focus_max', type=int, default=None, help='重点关注的最大节点数')
    parser.add_argument('--focus_weight', type=float, default=2.0, help='重点范围权重')
    
    # 其他参数
    parser.add_argument('--save_path', type=str, default='src/train/v1/checkpoints/unified_gat_policy.pth', help='模型保存路径')
    parser.add_argument('--resume', action='store_true', help='增量训练')
    parser.add_argument('--evaluate', action='store_true', help='训练后评估')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--quiet', action='store_true', help='安静模式')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("BiT-HyRL 统一模型训练")
        print("=" * 60)
    
    # 加载或生成训练数据
    if args.use_synthetic:
        graphs = generate_synthetic_data(
            num_graphs=args.synthetic_num,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            verbose=verbose,
        )
    else:
        graphs = load_training_data(
            train_dir=args.train_dir,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            max_graphs=args.max_graphs,
            verbose=verbose,
        )
    
    if not graphs:
        print("错误: 没有可用的训练数据")
        return
    
    # 确定训练策略 (默认: 规模平衡，20-1000全范围均等)
    use_curriculum = args.curriculum
    use_scale_balance = args.scale_balance
    focus_range = None
    
    if args.focus_min is not None and args.focus_max is not None:
        focus_range = (args.focus_min, args.focus_max)
    
    # 默认: 规模平衡采样，无课程学习，无重点范围 → 全范围均等
    if not use_curriculum and not use_scale_balance and not focus_range:
        use_scale_balance = True
        if verbose:
            print("默认启用规模平衡采样 (20-1000 全范围均等)")
    
    # 导入训练函数
    from src.bit_hyrl.ppo_trainer import train_unified_gnn_ppo
    
    # 设置增量训练路径
    resume_from = None
    if args.resume and os.path.exists(args.save_path):
        resume_from = args.save_path
    
    if verbose:
        print(f"\n训练配置:")
        print(f"  数据量: {len(graphs)} 个图")
        print(f"  Epochs: {args.epochs}")
        print(f"  模型: UnifiedGATPolicy ({args.num_layers}层GAT, {args.heads}头, {args.hidden_channels}维)")
        print(f"  课程学习: {use_curriculum}")
        print(f"  规模平衡: {use_scale_balance}")
        if focus_range:
            print(f"  重点范围: {focus_range[0]}-{focus_range[1]} 节点")
        print(f"  保存路径: {args.save_path}")
        if resume_from:
            print(f"  增量训练: 从 {resume_from} 继续")
        print()
    
    # 训练
    start_time = time.time()
    
    result = train_unified_gnn_ppo(
        graphs=graphs,
        epochs=args.epochs,
        save_path=args.save_path,
        lr=args.lr,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        scale_encoding_dim=args.scale_encoding_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        k_ratio=args.k_ratio,
        collect_per_epoch=args.collect_per_epoch,
        use_curriculum=use_curriculum,
        use_scale_balance=use_scale_balance,
        focus_range=focus_range,
        focus_weight=args.focus_weight,
        verbose=verbose,
        resume_from=resume_from,
    )
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n训练完成!")
        print(f"  耗时: {elapsed / 60:.1f} 分钟")
        print(f"  最佳奖励: {result['best_reward']:.4f}")
        print(f"  最终奖励: {result['final_reward']:.4f}")
    
    # 评估
    if args.evaluate:
        from src.bit_hyrl.gnn_model import UnifiedGATPolicy
        from src.bit_hyrl import config
        
        # 加载训练好的模型
        model = UnifiedGATPolicy(
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            scale_encoding_dim=args.scale_encoding_dim,
            heads=args.heads,
            num_layers=args.num_layers,
        ).to(config.DEVICE)
        
        checkpoint = torch.load(args.save_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        eval_result = evaluate_on_testdata(model, k_ratio=args.k_ratio, verbose=verbose)
        
        if eval_result:
            print(f"\n最终评估:")
            print(f"  整体: {eval_result['avg_reward']:.4f}")
            for scale, score in eval_result['scale_results'].items():
                if score > 0:
                    print(f"  {scale}: {score:.4f}")
    
    # 保存训练曲线
    if result.get('history'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        history = result['history']
        epochs = [h[0] for h in history]
        rewards = [h[1] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, rewards, 'b-', alpha=0.5, label='Reward')
        
        # 添加平滑曲线
        if len(rewards) > 10:
            window = min(20, len(rewards) // 5)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(epochs[window-1:], smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Unified GAT Policy Training Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        curve_path = args.save_path.replace('.pth', '_training_curve.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"\n训练曲线已保存: {curve_path}")


if __name__ == '__main__':
    main()
