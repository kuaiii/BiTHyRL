# -*- coding: utf-8 -*-
"""
BiT-HyRL Agent 离线训练脚本（优化版）

支持：
1. Node2Vec特征（优化1）
2. Step-wise Reward（优化2）
3. 多种优化目标（combined, robustness, csa, entropy, wcp）

使用方法:
    python train_bit_hyrl_agent.py --use_node2vec --use_stepwise --epochs 100
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sys
import os

# 自动将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import networkx as nx
import numpy as np
import random
import torch
from torch.distributions import Categorical
from tqdm import tqdm
import time

# 导入必要的模块
from src.topology.generators import load_graph
from src.utils.logger import get_logger
from src.bit_hyrl import (
    get_node_features, 
    MLPPolicy, 
    calculate_stepwise_reward,
    calculate_reward,
    train_offline_optimized,
    config,
)

DEVICE = config.DEVICE

logger = get_logger(__name__)

def generate_synthetic_graphs(num_graphs=100, min_nodes=20, max_nodes=200):
    """
    生成合成网络用于训练
    
    Args:
        num_graphs: 生成的图数量
        min_nodes: 最小节点数
        max_nodes: 最大节点数
    
    Returns:
        list: NetworkX图列表
    """
    graphs = []
    print(f"Generating {num_graphs} synthetic graphs...")
    
    for i in tqdm(range(num_graphs), desc="Generating graphs"):
        n = random.randint(min_nodes, max_nodes)
        # 生成BA无标度网络
        m = random.randint(2, min(5, n-1))
        try:
            G = nx.barabasi_albert_graph(n, m, seed=i)
            graphs.append(G)
        except:
            continue
    
    return graphs

def load_real_datasets(datasets=None):
    """
    加载真实数据集
    
    Args:
        datasets: 数据集名称列表，如果为None则自动查找
    
    Returns:
        list: NetworkX图列表
    """
    if datasets is None:
        datasets = ['Chinanet', 'GtsCe', 'UsCarrier', 'Colt', 'Cogentco']
    
    graphs = []
    print(f"Loading real datasets: {datasets}")
    
    for dataset_name in datasets:
        # 尝试多个路径
        paths = [
            f'dataset/all/{dataset_name}.gml',
            f'dataset/testdata/{dataset_name}.gml'
        ]
        
        for gml_path in paths:
            if os.path.exists(gml_path):
                try:
                    G, _ = load_graph(gml_path)
                    if G and G.number_of_nodes() > 0:
                        graphs.append(G)
                        print(f"  Loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {gml_path}: {e}")
                    continue
    
    return graphs

def train_offline_optimized(graphs, epochs=100, save_path='src/train/v1/checkpoints/rl_agent.pth', 
                           mode='combined', use_node2vec=True, use_stepwise_reward=True):
    """
    离线训练RL Agent（优化版）
    
    Args:
        graphs: 训练图列表
        epochs: 训练轮数
        save_path: 模型保存路径
        mode: 优化目标类型
        use_node2vec: 是否使用Node2Vec特征
        use_stepwise_reward: 是否使用Step-wise Reward
    """
    if not graphs:
        logger.error("No graphs provided for training")
        return
    
    print(f"\n{'='*60}")
    print(f"Training BiT-HyRL Agent (Optimized)")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Node2Vec: {use_node2vec}")
    print(f"Step-wise Reward: {use_stepwise_reward}")
    print(f"Epochs: {epochs}")
    print(f"Training graphs: {len(graphs)}")
    print(f"{'='*60}\n")
    
    # 确定特征维度
    sample_G = graphs[0]
    sample_features, _ = get_node_features(sample_G, use_node2vec=use_node2vec)
    num_features = sample_features.shape[1]
    
    print(f"Feature dimension: {num_features}")
    
    # 创建模型
    model = MLPPolicy(num_features=num_features, hidden_dim=64, output_dim=1)
    model = model.to(DEVICE)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练历史
    history = []
    
    # 梯度累积步数
    accumulation_steps = 32
    
    # Baseline (Moving Average Reward) 用于降低方差
    moving_avg_reward = 0.0
    
    print(f"\nStarting training on {DEVICE}...")
    
    for epoch in range(epochs):
        model.train()
        train_reward = 0
        random.shuffle(graphs)
        
        optimizer.zero_grad()
        
        for i, G in enumerate(graphs):
            if G.number_of_nodes() < 2:
                continue
            
            # 获取特征
            x, node_list = get_node_features(G, use_node2vec=use_node2vec)
            x = x.to(DEVICE)
            
            # 控制器数量（根据网络大小动态确定）
            k = max(1, int(G.number_of_nodes() * 0.1))
            k = min(k, G.number_of_nodes() - 1)
            
            log_probs = []
            selected_indices = []
            selected_centers = []
            mask = torch.zeros(len(node_list), dtype=torch.bool, device=DEVICE)
            
            # 逐步选择控制器
            total_loss = None
            for step in range(k):
                probs = model(x, mask)
                m = Categorical(probs)
                action = m.sample()
                action_idx = action.item()
                
                selected_indices.append(action_idx)
                log_probs.append(m.log_prob(action))
                
                # 重要：创建新的mask而不是inplace修改，避免梯度计算错误
                mask = mask.clone()
                mask[action_idx] = True
                
                current_center = node_list[action_idx]
                selected_centers.append(current_center)
                
                # 计算奖励
                if use_stepwise_reward:
                    # Step-wise Reward
                    step_reward = calculate_stepwise_reward(
                        G,
                        selected_centers[:-1],  # 之前的控制器
                        current_center,  # 新选择的控制器
                        mode=mode
                    )
                    reward = step_reward
                    
                    # 更新移动平均
                    if moving_avg_reward == 0.0:
                        moving_avg_reward = reward
                    else:
                        moving_avg_reward = 0.95 * moving_avg_reward + 0.05 * reward
                    
                    # Advantage = Reward - Baseline
                    advantage = reward - moving_avg_reward
                    
                    # 计算损失并累积（不立即backward）
                    loss_step = -log_probs[-1] * advantage / accumulation_steps
                    train_reward += reward
                    
                    # 累积损失，稍后一起backward
                    if total_loss is None:
                        total_loss = loss_step
                    else:
                        total_loss = total_loss + loss_step
                else:
                    # 端到端奖励（在最后计算）
                    if step == k - 1:
                        centers = [node_list[idx] for idx in selected_indices]
                        reward = calculate_reward(G, centers, mode=mode, use_stepwise=False)
                        
                        # 更新移动平均
                        if moving_avg_reward == 0.0:
                            moving_avg_reward = reward
                        else:
                            moving_avg_reward = 0.95 * moving_avg_reward + 0.05 * reward
                        
                        # Advantage = Reward - Baseline
                        advantage = reward - moving_avg_reward
                        
                        # 计算总损失
                        total_loss = sum([-log_prob * advantage / accumulation_steps for log_prob in log_probs])
                        train_reward += reward
            
            # 在完成所有步骤后，一次性backward（避免inplace操作问题）
            if total_loss is not None:
                total_loss.backward()
            
            # 梯度累积：每accumulation_steps步更新一次
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # 处理剩余的梯度
        if len(graphs) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_reward = train_reward / len(graphs) if graphs else 0
        history.append([epoch+1, avg_train_reward])
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Avg Reward: {avg_train_reward:.4f}")
            print(f"Epoch {epoch+1}/{epochs} | Avg Reward: {avg_train_reward:.4f}")
    
    # 保存模型
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    print(f"\nModel saved to: {os.path.abspath(save_path)}")
    
    # 保存训练历史
    import csv
    history_path = save_path.replace('.pth', '_training_history.csv')
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg_Reward'])
        writer.writerows(history)
    logger.info(f"Training history saved to {history_path}")
    print(f"Training history saved to: {os.path.abspath(history_path)}")


def main():
    parser = argparse.ArgumentParser(
        description='Train BiT-HyRL Agent (Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用所有优化训练combined模型
  python train_bit_hyrl_agent.py --use_node2vec --use_stepwise --epochs 100
  
  # 训练特定目标模型
  python train_bit_hyrl_agent.py --mode robustness --use_node2vec --use_stepwise
  
  # 只使用真实数据集
  python train_bit_hyrl_agent.py --no_synthetic --use_node2vec
        """
    )
    
    parser.add_argument('--mode', type=str, default='combined',
                       choices=['combined', 'robustness', 'csa', 'entropy', 'wcp'],
                       help='优化目标类型 (默认: combined)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--use_node2vec', action='store_true', default=True,
                       help='使用Node2Vec特征（优化1） (默认: True)')
    parser.add_argument('--no_node2vec', dest='use_node2vec', action='store_false',
                       help='禁用Node2Vec特征')
    parser.add_argument('--use_stepwise', action='store_true', default=True,
                       help='使用Step-wise Reward（优化2） (默认: True)')
    parser.add_argument('--no_stepwise', dest='use_stepwise', action='store_false',
                       help='禁用Step-wise Reward')
    parser.add_argument('--num_synthetic', type=int, default=200,
                       help='生成的合成图数量 (默认: 200)')
    parser.add_argument('--no_synthetic', action='store_true',
                       help='不使用合成图，只使用真实数据集')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='指定要加载的数据集名称列表')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='模型保存目录 (默认: models)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BiT-HyRL Agent Training (Optimized)")
    print("=" * 60)
    
    # 准备训练数据
    all_graphs = []
    
    # 1. 加载真实数据集
    real_graphs = load_real_datasets(args.datasets)
    all_graphs.extend(real_graphs)
    print(f"Loaded {len(real_graphs)} real graphs")
    
    # 2. 生成合成图（如果启用）
    if not args.no_synthetic:
        synthetic_graphs = generate_synthetic_graphs(num_graphs=args.num_synthetic)
        all_graphs.extend(synthetic_graphs)
        print(f"Generated {len(synthetic_graphs)} synthetic graphs")
    
    if not all_graphs:
        print("Error: No training graphs available!")
        return
    
    print(f"\nTotal training graphs: {len(all_graphs)}")
    
    # 确定保存路径
    model_name = f'rl_agent_{args.mode}.pth' if args.mode != 'combined' else 'rl_agent.pth'
    save_path = os.path.join(args.output_dir, model_name)
    
    # 开始训练（使用 src.bit_hyrl 中的实现）
    start_time = time.time()
    train_offline_optimized(
        all_graphs,
        epochs=args.epochs,
        save_path=save_path,
        mode=args.mode,
        use_node2vec=args.use_node2vec,
        use_stepwise_reward=args.use_stepwise
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training completed in {elapsed_time/60:.2f} minutes")
    print(f"Model saved to: {os.path.abspath(save_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
