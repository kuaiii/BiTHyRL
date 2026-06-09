# -*- coding: utf-8 -*-
"""
BiT-HyRL 主函数入口

整合训练和运行功能，提供统一的命令行接口。

功能模块：
1. 模型训练：离线批量训练 BiT-HyRL Agent
2. 网络重构：使用训练好的模型进行双峰拓扑重构和控制器选择
3. 攻击仿真：可选执行攻击仿真并生成可视化结果

函数调用关系：
main()
├── train_model() 
│   ├── load_real_datasets() / generate_synthetic_graphs()
│   └── train_offline_optimized() [src.bit_hyrl.training]
│       ├── get_node_features() [src.bit_hyrl.features]
│       ├── MLPPolicy() [src.bit_hyrl.model]
│       └── calculate_stepwise_reward() / calculate_reward() [src.bit_hyrl.reward]
│
└── run_reconstruction()
    ├── load_graph() [src.topology.generators]
    └── bit_hyrl_reconstruct() [src.bit_hyrl.reconstruct]
        ├── create_bimodal_network_exact() [src.bit_hyrl.topology]
        └── train_and_select() [src.bit_hyrl.training]
            ├── hybrid_rl_select() [src.bit_hyrl.selection]
            │   ├── predict() [src.bit_hyrl.selection]
            │   │   ├── get_node_features() [src.bit_hyrl.features]
            │   │   └── load_or_train_model() [src.bit_hyrl.selection]
            │   └── k_median_vectorized_subset() / ci_select_subset() [src.bit_hyrl.selection]
            └── train_offline_single() [src.bit_hyrl.training]
                ├── get_node_features() [src.bit_hyrl.features]
                └── calculate_stepwise_reward() / calculate_reward() [src.bit_hyrl.reward]
        └── simulate_attack_and_plot() [src.bit_hyrl.attack_plot] (可选)

使用方法:
    # 训练模型
    python main_bit_hyrl.py train --mode combined --epochs 100 --use_node2vec --use_stepwise
    
    # 运行重构
    python main_bit_hyrl.py run --dataset GtsCe --rate 0.1 --episodes 100
    
    # 训练并运行（完整流程）
    python main_bit_hyrl.py full --dataset GtsCe --rate 0.1 --train_epochs 100 --run_episodes 100
"""

import sys
import os

# 确保项目根目录在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import pickle
import time
import random
import networkx as nx
from tqdm import tqdm

# 导入项目模块
from src.topology.generators import load_graph, construct_ba
from src.utils.logger import get_logger
from src.bit_hyrl import (
    config,
    train_offline_optimized,
    bit_hyrl_reconstruct,
    simulate_attack_and_plot,
)

logger = get_logger(__name__)


# ==================== 数据加载函数 ====================

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
    logger.info(f"Loading real datasets: {datasets}")
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
                        print(f"  ✓ Loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                        logger.info(f"Loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {gml_path}: {e}")
                    continue
    
    return graphs


def generate_synthetic_graphs(num_graphs=200, min_nodes=20, max_nodes=200):
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
    logger.info(f"Generating {num_graphs} synthetic graphs")
    
    for i in tqdm(range(num_graphs), desc="Generating graphs"):
        n = random.randint(min_nodes, max_nodes)
        # 生成BA无标度网络
        m = random.randint(2, min(5, n-1))
        try:
            G = nx.barabasi_albert_graph(n, m, seed=i)
            graphs.append(G)
        except Exception as e:
            logger.warning(f"Failed to generate graph {i}: {e}")
            continue
    
    return graphs


# ==================== 训练功能 ====================

def train_model(args):
    """
    训练 BiT-HyRL Agent 模型
    
    Args:
        args: 命令行参数对象
    """
    print("=" * 60)
    print("BiT-HyRL Agent 训练")
    print("=" * 60)
    
    # 准备训练数据
    all_graphs = []
    
    # 1. 加载真实数据集
    if args.datasets:
        real_graphs = load_real_datasets(args.datasets)
        all_graphs.extend(real_graphs)
        print(f"Loaded {len(real_graphs)} real graphs")
    
    # 2. 生成合成图（如果启用）
    if not args.no_synthetic:
        synthetic_graphs = generate_synthetic_graphs(
            num_graphs=args.num_synthetic,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes
        )
        all_graphs.extend(synthetic_graphs)
        print(f"Generated {len(synthetic_graphs)} synthetic graphs")
    
    if not all_graphs:
        print("Error: No training graphs available!")
        logger.error("No training graphs available")
        return False
    
    print(f"\nTotal training graphs: {len(all_graphs)}")
    
    # 确定保存路径
    model_name = f'rl_agent_{args.mode}.pth' if args.mode != 'combined' else 'rl_agent.pth'
    save_path = os.path.join(args.output_dir, model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印训练配置
    print(f"\n训练配置:")
    print(f"  Mode: {args.mode}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Node2Vec: {args.use_node2vec}")
    print(f"  Step-wise Reward: {args.use_stepwise}")
    print(f"  Model save path: {save_path}")
    print("=" * 60)
    
    # 开始训练
    start_time = time.time()
    try:
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
        print(f"训练完成! 耗时: {elapsed_time/60:.2f} 分钟")
        print(f"模型已保存: {os.path.abspath(save_path)}")
        print(f"{'='*60}")
        logger.info(f"Training completed in {elapsed_time/60:.2f} minutes")
        return True
    except Exception as e:
        print(f"\n训练出错: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


# ==================== 运行重构功能 ====================

def load_graph_from_args(args):
    """
    根据参数加载图
    
    Args:
        args: 命令行参数对象
    
    Returns:
        tuple: (Graph, dataset_name)
    """
    if args.synthetic:
        print(f"生成合成BA网络: {args.nodes} 节点")
        G = construct_ba(args.nodes, args.nodes * 2)
        return G, None
    
    if args.dataset:
        print(f"从数据集加载网络: {args.dataset}")
        for base in ('dataset/testdata', 'dataset/all'):
            p = os.path.join(current_dir, base, f'{args.dataset}.gml')
            if os.path.exists(p):
                G, _ = load_graph(p)
                return G, args.dataset
        raise FileNotFoundError(f"Dataset {args.dataset} not found")
    
    if args.input_graph:
        print(f"从文件加载网络: {args.input_graph}")
        r = load_graph(args.input_graph)
        G = r[0] if isinstance(r, tuple) else r
        dataset_name = os.path.splitext(os.path.basename(args.input_graph))[0]
        return G, dataset_name
    
    raise ValueError("Need --synthetic, --dataset, or --input_graph")


def run_reconstruction(args):
    """
    运行 BiT-HyRL 重构
    
    Args:
        args: 命令行参数对象
    """
    print("=" * 60)
    print("BiT-HyRL 算法重构")
    print("=" * 60)
    
    try:
        # 加载图
        G, dataset_name = load_graph_from_args(args)
        print(f"成功加载网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        # 打印配置
        print(f"\n重构配置:")
        print(f"  Controller rate: {args.rate}")
        print(f"  Hub ratio: {args.hub_ratio}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Metric type: {args.metric_type}")
        print(f"  Node2Vec: {args.use_node2vec}")
        print(f"  Step-wise Reward: {args.use_stepwise}")
        print(f"  CI algorithm: {args.use_ci}")
        if args.model_path:
            print(f"  Model path: {args.model_path}")
        print("=" * 60)
        
        # 执行重构
        start_time = time.time()
        G_bimodal, controllers = bit_hyrl_reconstruct(
            G=G,
            controller_rate=args.rate,
            hub_ratio=args.hub_ratio,
            episodes=args.episodes,
            metric_type=args.metric_type,
            seed=args.seed,
            model_path=args.model_path,
            use_node2vec=args.use_node2vec,
            use_stepwise_reward=args.use_stepwise,
            use_ci=args.use_ci,
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n重构完成!")
        print(f"  原始网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        print(f"  重构后网络: {G_bimodal.number_of_nodes()} 节点, {G_bimodal.number_of_edges()} 边")
        print(f"  控制器数量: {len(controllers)}")
        print(f"  控制器位置: {sorted(controllers)}")
        print(f"  耗时: {elapsed_time:.2f} 秒")
        
        # 保存结果
        out_dir = args.output_dir or config.RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        if args.synthetic:
            base = f"bit_hyrl_synthetic_{args.nodes}nodes_{ts}"
        elif args.dataset:
            base = f"bit_hyrl_{args.dataset}_{ts}"
        else:
            base = f"bit_hyrl_{dataset_name or 'run'}_{ts}"
        
        # 保存Pickle文件
        pkl_path = args.output or os.path.join(out_dir, f"{base}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'original_graph': G,
                'bimodal_graph': G_bimodal,
                'controllers': controllers,
                'parameters': {
                    'rate': args.rate,
                    'hub_ratio': args.hub_ratio,
                    'episodes': args.episodes,
                    'metric_type': args.metric_type,
                    'seed': args.seed,
                },
            }, f)
        print(f"\n结果已保存:")
        print(f"  Pickle文件: {os.path.abspath(pkl_path)}")
        
        # 保存GML文件
        gml_path = pkl_path.replace('.pkl', '.gml')
        nx.write_gml(G_bimodal, gml_path)
        print(f"  GML文件: {os.path.abspath(gml_path)}")
        
        # 保存文本文件
        txt_path = pkl_path.replace('.pkl', '_controllers.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"BiT-HyRL 控制器选择结果\n{'='*60}\n")
            f.write(f"原始网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边\n")
            f.write(f"重构后网络: {G_bimodal.number_of_nodes()} 节点, {G_bimodal.number_of_edges()} 边\n")
            f.write(f"控制器数量: {len(controllers)}\n")
            f.write(f"控制器位置: {sorted(controllers)}\n\n参数:\n")
            f.write(f"  controller_rate: {args.rate}\n")
            f.write(f"  hub_ratio: {args.hub_ratio}\n")
            f.write(f"  episodes: {args.episodes}\n")
            f.write(f"  metric_type: {args.metric_type}\n")
            f.write(f"  seed: {args.seed}\n")
        print(f"  文本文件: {os.path.abspath(txt_path)}")
        
        # 执行攻击仿真（如果启用）
        if args.run_attack:
            print("\n" + "=" * 60)
            print("开始攻击仿真和结果可视化")
            print("=" * 60)
            attack_dir = os.path.join(out_dir, f"attack_{args.attack_mode}")
            simulate_attack_and_plot(G_bimodal, controllers, args.attack_mode, attack_dir)
            print(f"\n攻击仿真结果已保存到: {os.path.abspath(attack_dir)}")
        
        print("\n" + "=" * 60)
        print("程序执行完成!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n错误: {e}")
        logger.error(f"Reconstruction failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return False


# ==================== 完整流程 ====================

def run_full_pipeline(args):
    """
    运行完整流程：训练 + 重构
    
    Args:
        args: 命令行参数对象
    """
    print("=" * 60)
    print("BiT-HyRL 完整流程")
    print("=" * 60)
    
    # 步骤1: 训练模型
    print("\n[步骤 1/2] 训练模型")
    print("-" * 60)
    train_success = train_model(args)
    
    if not train_success:
        print("训练失败，终止流程")
        return False
    
    # 步骤2: 运行重构
    print("\n[步骤 2/2] 运行重构")
    print("-" * 60)
    recon_success = run_reconstruction(args)
    
    if recon_success:
        print("\n" + "=" * 60)
        print("完整流程执行成功!")
        print("=" * 60)
        return True
    else:
        print("\n重构失败")
        return False


# ==================== 主函数 ====================

def main():
    """主函数：解析命令行参数并执行相应功能"""
    parser = argparse.ArgumentParser(
        description='BiT-HyRL 主函数入口 - 整合训练和运行功能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练模型
  python main_bit_hyrl.py train --mode combined --epochs 100
  
  # 运行重构
  python main_bit_hyrl.py run --dataset GtsCe --rate 0.1
  
  # 完整流程（训练+重构）
  python main_bit_hyrl.py full --dataset GtsCe --rate 0.1 --train_epochs 100
  
  # 使用预训练模型运行重构并执行攻击仿真
  python main_bit_hyrl.py run --dataset GtsCe --rate 0.1 --run_attack --attack_mode degree
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令: train, run, 或 full')
    
    # ========== 训练命令 ==========
    train_parser = subparsers.add_parser('train', help='训练 BiT-HyRL Agent 模型')
    train_parser.add_argument('--mode', type=str, default='combined',
                             choices=['combined', 'robustness', 'csa', 'entropy', 'wcp'],
                             help='优化目标类型 (默认: combined)')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数 (默认: 100)')
    train_parser.add_argument('--use_node2vec', action='store_true', default=True,
                             help='使用Node2Vec特征 (默认: True)')
    train_parser.add_argument('--no_node2vec', dest='use_node2vec', action='store_false')
    train_parser.add_argument('--use_stepwise', action='store_true', default=True,
                             help='使用Step-wise Reward (默认: True)')
    train_parser.add_argument('--no_stepwise', dest='use_stepwise', action='store_false')
    train_parser.add_argument('--num_synthetic', type=int, default=200,
                             help='生成的合成图数量 (默认: 200)')
    train_parser.add_argument('--no_synthetic', action='store_true',
                             help='不使用合成图，只使用真实数据集')
    train_parser.add_argument('--min_nodes', type=int, default=20,
                             help='合成图最小节点数 (默认: 20)')
    train_parser.add_argument('--max_nodes', type=int, default=200,
                             help='合成图最大节点数 (默认: 200)')
    train_parser.add_argument('--datasets', type=str, nargs='+', default=None,
                             help='指定要加载的数据集名称列表')
    train_parser.add_argument('--output_dir', type=str, default='models',
                             help='模型保存目录 (默认: models)')
    
    # ========== 运行命令 ==========
    run_parser = subparsers.add_parser('run', help='运行 BiT-HyRL 重构')
    input_group = run_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_graph', type=str, help='输入图文件路径 (GML格式)')
    input_group.add_argument('--dataset', type=str, help='数据集名称')
    input_group.add_argument('--synthetic', action='store_true', help='使用合成BA网络')
    run_parser.add_argument('--nodes', type=int, default=100, help='合成网络的节点数')
    run_parser.add_argument('--rate', type=float, default=0.1, help='控制器部署比例 (默认: 0.1)')
    run_parser.add_argument('--hub_ratio', type=float, default=0.15, help='Hub节点比例 (默认: 0.15)')
    run_parser.add_argument('--episodes', type=int, default=100, help='RL训练轮数 (默认: 100)')
    run_parser.add_argument('--metric_type', type=str, default='robustness',
                           choices=['combined', 'robustness', 'csa', 'entropy', 'wcp'],
                           help='优化目标类型 (默认: robustness)')
    run_parser.add_argument('--seed', type=int, default=None, help='随机种子')
    run_parser.add_argument('--model_path', type=str, default=None,
                           help='预训练模型路径 (默认: 使用models目录下的对应模型)')
    run_parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    run_parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    run_parser.add_argument('--attack_mode', type=str, default='degree',
                           choices=['degree', 'random', 'target', 'pagerank', 'betweenness', 'eigenvector', 'hybrid'],
                           help='攻击模式 (默认: degree)')
    run_parser.add_argument('--run_attack', action='store_true', help='是否执行攻击仿真并绘制图表')
    run_parser.add_argument('--use_node2vec', action='store_true', default=True, help='使用Node2Vec特征')
    run_parser.add_argument('--no_node2vec', dest='use_node2vec', action='store_false')
    run_parser.add_argument('--use_stepwise', action='store_true', default=True, help='使用Step-wise Reward')
    run_parser.add_argument('--no_stepwise', dest='use_stepwise', action='store_false')
    run_parser.add_argument('--use_ci', action='store_true', default=True, help='使用CI算法')
    run_parser.add_argument('--no_ci', dest='use_ci', action='store_false')
    
    # ========== 完整流程命令 ==========
    full_parser = subparsers.add_parser('full', help='完整流程：训练 + 重构')
    # 训练参数
    full_parser.add_argument('--mode', type=str, default='combined',
                           choices=['combined', 'robustness', 'csa', 'entropy', 'wcp'],
                           help='优化目标类型 (默认: combined)')
    full_parser.add_argument('--train_epochs', type=int, default=100, help='训练轮数 (默认: 100)')
    full_parser.add_argument('--use_node2vec', action='store_true', default=True,
                           help='使用Node2Vec特征 (默认: True)')
    full_parser.add_argument('--no_node2vec', dest='use_node2vec', action='store_false')
    full_parser.add_argument('--use_stepwise', action='store_true', default=True,
                           help='使用Step-wise Reward (默认: True)')
    full_parser.add_argument('--no_stepwise', dest='use_stepwise', action='store_false')
    full_parser.add_argument('--num_synthetic', type=int, default=200,
                           help='生成的合成图数量 (默认: 200)')
    full_parser.add_argument('--no_synthetic', action='store_true',
                           help='不使用合成图，只使用真实数据集')
    full_parser.add_argument('--datasets', type=str, nargs='+', default=None,
                           help='指定要加载的数据集名称列表（用于训练）')
    full_parser.add_argument('--train_output_dir', type=str, default='models',
                           help='模型保存目录 (默认: models)')
    # 运行参数
    input_group_full = full_parser.add_mutually_exclusive_group(required=True)
    input_group_full.add_argument('--input_graph', type=str, help='输入图文件路径 (GML格式)')
    input_group_full.add_argument('--dataset', type=str, help='数据集名称')
    input_group_full.add_argument('--synthetic', action='store_true', help='使用合成BA网络')
    full_parser.add_argument('--nodes', type=int, default=100, help='合成网络的节点数')
    full_parser.add_argument('--rate', type=float, default=0.1, help='控制器部署比例 (默认: 0.1)')
    full_parser.add_argument('--hub_ratio', type=float, default=0.15, help='Hub节点比例 (默认: 0.15)')
    full_parser.add_argument('--run_episodes', type=int, default=100, help='RL训练轮数 (默认: 100)')
    full_parser.add_argument('--seed', type=int, default=None, help='随机种子')
    full_parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    full_parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    full_parser.add_argument('--attack_mode', type=str, default='degree',
                           choices=['degree', 'random', 'target', 'pagerank', 'betweenness', 'eigenvector', 'hybrid'],
                           help='攻击模式 (默认: degree)')
    full_parser.add_argument('--run_attack', action='store_true', help='是否执行攻击仿真并绘制图表')
    full_parser.add_argument('--use_ci', action='store_true', default=True, help='使用CI算法')
    full_parser.add_argument('--no_ci', dest='use_ci', action='store_false')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 处理完整流程的特殊参数映射
    if args.command == 'full':
        # 将 train_epochs 映射到 epochs
        args.epochs = args.train_epochs
        args.output_dir = args.train_output_dir
        args.episodes = args.run_episodes
        args.metric_type = args.mode  # 使用训练时的mode作为metric_type
    
    # 执行相应命令
    if args.command == 'train':
        success = train_model(args)
        return 0 if success else 1
    elif args.command == 'run':
        success = run_reconstruction(args)
        return 0 if success else 1
    elif args.command == 'full':
        success = run_full_pipeline(args)
        return 0 if success else 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
