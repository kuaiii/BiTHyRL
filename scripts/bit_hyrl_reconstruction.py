# -*- coding: utf-8 -*-
"""
BiT-HyRL 重构入口（向后兼容包装）。

本文件作为向后兼容的入口，实际实现位于 src.bit_hyrl 包中。
推荐使用: python scripts/run_bit_hyrl.py
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import pickle
import time
import networkx as nx

from src.topology.generators import load_graph
from src.bit_hyrl import (
    bit_hyrl_reconstruct,
    simulate_attack_and_plot,
    config,
)

logger = config.logger if hasattr(config, 'logger') else None
if logger is None:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)


def main():
    """命令行入口（向后兼容）。"""
    parser = argparse.ArgumentParser(
        description='BiT-HyRL (Bi-modal Topology Hybrid Reinforcement Learning) 算法重构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从数据集名称加载网络并重构
  python bit_hyrl_reconstruction.py --dataset GtsCe --rate 0.1
  
  # 从GML文件加载网络并重构
  python bit_hyrl_reconstruction.py --input_graph data/GtsCe.gml --rate 0.1
  
  # 使用合成BA网络
  python bit_hyrl_reconstruction.py --synthetic --nodes 100 --rate 0.1 --episodes 100
  
  # 指定模型路径和优化目标
  python bit_hyrl_reconstruction.py --dataset GtsCe --rate 0.1 \\
      --model_path src/train/v1/checkpoints/rl_agent_robustness.pth --metric_type robustness
  
  # 执行攻击仿真并绘制四个指标的折线图
  python bit_hyrl_reconstruction.py --dataset GtsCe --rate 0.1 --run_attack --attack_mode degree
  python bit_hyrl_reconstruction.py --synthetic --nodes 100 --rate 0.1 --run_attack --attack_mode degree
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_graph', type=str, help='输入图文件路径 (GML格式)')
    input_group.add_argument('--dataset', type=str, help='数据集名称')
    input_group.add_argument('--synthetic', action='store_true', help='使用合成BA网络')
    parser.add_argument('--nodes', type=int, default=100, help='合成网络的节点数')
    parser.add_argument('--rate', type=float, default=0.1, help='控制器部署比例')
    parser.add_argument('--hub_ratio', type=float, default=0.15, help='Hub节点比例')
    parser.add_argument('--episodes', type=int, default=100, help='RL训练轮数')
    parser.add_argument('--metric_type', type=str, default='robustness',
                       choices=['combined', 'robustness', 'csa', 'entropy', 'wcp'],
                       help='优化目标类型')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--model_path', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--output_dir', type=str, default='results/bit_hyrl', help='输出目录')
    parser.add_argument('--attack_mode', type=str, default='degree',
                       choices=['degree', 'random', 'target', 'pagerank', 'betweenness', 'eigenvector', 'hybrid'],
                       help='攻击模式')
    parser.add_argument('--run_attack', action='store_true', help='是否执行攻击仿真并绘制图表')
    parser.add_argument('--use_node2vec', action='store_true', default=True, help='使用Node2Vec特征')
    parser.add_argument('--no_node2vec', dest='use_node2vec', action='store_false')
    parser.add_argument('--use_stepwise', action='store_true', default=True, help='使用Step-wise Reward')
    parser.add_argument('--no_stepwise', dest='use_stepwise', action='store_false')
    parser.add_argument('--use_ci', action='store_true', default=True, help='使用CI算法')
    parser.add_argument('--no_ci', dest='use_ci', action='store_false')

    args = parser.parse_args()

    try:
        print("=" * 60)
        print("BiT-HyRL 算法重构")
        print("=" * 60)

        if args.synthetic:
            from src.topology.generators import construct_ba
            print(f"生成合成BA网络: {args.nodes} 节点")
            G = construct_ba(args.nodes, args.nodes * 2)
            dataset_name = None
        elif args.dataset:
            print(f"从数据集加载网络: {args.dataset}")
            for base in ('dataset/testdata', 'dataset/all'):
                p = os.path.join(current_dir, base, f'{args.dataset}.gml')
                if os.path.exists(p):
                    G, _ = load_graph(p)
                    dataset_name = args.dataset
                    break
            else:
                raise FileNotFoundError(f"Dataset {args.dataset} not found")
        else:
            print(f"从文件加载网络: {args.input_graph}")
            r = load_graph(args.input_graph)
            G = r[0] if isinstance(r, tuple) else r
            dataset_name = os.path.splitext(os.path.basename(args.input_graph))[0]

        print(f"成功加载网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        print(f"优化设置: Node2Vec={args.use_node2vec}, Step-wise Reward={args.use_stepwise}, CI算法={args.use_ci}")

        G_bimodal, controllers = bit_hyrl_reconstruct(
            G=G, controller_rate=args.rate, hub_ratio=args.hub_ratio,
            episodes=args.episodes, metric_type=args.metric_type, seed=args.seed,
            model_path=args.model_path, use_node2vec=args.use_node2vec,
            use_stepwise_reward=args.use_stepwise, use_ci=args.use_ci,
        )

        print(f"\n重构完成!")
        print(f"  原始网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        print(f"  重构后网络: {G_bimodal.number_of_nodes()} 节点, {G_bimodal.number_of_edges()} 边")
        print(f"  控制器数量: {len(controllers)}")
        print(f"  控制器位置: {sorted(controllers)}")

        if args.output is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            if args.synthetic:
                base = f"bit_hyrl_synthetic_{args.nodes}nodes_{ts}"
            elif args.dataset:
                base = f"bit_hyrl_{args.dataset}_{ts}"
            else:
                base = f"bit_hyrl_{dataset_name}_{ts}" if dataset_name else f"bit_hyrl_{ts}"
            out_dir = args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            args.output = os.path.join(out_dir, f"{base}.pkl")

        out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
        os.makedirs(out_dir, exist_ok=True)

        with open(args.output, 'wb') as f:
            pickle.dump({
                'original_graph': G, 'bimodal_graph': G_bimodal, 'controllers': controllers,
                'parameters': {'rate': args.rate, 'hub_ratio': args.hub_ratio,
                              'episodes': args.episodes, 'metric_type': args.metric_type, 'seed': args.seed},
            }, f)

        print(f"\n结果已保存:")
        print(f"  Pickle文件: {os.path.abspath(args.output)}")
        gml_path = args.output.replace('.pkl', '.gml') if args.output.endswith('.pkl') else args.output + '.gml'
        nx.write_gml(G_bimodal, gml_path)
        print(f"  GML文件: {os.path.abspath(gml_path)}")

        txt_path = args.output.replace('.pkl', '_controllers.txt') if args.output.endswith('.pkl') else args.output + '_controllers.txt'
        with open(txt_path, 'w') as f:
            f.write(f"BiT-HyRL 控制器选择结果\n{'='*60}\n")
            f.write(f"原始网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边\n")
            f.write(f"重构后网络: {G_bimodal.number_of_nodes()} 节点, {G_bimodal.number_of_edges()} 边\n")
            f.write(f"控制器数量: {len(controllers)}\n")
            f.write(f"控制器位置: {sorted(controllers)}\n\n参数:\n")
            f.write(f"  controller_rate: {args.rate}\n  hub_ratio: {args.hub_ratio}\n")
            f.write(f"  episodes: {args.episodes}\n  metric_type: {args.metric_type}\n  seed: {args.seed}\n")
        print(f"  文本文件: {os.path.abspath(txt_path)}")

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

    except Exception as e:
        print(f"\n错误: {e}")
        logger.error(f"程序执行出错: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)
