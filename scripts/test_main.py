# -*- coding: utf-8 -*-
"""
BiT-HyRL 测试入口脚本

功能:
1. 测试模式（默认）: 加载已训练的模型进行推理，不进行训练
2. 训练模式: 在线训练/微调模型

使用方法:
  # 测试模式（推荐）- 加载已有模型，不训练
  python test_main.py --dataset GtsCe --attack degree
  
  # 训练模式 - 会进行在线训练/微调
  python test_main.py --dataset GtsCe --attack degree --train_mode
  
  # 指定更多参数
  python test_main.py --dataset BA-200 --attack random --batch_size 3 --rate 0.1

训练流程:
  1. 首先运行离线训练脚本生成模型:
     python scripts/train_bit_hyrl.py --epochs 100 --mode robustness
     
  2. 然后使用测试脚本验证效果:
     python test_main.py --dataset GtsCe --attack degree
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import sys
import os

# 自动将项目根目录添加到 sys.path，防止直接运行时找不到模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from main import main


def check_model_exists():
    """检查模型文件是否存在"""
    model_dir = os.path.join(current_dir, 'models')
    models = ['rl_agent.pth', 'rl_agent_robustness.pth']
    existing = []
    for m in models:
        path = os.path.join(model_dir, m)
        if os.path.exists(path):
            existing.append(m)
    return existing


def test_main():
    """
    测试入口函数，支持测试模式和训练模式。
    
    测试模式（默认）: 仅加载已训练模型进行推理，不进行任何训练
    训练模式: 进行在线训练/微调
    """
    parser = argparse.ArgumentParser(
        description='BiT-HyRL 测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试模式（默认）
  python test_main.py --dataset GtsCe --attack degree
  
  # 训练模式
  python test_main.py --dataset GtsCe --attack degree --train_mode
  
  # 查看可用模型
  python test_main.py --check_models
        """
    )
    parser.add_argument('--dataset', type=str, default='BA-200', help='数据集名称')
    parser.add_argument('--attack', type=str, default='degree', 
                        choices=['hybrid', 'random', 'target', 'degree', 'pagerank', 
                                 'betweenness', 'eigenvector', 'bruteforce', 'wgcc'],
                        help='攻击模式。wgcc=同时跑 degree 与 random，对 GCC/CSA/CCE/WCP 求加权，结果存 results/{dataset}/wgcc/')
    parser.add_argument('--metric', type=str, default='gcc', help='评估指标')
    parser.add_argument('--batch_size', type=int, default=10, help='仿真批次数')
    parser.add_argument('--rate', type=float, default=0.1, help='控制器部署比例')
    parser.add_argument('--train_mode', action='store_true', 
                        help='训练模式：进行在线训练/微调。默认为测试模式（仅推理）')
    parser.add_argument('--check_models', action='store_true', help='检查可用模型并退出')
    parser.add_argument('--debug', action='store_true', help='启用调试日志')
    
    args = parser.parse_args()
    
    # 检查模型
    if args.check_models:
        print("=" * 50)
        print("检查 BiT-HyRL 模型文件")
        print("=" * 50)
        existing = check_model_exists()
        if existing:
            print(f"找到以下模型文件:")
            for m in existing:
                print(f"  - src/train/v1/checkpoints/{m}")
            print("\n可以使用测试模式运行实验。")
        else:
            print("未找到模型文件！")
            print("\n请先运行训练脚本:")
            print("  python scripts/train_bit_hyrl.py --epochs 100")
        return
    
    # 构建命令行参数
    mode_str = "训练模式" if args.train_mode else "测试模式（仅推理）"
    print("=" * 60)
    print(f"BiT-HyRL 实验 - {mode_str}")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"攻击模式: {args.attack}" + (" (degree+random 加权 GCC/CSA/CCE/WCP → results/{}/wgcc/)".format(args.dataset) if args.attack == 'wgcc' else ""))
    print(f"批次数: {args.batch_size}")
    print(f"控制器比例: {args.rate}")
    
    # 检查测试模式下是否有模型
    if not args.train_mode:
        existing = check_model_exists()
        if existing:
            print(f"使用模型: {existing[0]}")
        else:
            print("\n警告: 未找到预训练模型，将降级使用CI算法")
            print("建议先运行: python scripts/train_bit_hyrl.py")
    print("=" * 60)
    
    # 构建 main.py 的参数
    main_params = [
        'main.py',
        '--dataset', args.dataset,
        '--attack', args.attack,
        '--metric', args.metric,
        '--batch_size', str(args.batch_size),
        '--rate', str(args.rate),
    ]
    
    # 测试模式：添加 --test_mode 参数
    if not args.train_mode:
        main_params.append('--test_mode')
    
    if args.debug:
        main_params.append('--debug')
    
    # 备份并替换 sys.argv
    original_argv = sys.argv
    
    try:
        sys.argv = main_params
        main()
        
    except SystemExit as e:
        if e.code != 0:
            print(f"程序退出，代码: {e.code}")
    except Exception as e:
        print(f"运行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv
    
    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)


if __name__ == "__main__":
    test_main()
