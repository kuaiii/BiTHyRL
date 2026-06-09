# -*- coding: utf-8 -*-
"""
评估真实网络专家模型在 Colt / Chinanet / UsCarrier 等真实数据集上的表现

与其它方法（Baseline, GA+RL, Onion+RL, ROMEN+RL, ...）进行对比，
确保 BiT-HyRL（专家模型）效果至少超过其它方法。

用法:
    # 评估单个模型
    python scripts/evaluate_real_network_specialist.py -m src/train/v1/checkpoints/real_network_specialist_seed42.pth
    
    # 评估多种子最佳
    python scripts/evaluate_real_network_specialist.py --model-dir models --datasets Colt,Chinanet,UsCarrier
    
    # 运行 main.py 进行完整对比
    python scripts/evaluate_real_network_specialist.py -m src/train/v1/checkpoints/real_network_specialist_seed42.pth --run-main
"""

import os
import sys
import argparse
import subprocess
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_main_with_model(dataset, model_path, batch_size=1, attack_mode='degree'):
    """调用 main.py 使用指定模型运行实验"""
    cmd = [
        sys.executable, 'main.py',
        '-d', dataset,
        '-m', model_path,
        '-b', str(batch_size),
        '-a', attack_mode,
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Evaluate Real Network Specialist Model')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Model path (e.g. src/train/v1/checkpoints/real_network_specialist_seed42.pth)')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory containing specialist models (for multi-seed)')
    parser.add_argument('--datasets', type=str, default='Colt,Chinanet,UsCarrier',
                        help='Comma-separated datasets to evaluate')
    parser.add_argument('--run-main', action='store_true',
                        help='Run main.py for full comparison with other methods')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for main.py')
    parser.add_argument('--attack-mode', type=str, default='degree',
                        choices=['degree', 'betweenness'])
    
    args = parser.parse_args()
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    # 确定模型路径
    models_to_eval = []
    if args.model:
        if os.path.exists(args.model):
            models_to_eval.append(args.model)
        else:
            print(f"Warning: Model not found: {args.model}")
    
    if not models_to_eval and args.model_dir:
        # 查找 specialist 模型
        pattern = os.path.join(args.model_dir, 'real_network_specialist*.pth')
        models_to_eval = glob.glob(pattern)
        models_to_eval.sort()
    
    if not models_to_eval:
        print("Error: No model found. Specify -m or ensure models exist in --model-dir")
        return 1
    
    print("="*60)
    print("Real Network Specialist Evaluation")
    print("="*60)
    print(f"Models: {models_to_eval}")
    print(f"Datasets: {datasets}")
    print(f"Run main.py (full comparison): {args.run_main}")
    print("="*60)
    
    if args.run_main:
        for model_path in models_to_eval:
            for dataset in datasets:
                ok = run_main_with_model(
                    dataset, model_path,
                    batch_size=args.batch_size,
                    attack_mode=args.attack_mode,
                )
                if not ok:
                    print(f"  Failed: {dataset} with {model_path}")
    else:
        print("\nTo run full comparison with other methods, use --run-main")
        print("Example:")
        for model_path in models_to_eval[:1]:
            for dataset in datasets[:1]:
                print(f"  python main.py -d {dataset} -m {model_path} --batch-size 1")
    
    print("\n" + "="*60)
    print("Evaluation setup complete.")
    print("Results will be in results/<dataset>/<attack_mode>/<experiment_id>/")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
