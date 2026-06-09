# -*- coding: utf-8 -*-
"""
BiT-HyRL 训练历史可视化模块。

功能:
  - 绘制单个训练历史文件的训练曲线
  - 批量绘制所有训练历史文件
  - 支持训练集和测试集奖励对比
  - 支持移动平均线

用法:
  # 作为模块导入
  from src.utils.visualization.training_history import plot_all_training_histories, plot_training_history
  
  # 命令行使用
  python -m src.utils.visualization.training_history
  python -m src.utils.visualization.training_history --file src/train/v1/training_history_robustness.csv
  python -m src.utils.visualization.training_history --output_dir results/training_curves
"""
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history_file, output_dir=None, show=False):
    """
    绘制单个训练历史文件的训练曲线。
    
    Args:
        history_file (str): 训练历史CSV文件路径
        output_dir (str, optional): 输出目录，默认与输入文件同目录
        show (bool): 是否显示图形（默认False，仅保存）
        
    Returns:
        str: 保存的图片路径，失败返回None
    """
    if not os.path.exists(history_file):
        print(f"Error: History file not found: {history_file}")
        return None

    try:
        df = pd.read_csv(history_file)
        
        # 从文件名提取模式名称
        basename = os.path.basename(history_file)
        if basename == 'training_history.csv':
            mode = 'Combined (Legacy)'
        elif basename.startswith('rl_agent_') and basename.endswith('_training_history.csv'):
            mode = basename.replace('rl_agent_', '').replace('_training_history.csv', '').capitalize()
        elif basename.startswith('gnn_ppo_agent_') and basename.endswith('_training_history.csv'):
            mode = 'GNN PPO ' + basename.replace('gnn_ppo_agent_', '').replace('_training_history.csv', '').capitalize()
        else:
            mode = basename.replace('training_history_', '').replace('.csv', '').capitalize()
        
        plt.figure(figsize=(12, 7))
        
        # 检查列名
        if 'Train_Reward' in df.columns and 'Test_Reward' in df.columns:
            epochs = df['Epoch'].values
            train_rewards = df['Train_Reward'].values
            test_rewards = df['Test_Reward'].values
            
            # 绘制训练和测试曲线
            plt.plot(epochs, train_rewards, 'b-', linewidth=2, label='Train Reward', marker='o', markersize=3)
            plt.plot(epochs, test_rewards, 'r-', linewidth=2, label='Test Reward', marker='s', markersize=3)
            
            # 添加移动平均线
            if len(epochs) > 10:
                window = min(10, len(epochs) // 5)
                train_ma = pd.Series(train_rewards).rolling(window=window, min_periods=1).mean()
                test_ma = pd.Series(test_rewards).rolling(window=window, min_periods=1).mean()
                plt.plot(epochs, train_ma, 'b--', linewidth=1.5, alpha=0.5, label=f'Train MA (w={window})')
                plt.plot(epochs, test_ma, 'r--', linewidth=1.5, alpha=0.5, label=f'Test MA (w={window})')
            
            # 标注最佳测试点
            best_test_idx = np.argmax(test_rewards)
            plt.annotate(f'Best Test: {test_rewards[best_test_idx]:.4f}', 
                        xy=(epochs[best_test_idx], test_rewards[best_test_idx]),
                        xytext=(epochs[best_test_idx] + len(epochs)*0.05, test_rewards[best_test_idx]),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green')
            
        elif 'Reward' in df.columns or 'Avg_Reward' in df.columns:
            # 单列奖励格式（支持 Epoch, Reward 或 Epoch, Avg_Reward）
            reward_col = 'Reward' if 'Reward' in df.columns else 'Avg_Reward'
            epoch_col = 'Epoch' if 'Epoch' in df.columns else df.columns[0]
            epochs = df[epoch_col].values
            rewards = df[reward_col].values
            plt.plot(epochs, rewards, 'b-', linewidth=2, label='Reward', marker='o', markersize=3)
            
            # 添加移动平均线
            if len(epochs) > 10:
                window = min(10, len(epochs) // 5)
                ma = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                plt.plot(epochs, ma, 'r--', linewidth=1.5, alpha=0.7, label=f'Moving Avg (w={window})')
                
            # 标注最佳点
            best_idx = np.argmax(rewards)
            plt.annotate(f'Best: {rewards[best_idx]:.4f}', 
                        xy=(epochs[best_idx], rewards[best_idx]),
                        xytext=(epochs[best_idx] + len(epochs)*0.05, rewards[best_idx]),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green')
        else:
            print(f"Warning: Unknown column format in {history_file}")
            plt.close()
            return None
        
        plt.title(f'BiT-HyRL Training History - {mode}', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # 确定输出路径
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'training_curve_{mode.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_filename = f'training_curve_{mode.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
            output_path = os.path.join(os.path.dirname(history_file), output_filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return output_path
        
    except Exception as e:
        print(f"Error plotting {history_file}: {e}")
        return None


def plot_all_training_histories(models_dir='models', output_dir=None, show=False):
    """
    批量绘制所有训练历史文件的训练曲线。
    
    Args:
        models_dir (str): 模型目录，默认'models'
        output_dir (str, optional): 输出目录，默认与模型目录相同
        show (bool): 是否显示图形
        
    Returns:
        list: 成功保存的图片路径列表
    """
    # 搜索所有历史文件
    patterns = [
        os.path.join(models_dir, 'training_history_*.csv'),
        os.path.join(models_dir, 'training_history.csv'),
        os.path.join(models_dir, 'rl_agent_*_training_history.csv'),
        os.path.join(models_dir, 'gnn_ppo_agent_*_training_history.csv'),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    # 去重
    files = list(set(files))
    
    if not files:
        print(f"No training history files found in {models_dir}")
        return []
    
    print(f"Found {len(files)} training history file(s)")
    
    saved_paths = []
    for f in files:
        path = plot_training_history(f, output_dir=output_dir, show=show)
        if path:
            saved_paths.append(path)
    
    return saved_paths


def plot_comparison(history_files, labels=None, output_path='src/train/v1/training_comparison.png', show=False):
    """
    在同一张图上比较多个训练历史。
    
    Args:
        history_files (list): 训练历史CSV文件路径列表
        labels (list, optional): 图例标签列表
        output_path (str): 输出文件路径
        show (bool): 是否显示图形
        
    Returns:
        str: 保存的图片路径
    """
    plt.figure(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, history_file in enumerate(history_files):
        if not os.path.exists(history_file):
            print(f"Warning: File not found: {history_file}")
            continue
            
        try:
            df = pd.read_csv(history_file)
            
            # 确定标签
            if labels and i < len(labels):
                label = labels[i]
            else:
                basename = os.path.basename(history_file)
                label = basename.replace('training_history_', '').replace('_training_history', '').replace('.csv', '').replace('rl_agent_', '').replace('gnn_ppo_agent_', '').capitalize()
            
            color = colors[i % len(colors)]
            
            reward_col = 'Reward' if 'Reward' in df.columns else ('Avg_Reward' if 'Avg_Reward' in df.columns else None)
            if 'Test_Reward' in df.columns:
                plt.plot(df['Epoch'], df['Test_Reward'], color=color, linewidth=2, 
                        label=f'{label} (Test)', marker='o', markersize=2)
            elif reward_col:
                epoch_col = 'Epoch' if 'Epoch' in df.columns else df.columns[0]
                plt.plot(df[epoch_col], df[reward_col], color=color, linewidth=2, 
                        label=label, marker='o', markersize=2)
                        
        except Exception as e:
            print(f"Error reading {history_file}: {e}")
    
    plt.title('BiT-HyRL Training Comparison', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return output_path


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description='BiT-HyRL 训练历史可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制所有训练历史
  python -m src.utils.visualization.training_history
  
  # 绘制指定文件
  python -m src.utils.visualization.training_history --file src/train/v1/training_history_robustness.csv
  
  # 指定输出目录
  python -m src.utils.visualization.training_history --output_dir results/training_curves
  
  # 比较多个训练历史
  python -m src.utils.visualization.training_history --compare src/train/v1/training_history_robustness.csv src/train/v1/training_history_combined.csv
        """
    )
    
    parser.add_argument('--file', type=str, help='指定单个训练历史文件')
    parser.add_argument('--models_dir', type=str, default='models', help='模型目录（默认: models）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--show', action='store_true', help='显示图形（默认仅保存）')
    parser.add_argument('--compare', nargs='+', help='比较多个训练历史文件')
    parser.add_argument('--labels', nargs='+', help='比较图的标签')
    
    args = parser.parse_args()
    
    if args.compare:
        plot_comparison(args.compare, labels=args.labels, 
                       output_path=os.path.join(args.output_dir or 'models', 'training_comparison.png'),
                       show=args.show)
    elif args.file:
        plot_training_history(args.file, output_dir=args.output_dir, show=args.show)
    else:
        plot_all_training_histories(models_dir=args.models_dir, output_dir=args.output_dir, show=args.show)


if __name__ == '__main__':
    main()
