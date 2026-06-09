import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_history(history_file=None, mode_name=None):
    # 自动搜索所有历史文件
    if history_file is None:
        import glob
        files = glob.glob('src/train/v1/training_history_*.csv')
        # 兼容旧文件名
        if os.path.exists('src/train/v1/training_history.csv'):
            files.append('src/train/v1/training_history.csv')
            
        if not files:
            print("No training history files found in src/train/v1/ directory.")
            return
            
        for f in files:
            plot_single_file(f)
    else:
        plot_single_file(history_file)

def plot_single_file(history_file):
    if not os.path.exists(history_file):
        print(f"Error: History file not found at {history_file}")
        return

    try:
        df = pd.read_csv(history_file)
        
        # 从文件名提取模式名称
        basename = os.path.basename(history_file)
        if basename == 'training_history.csv':
            mode = 'Combined (Legacy)'
        else:
            # 提取 training_history_xxx.csv 中的 xxx
            mode = basename.replace('training_history_', '').replace('.csv', '').capitalize()
            
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train_Reward'], label='Train Reward', marker='o', markersize=3)
        plt.plot(df['Epoch'], df['Test_Reward'], label='Test Reward', marker='s', markersize=3)
        
        plt.title(f'RL Agent Training History - {mode}')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存为对应名称的图片
        output_filename = f'training_curve_{mode.lower().replace(" ", "_")}.png'
        output_file = os.path.join(os.path.dirname(history_file), output_filename)
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting {history_file}: {e}")

if __name__ == "__main__":
    plot_history()
