# -*- coding: utf-8 -*-
"""
轻量监控训练：解析 Unified-PPO / BiT-HyRL 日志中的 epoch/reward，检测平台期。
"""
import os
import re
import sys
import glob
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_log(log_path):
    """从训练日志解析 (epoch, avg_reward, best_reward) 列表。支持 Unified-PPO 与 BiT-HyRL optimized 两种格式。"""
    # 格式 1: Epoch 10/80 | Avg Reward: 0.7046 | Best: 0.7046
    pattern1 = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+Avg Reward:\s+([0-9.]+)\s+\|\s+Best:\s+([0-9.]+)"
    )
    # 格式 2: Training Deep GAT+PPO: ... | 37/200 [..., reward=0.4980, best=0.5604, ...]
    pattern2 = re.compile(
        r"Training Deep GAT\+PPO:.*\|\s+(\d+)/(\d+)\s+\[.*reward=([0-9.]+),\s*best=([0-9.]+)"
    )
    rows = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern1.search(line)
            if m:
                rows.append({
                    'epoch': int(m.group(1)),
                    'total': int(m.group(2)),
                    'avg_reward': float(m.group(3)),
                    'best_reward': float(m.group(4)),
                })
                continue
            m = pattern2.search(line)
            if m:
                rows.append({
                    'epoch': int(m.group(1)),
                    'total': int(m.group(2)),
                    'avg_reward': float(m.group(3)),
                    'best_reward': float(m.group(4)),
                })
    return rows


def find_latest_log():
    patterns = [
        'logs/train_unified_ppo_*.log',
        'logs/train_optimized_*.log',
        'logs/train_bit_hyrl_*.log',
    ]
    all_logs = []
    for p in patterns:
        all_logs.extend(glob.glob(str(ROOT / p)))
    # 按修改时间取最新
    all_logs.sort(key=lambda p: Path(p).stat().st_mtime)
    return all_logs[-1] if all_logs else None


def monitor(log_path, patience=15, auto_run=False, model_path=None, B_budget=5,
            run_gap_analysis=False, summary_csv="results/unified_ppo_comparison_summary_v3.csv",
            target="Unified-PPO", output_dir="results/unified_ppo_gap_model"):
    rows = parse_log(log_path)
    if not rows:
        print(f"[warn] 未从日志解析到 epoch 信息: {log_path}")
        return 1

    last = rows[-1]
    total = last['total']
    current_epoch = last['epoch']
    best_reward = last['best_reward']

    # 计算最佳奖励出现的最近 epoch
    best_epoch = None
    for r in reversed(rows):
        if abs(r['best_reward'] - best_reward) < 1e-9 and r['avg_reward'] >= best_reward - 1e-9:
            best_epoch = r['epoch']
            break
    if best_epoch is None:
        best_epoch = max(r['epoch'] for r in rows if abs(r['best_reward'] - best_reward) < 1e-9)

    epochs_since_best = current_epoch - best_epoch
    plateau = epochs_since_best >= patience

    print("=" * 60)
    print("训练监控 (Unified-PPO / BiT-HyRL)")
    print("=" * 60)
    print(f"日志文件: {log_path}")
    print(f"总 Epoch: {total}")
    print(f"当前 Epoch: {current_epoch}")
    print(f"当前 Avg Reward: {last['avg_reward']:.4f}")
    print(f"历史最佳: {best_reward:.4f} (Epoch {best_epoch})")
    print(f"距最佳已过去: {epochs_since_best} epochs")
    print(f"平台期阈值 (patience): {patience}")
    print("=" * 60)

    if plateau:
        print(f"[ALERT] 已连续 {epochs_since_best} 个 epoch 未刷新最佳奖励，判定为平台期。")
        if auto_run and model_path:
            print("[auto-run] 正在启动对比实验 ...")
            cmd = [
                sys.executable, "scripts/run_real_network_comparison.py",
                "--model-path", model_path,
                "--B-budget", str(B_budget),
                "--output", "results/unified_ppo_comparison_summary.csv",
            ]
            subprocess.run(cmd, cwd=ROOT, check=False)
        else:
            print("建议运行以下命令进行性能对比：")
            print(f"  python scripts/run_real_network_comparison.py \\")
            print(f"      --model-path {model_path or 'src/train/v1/checkpoints/unified_ppo_agent_full_v3.pth'} \\")
            print(f"      --B-budget {B_budget}")

        if run_gap_analysis:
            print(f"[auto-run] 正在启动 {target} 差距分析模型 ...")
            cmd = [
                sys.executable, "scripts/analyze_unified_ppo_gap_model.py",
                "--summary", summary_csv,
                "--target", target,
                "--output-dir", output_dir,
            ]
            subprocess.run(cmd, cwd=ROOT, check=False)
        else:
            print("建议运行以下命令进行差距分析：")
            print(f"  python scripts/analyze_unified_ppo_gap_model.py \\")
            print(f"      --summary {summary_csv} \\")
            print(f"      --target {target} \\")
            print(f"      --output-dir {output_dir}")
    else:
        remaining = patience - epochs_since_best
        print(f"[OK] 尚未达到平台期，预计再过 {remaining} 个 epoch 无改善则触发告警。")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="训练日志路径，默认取 logs/ 下最新的训练日志（Unified-PPO / BiT-HyRL）")
    parser.add_argument("--patience", type=int, default=15, help="连续多少 epoch 未刷新最佳奖励判定为平台期")
    parser.add_argument("--auto-run", action="store_true", help="检测到平台期后自动运行对比脚本")
    parser.add_argument("--run-gap-analysis", action="store_true",
                        help="检测到平台期后自动运行差距分析模型（需已有 summary CSV）")
    parser.add_argument("--summary-csv", default="results/unified_ppo_comparison_summary_v3.csv",
                        help="差距分析使用的对比汇总 CSV")
    parser.add_argument("--target", default=None,
                        help="差距分析目标方法，默认根据日志名自动推断（BiT-HyRL / Unified-PPO）")
    parser.add_argument("--output-dir", default=None,
                        help="差距分析输出目录，默认根据 target 自动生成")
    parser.add_argument("--model-path", default="src/train/v1/checkpoints/unified_ppo_agent_full_v3.pth",
                        help="平台期后若 --auto-run 则运行对比实验使用的模型路径")
    parser.add_argument("--B-budget", type=int, default=5)
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    if not log_path or not os.path.exists(log_path):
        print(f"[error] 未找到训练日志: {log_path}")
        return 1

    # 根据日志名自动推断 target
    log_name = Path(log_path).name.lower()
    if args.target:
        target = args.target
    elif 'optimized' in log_name or 'bit_hyrl' in log_name:
        target = "BiT-HyRL"
    else:
        target = "Unified-PPO"

    output_dir = args.output_dir or ("results/bithyrl_gap_model" if target == "BiT-HyRL" else "results/unified_ppo_gap_model")

    return monitor(log_path, args.patience, args.auto_run, args.model_path, args.B_budget,
                   args.run_gap_analysis, args.summary_csv, target, output_dir)


if __name__ == "__main__":
    sys.exit(main())
