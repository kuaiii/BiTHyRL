# -*- coding: utf-8 -*-
"""BiT-HyRL 攻击仿真与绘图（四指标折线图、CSV 保存）。"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator

from src.utils.logger import get_logger
from . import config

logger = get_logger(__name__)


def _setup_matplotlib():
    try:
        from matplotlib import font_manager
        fp = 'C:/Windows/Fonts/simhei.ttf'
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams['font.sans-serif'] = ['SimHei']
        else:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False


def simulate_attack_and_plot(G, controllers, attack_mode='degree', output_dir=None):
    """
    对重构后的网络进行攻击仿真，绘制四指标（GCC/CSA/CCE/WCP）的
    指标-移除比例、指标-攻击步数 折线图，并保存 CSV。
    """
    from src.simulation.dismantling import node_attack
    from network_metrics import compute as _nm_compute

    _setup_matplotlib()
    logger.info(f"开始攻击仿真: 攻击模式={attack_mode}")
    print(f"\n开始攻击仿真: 攻击模式={attack_mode}")

    n = G.number_of_nodes()
    metrics_dict, _, attack_steps_dict = node_attack(G.copy(), n, set(controllers), attack_mode)

    if output_dir is None:
        output_dir = config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    metric_types = ['gcc', 'csa', 'cce', 'wcp']
    labels = {
        'gcc': 'GCC (Giant Connected Component)',
        'csa': 'CSA (Control Supply Availability)',
        'cce': 'CCE (Control Entropy)',
        'wcp': 'WCP (Weighted Control Potential)',
    }
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5, 'axes.facecolor': '#f8f9fa'})
    sns.set_context("notebook", font_scale=1.2)
    results_data = {}

    for mt in metric_types:
        if mt not in metrics_dict:
            continue
        curve = metrics_dict[mt]
        steps_data = attack_steps_dict.get(mt, [])
        x_remove = [p[1] for p in curve]
        y_metric = [p[0] for p in curve]
        if steps_data:
            x_steps = [p[2] for p in steps_data]
            y_steps = [p[0] for p in steps_data]
            ml = min(len(y_metric), len(x_steps))
            x_steps = x_steps[:ml]
            y_steps = y_steps[:ml]
        else:
            x_steps = list(range(len(y_metric)))
            y_steps = y_metric

        r_val = _nm_compute('r_value_interpolated', x_remove, y_metric, num_points=101)
        results_data[mt] = {
            'remove_ratio': x_remove, 'metric_value': y_metric,
            'attack_steps': x_steps if steps_data else None,
            'metric_value_vs_steps': y_steps if steps_data else None,
            'r_value': r_val,
        }

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(x_remove, y_metric, color='#4C72B0', linestyle='-', marker='o', markersize=4, linewidth=1.5, label='BiT-HyRL')
        ax1.xaxis.set_major_locator(MultipleLocator(0.2))
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Proportion of Removed Nodes : f", fontsize=12)
        ax1.set_ylabel(labels[mt], fontsize=12)
        ax1.set_title(f"{attack_mode.upper()} Attack - {labels[mt]} vs Removal Ratio", fontsize=14)
        ax1.legend(fontsize=10, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        p1 = os.path.join(output_dir, f"{mt.upper()}_vs_RemovalRatio_{attack_mode}.png")
        plt.savefig(p1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {p1}")

        if steps_data and len(x_steps) > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(x_steps, y_steps, color='#C44E52', linestyle='--', marker='s', markersize=4, linewidth=1.5, label='BiT-HyRL')
            ax2.set_xlabel("Attack Steps", fontsize=12)
            ax2.set_ylabel(labels[mt], fontsize=12)
            ax2.set_title(f"{attack_mode.upper()} Attack - {labels[mt]} vs Attack Steps", fontsize=14)
            ax2.legend(fontsize=10, loc='best', framealpha=0.9)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            p2 = os.path.join(output_dir, f"{mt.upper()}_vs_AttackSteps_{attack_mode}.png")
            plt.savefig(p2, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已保存: {p2}")

        csv_data = {'Remove_Ratio': x_remove, f'{mt.upper()}_Value': y_metric}
        if steps_data and len(x_steps) > 0:
            ext = x_steps + [x_steps[-1] if x_steps else 0] * (len(x_remove) - len(x_steps)) if len(x_steps) < len(x_remove) else x_steps[:len(x_remove)]
            csv_data['Attack_Steps'] = ext
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f"{mt.upper()}_results_{attack_mode}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  已保存: {csv_path}")

    summary = {'Metric': [], 'R_Value': [], 'Collapse_Point': []}
    for mt in metric_types:
        if mt not in results_data:
            continue
        summary['Metric'].append(mt.upper())
        summary['R_Value'].append(results_data[mt]['r_value'])
        y = results_data[mt]['metric_value']
        if y:
            thresh = y[0] * 0.2
            cp = None
            for i, v in enumerate(y):
                if v <= thresh:
                    cp = results_data[mt]['remove_ratio'][i]
                    break
            summary['Collapse_Point'].append(cp if cp is not None else 1.0)
        else:
            summary['Collapse_Point'].append(1.0)
    sp = os.path.join(output_dir, f"summary_statistics_{attack_mode}.csv")
    pd.DataFrame(summary).to_csv(sp, index=False)
    print(f"  已保存: {sp}")

    logger.info(f"攻击仿真完成，结果已保存到: {output_dir}")
    print(f"\n攻击仿真完成，所有结果已保存到: {os.path.abspath(output_dir)}")
    return results_data
