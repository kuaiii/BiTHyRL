# -*- coding: utf-8 -*-
import argparse
import networkx as nx
import os
import random
import logging
import numpy as np
from math import ceil
from tqdm import tqdm
import torch

# 设置基础随机种子（仅用于初始化，各batch会使用不同种子以产生变化）
BASE_RANDOM_SEED = 42
random.seed(BASE_RANDOM_SEED)
np.random.seed(BASE_RANDOM_SEED)

from src.topology.generators import load_graph, construct_random, construct_ba
from src.topology.reconstruction import (
    create_bimodal_adaptive_robust,
    create_bimodal_network_exact,
    create_bimodal_theoretical,
)
import network_construction as nc  # 统一重构接口
from src.controller.manager import ControllerManager
from src.unified_ppo.inference import unified_solve
from src.unified_ppo.topology_policy import TopologyPolicy
from src.utils.visualization.result_plotter import plot_collapse_point_charts, plot_attack_steps_for_batches, plot_metric_vs_attack_steps
from src.utils.io import save_comprehensive_metrics, save_execution_times, save_collapse_point_data, save_area_data, save_attack_steps_data, save_r_values_data
from src.utils.logger import setup_logger, get_logger, algorithm_timer
from network_metrics import compute as _nm_compute
import time
import numpy as np

logger = get_logger(__name__)

def extract_curve_data(curve_data, attack_steps_data):
    """
    从曲线数据中提取x、y和步数数据。
    
    Args:
        curve_data: 曲线数据列表 [(value, x_ratio), ...]
        attack_steps_data: 攻击步数数据列表 [(value, x_ratio, step), ...]
    
    Returns:
        tuple: (x_curve, y_curve, steps_curve)
    """
    x_curve = [point[1] for point in curve_data] if curve_data else []
    y_curve = [point[0] for point in curve_data] if curve_data else []
    steps_curve = [point[2] for point in attack_steps_data] if attack_steps_data else []
    return x_curve, y_curve, steps_curve

def calculate_collapse_point(x_curve, y_curve, threshold_ratio=0.2):
    """
    计算崩溃点（指标下降到初始值的指定比例时的移除节点比例）。
    
    Args:
        x_curve: x轴数据（移除节点比例）
        y_curve: y轴数据（指标值）
        threshold_ratio: 阈值比例，默认0.2（20%）
    
    Returns:
        float: 崩溃点值，如果未找到则返回最后一个x值
    """
    if not y_curve:
        return 1.0
    
    initial_value = y_curve[0] if y_curve[0] > 0 else 1.0
    threshold = initial_value * threshold_ratio
    
    for i, y_val in enumerate(y_curve):
        if y_val <= threshold:
            return x_curve[i] if i < len(x_curve) else 1.0
    
    return x_curve[-1] if x_curve else 1.0


def merge_curves_weighted(x_degree, y_degree, steps_degree, x_random, y_random, steps_random, initial_nodes, num_points=101):
    """
    将 degree 与 random 两条曲线按 0.5*degree + 0.5*random 加权合并。
    使用公共 x 轴（removed_ratio 0~1）插值后加权。
    
    Returns:
        (x_common, y_weighted, steps_weighted): 列表
    """
    if not x_degree or not y_degree:
        x_common = np.linspace(0, 1, num_points).tolist()
        y_weighted = np.interp(x_common, x_random or [0, 1], y_random or [0, 0]).tolist()
        steps_weighted = (np.array(x_common) * initial_nodes).tolist()
        return x_common, y_weighted, steps_weighted
    if not x_random or not y_random:
        x_common = np.linspace(0, 1, num_points).tolist()
        y_weighted = np.interp(x_common, x_degree, y_degree).tolist()
        steps_weighted = (np.array(x_common) * initial_nodes).tolist()
        return x_common, y_weighted, steps_weighted
    x_common = np.linspace(0, 1, num_points).tolist()
    y_d = np.interp(x_common, x_degree, y_degree)
    y_r = np.interp(x_common, x_random, y_random)
    y_weighted = (0.5 * y_d + 0.5 * y_r).tolist()
    steps_weighted = (np.array(x_common) * initial_nodes).tolist()
    return x_common, y_weighted, steps_weighted


def process_metrics_from_manager(manager, metric_types, curves_data_by_metric, 
                                  collapse_points_by_metric, area_values_by_metric, 
                                  attack_steps_by_metric, initial_nodes):
    """
    从manager的metrics_summary中提取并处理所有指标数据。
    
    Args:
        manager: ControllerManager实例
        metric_types: 指标类型列表
        curves_data_by_metric: 曲线数据字典（会被更新）
        collapse_points_by_metric: 崩溃点数据字典（会被更新）
        area_values_by_metric: 面积数据字典（会被更新）
        attack_steps_by_metric: 攻击步数数据字典（会被更新）
        initial_nodes: 初始节点数
    """
    if not hasattr(manager, 'metrics_summary'):
        return
    
    for name, metrics_list in manager.metrics_summary.items():
        if not metrics_list:
            continue
        
        current_metric = metrics_list[-1]
        
        for metric_type in metric_types:
            curve_key = f'{metric_type.upper()}_Curve'
            attack_steps_key = f'{metric_type.upper()}_AttackSteps'
            curve_data = current_metric.get(curve_key, [])
            attack_steps_data = current_metric.get(attack_steps_key, [])
            
            if not curve_data:
                continue
            
            x_curve, y_curve, steps_curve = extract_curve_data(curve_data, attack_steps_data)
            
            # 计算removed_nodes_ratio = attack_step/initial_nodes
            removed_ratio_list = [step / initial_nodes for step in steps_curve] if steps_curve else x_curve
            
            # 计算R值（面积）：统一使用removed_nodes_ratio
            r_val = _nm_compute('r_value_interpolated', removed_ratio_list, y_curve, num_points=101)
            
            # 保存曲线数据
            if name not in curves_data_by_metric[metric_type]:
                curves_data_by_metric[metric_type][name] = []
            curves_data_by_metric[metric_type][name].append((x_curve, y_curve, r_val))
            
            # 保存攻击步数数据（包含R值用于图例显示）
            if name not in attack_steps_by_metric[metric_type]:
                attack_steps_by_metric[metric_type][name] = []
            attack_steps_by_metric[metric_type][name].append((x_curve, y_curve, steps_curve, r_val))
            
            # 保存面积数据
            if name not in area_values_by_metric[metric_type]:
                area_values_by_metric[metric_type][name] = []
            area_values_by_metric[metric_type][name].append(r_val)

def save_iteration_curves(manager, metric_types, ijk, current_attack_mode, extra_data_dir):
    """
    保存每次迭代的详细曲线数据到CSV文件。
    
    Args:
        manager: ControllerManager实例
        metric_types: 指标类型列表
        ijk: 当前迭代索引
        current_attack_mode: 当前攻击模式
        extra_data_dir: 保存目录
    """
    if not hasattr(manager, 'metrics_summary'):
        return
    
    import pandas as pd
    
    for name, metrics_list in manager.metrics_summary.items():
        if not metrics_list:
            continue
        
        current_metric = metrics_list[-1]
        
        for metric_type in metric_types:
            curve_key = f'{metric_type.upper()}_Curve'
            attack_steps_key = f'{metric_type.upper()}_AttackSteps'
            curve_data = current_metric.get(curve_key, [])
            attack_steps_data = current_metric.get(attack_steps_key, [])
            
            if not curve_data:
                continue
            
            x_curve, y_curve, steps_curve = extract_curve_data(curve_data, attack_steps_data)
            
            df = pd.DataFrame({
                'x_val': x_curve,
                'y_val': y_curve,
                'attack_step': steps_curve if steps_curve else [0] * len(x_curve)
            })
            
            filename = f"{name}_iter{ijk}_{current_attack_mode}_{metric_type}.csv"
            filepath = os.path.join(extra_data_dir, filename)
            df.to_csv(filepath, index=False)

def parse_arguments():
    """
    解析命令行参数。
    
    参数分组：
    - 基础参数：数据集、攻击模式、批次等
    - 模型参数：GNN 模型路径（支持课程学习模型）
    """
    parser = argparse.ArgumentParser(
        description='SDN Topology Resilience Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 使用默认课程学习模型运行对比实验
  python main.py --dataset GtsCe --attack random

  # 指定 GNN 模型路径
  python main.py --dataset GtsCe --model src/train/v1/checkpoints/curriculum_final_dynamic_gat.pth

  # 多批次实验
  python main.py --dataset GtsCe --batch 5 --attack degree
        '''
    )
    
    parser.add_argument('--dataset', '-d', type=str, default='GtsCe', 
                        help='数据集名称 (默认: GtsCe)')
    parser.add_argument('--rate', '-r', type=float, default=0.1, 
                        help='控制器部署比例 (默认: 0.1)')
    parser.add_argument('--batch', '-b', type=int, default=1, dest='batch_size',
                        help='仿真批次数 (默认: 1)')
    parser.add_argument('--attack', '-a', type=str, default='random', 
                        choices=['random', 'degree', 'betweenness', 'pagerank', 'eigenvector', 'bruteforce', 'wgcc'], 
                        help='攻击模式 (默认: random)。wgcc=同时跑 degree 与 random，对 GCC/CSA/CCE/WCP 求加权 (0.5*degree+0.5*random)，结果存 results/{dataset}/wgcc/')
    parser.add_argument('--debug', action='store_true', 
                        help='启用调试日志')
    
    parser.add_argument('--model', '-m', type=str, default=None, dest='gnn_model',
                        help='GNN 模型路径。默认自动搜索: curriculum_final > curriculum_phase3 > deep_gat > ...')
    parser.add_argument('--unified-ppo-model', type=str, default=None,
                        help='Unified PPO 模型路径。指定后将在对比中加入 Unified-PPO 方法')
    parser.add_argument('--unified-ppo-B', type=int, default=3,
                        help='Unified PPO 拓扑微调预算 (默认 3)')
    parser.add_argument('--unified-ppo-M', type=int, default=50,
                        help='Unified PPO 候选边池大小 (默认 50)')
    parser.add_argument('--bimodal-strategy', type=str, default='adaptive_robust',
                        choices=['adaptive_robust', 'theoretical', 'exact'],
                        help='BiT-HyRL 拓扑构造策略 (默认: adaptive_robust)')
    parser.add_argument('--bimodal-max-hub-ratio', type=float, default=0.25,
                        help='adaptive_robust 的最大 hub 比例 (默认: 0.25)')
    parser.add_argument('--bimodal-samples', type=int, default=12,
                        help='adaptive_robust 的 hub 数量采样数 (默认: 12)')
    parser.add_argument('--bimodal-attacks', type=str, default='degree,betweenness,random',
                        help='adaptive_robust 拓扑筛选使用的攻击集合')
    parser.add_argument('--bimodal-score-profile', type=str, default='auto',
                        choices=['auto', 'balanced', 'targeted', 'random', 'wgcc'],
                        help='adaptive_robust 拓扑评分权重配置 (默认: auto)')
    
    # 高级参数（向后兼容）
    parser.add_argument('--hub_ratio', type=float, default=0.15, 
                        help=argparse.SUPPRESS)  # BiT-HyRL: Hub节点比例
    parser.add_argument('--episodes', type=int, default=100, 
                        help=argparse.SUPPRESS)  # BiT-HyRL: RL训练轮数
    parser.add_argument('--metric_type', type=str, default='robustness',
                        choices=['combined', 'robustness', 'csa', 'entropy', 'wcp', 'gcc'],
                        help=argparse.SUPPRESS)  # BiT-HyRL: 优化目标类型
    parser.add_argument('--use_node2vec', action='store_true', default=True, 
                        help=argparse.SUPPRESS)
    parser.add_argument('--no_node2vec', dest='use_node2vec', action='store_false',
                        help=argparse.SUPPRESS)
    parser.add_argument('--use_stepwise', action='store_true', default=True, 
                        help=argparse.SUPPRESS)
    parser.add_argument('--no_stepwise', dest='use_stepwise', action='store_false',
                        help=argparse.SUPPRESS)
    parser.add_argument('--use_ci', action='store_true', default=True, 
                        help=argparse.SUPPRESS)
    parser.add_argument('--no_ci', dest='use_ci', action='store_false',
                        help=argparse.SUPPRESS)
    parser.add_argument('--test_mode', action='store_true', default=False, 
                        help=argparse.SUPPRESS)  # 测试模式
    parser.add_argument('--use_gnn', action='store_true', default=True,
                        help=argparse.SUPPRESS)  # 默认启用 GNN
    parser.add_argument('--metric', type=str, default='gcc',
                        choices=['gcc', 'efficiency', 'coverage'],
                        help=argparse.SUPPRESS)
    
    return parser.parse_args()

def load_or_generate_graph(args):
    """
    从 GML 文件加载图。
    训练用的数据集在 dataset/all 中；测试的数据集在 dataset/testdata 中。
    
    Args:
        args (argparse.Namespace): 解析后的参数。
        
    Returns:
        tuple: (G, dataset_name)
    """
    dataset_name = args.dataset

    gml_path = os.path.join('dataset', 'testdata', f'{dataset_name}.gml')
    if not os.path.exists(gml_path):
        gml_path = os.path.join('dataset', 'all', f'{dataset_name}.gml')
        if not os.path.exists(gml_path):
            logger.error(f"未找到数据集文件: {dataset_name}.gml (已搜索 dataset/testdata 和 dataset/all)")
            return None, None

    G, pos = load_graph(gml_path)

    # # region 调试用网络统计写 debug.log，已关闭
    # import json
    # debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
    # try:
    #     os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
    #     degrees = [d for n, d in G.degree()]
    #     ...
    #     with open(debug_log_path, 'a', encoding='utf-8') as f:
    #         f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    # except Exception:
    #     pass
    # # endregion

    return G, dataset_name

def setup_experiment_dir(dataset_name, attack_mode):
    """
    确定实验 ID 并创建结果目录。
    
    Args:
        dataset_name (str): 数据集名称。
        attack_mode (str): 攻击模式。
        
    Returns:
        tuple: (experiment_id)
    """
    base_dir = os.path.join('results', dataset_name, attack_mode)
    experiment_id = 1
    if os.path.exists(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        numeric_subdirs = []
        for d in subdirs:
            try:
                numeric_subdirs.append(int(d))
            except ValueError:
                pass
        if numeric_subdirs:
            experiment_id = max(numeric_subdirs) + 1
    
    # 创建实验主目录
    experiment_dir = os.path.join(base_dir, str(experiment_id))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # 创建日志目录（在实验ID目录下）
    log_dir = os.path.join(experiment_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    setup_logger(log_dir=log_dir, log_filename="experiment.log")
    if not os.environ.get("ALLTEST_QUIET") == "1":
        logger.info(f"实验ID: {experiment_id}, 结果保存路径: {experiment_dir}")
    
    return experiment_id

def run_simulation_batch(G, args, experiment_id, dataset_name):
    """
    运行批量仿真。
    
    Args:
        G (nx.Graph): 原始图。
        args (argparse.Namespace): 参数。
        experiment_id (int): 实验 ID。
        dataset_name (str): 数据集名称。
        
    Returns:
        tuple: (x_values, r_values)
    """
    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()
    cover_rate = args.rate
    batch_size = args.batch_size
    current_attack_mode = args.attack
    is_wgcc_run = (current_attack_mode == 'wgcc')
    
    metric_types = ['gcc', 'csa', 'cce', 'wcp']
    experiment_base_dir = os.path.join('results', dataset_name, current_attack_mode, str(experiment_id))
    if not os.path.exists(experiment_base_dir):
        os.makedirs(experiment_base_dir)
    
    # 为每个指标类型在实验目录下创建子目录，每个指标下再分为三个文件夹
    metric_dirs = {}
    metric_subdirs = {}
    for metric_type in metric_types:
        metric_dir = os.path.join(experiment_base_dir, metric_type)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
        metric_dirs[metric_type] = metric_dir
        
        # 为每个指标创建子文件夹（不再创建curves目录）
        collapse_point_dir = os.path.join(metric_dir, "collapse_point")
        collapse_r_dir = os.path.join(metric_dir, "collapse_R")
        area_dir = os.path.join(metric_dir, "area")
        
        for subdir in [collapse_point_dir, collapse_r_dir, area_dir]:
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        
        metric_subdirs[metric_type] = {
            'collapse_point': collapse_point_dir,
            'collapse_R': collapse_r_dir,
            'area': area_dir
        }
    
    # 创建extradata目录（用于保存每次迭代的详细曲线）
    extra_data_dir = os.path.join(experiment_base_dir, "extradata")
    if not os.path.exists(extra_data_dir):
        os.makedirs(extra_data_dir)
    
    # 共同目录（用于保存运行时间等共同指标）- 保持在attack_mode级别
    common_dir = experiment_base_dir

    # 为每个指标类型初始化 x_values 和 r_values
    x_values_by_metric = {}
    r_values_by_metric = {}
    # 用于保存曲线数据、崩溃点数据、面积数据和攻击步数数据
    curves_data_by_metric = {}  
    collapse_points_by_metric = {}
    area_values_by_metric = {}
    attack_steps_by_metric = {}
    
    for metric_type in metric_types:
        x_values_by_metric[metric_type] = {
            "Baseline": [],
            "GA+RCP": [],
            "GA+RL": [],
            "Onion+RL": [],
            "ROMEN+RL": [],
            "UNITY+RL": [],
            "BiT-HyRL": [],
            "FRED-ABL": [],
            # "TEAM": [],  # TEAM 方法已禁用
            "QDLM": [],
            "Unified-PPO": []
        }
        
        r_values_by_metric[metric_type] = {
            "Baseline": [],
            "GA+RCP": [],
            "GA+RL": [],
            "Onion+RL": [],
            "ROMEN+RL": [],
            "UNITY+RL": [],
            "BiT-HyRL": [],
            "FRED-ABL": [],
            # "TEAM": [],  # TEAM 方法已禁用
            "QDLM": [],
            "Unified-PPO": []
        }
        
        # 初始化曲线数据、崩溃点数据、面积数据和攻击步数数据
        curves_data_by_metric[metric_type] = {}
        collapse_points_by_metric[metric_type] = {}
        area_values_by_metric[metric_type] = {}
        attack_steps_by_metric[metric_type] = {}
    
    plot_flag = True
    
    global_metrics_summary = {}
    
    # 记录运行时间
    execution_times = {}

    for ijk in tqdm(range(batch_size), desc=f"Simulating {dataset_name}"):
        
        iter_start_time = time.time()
        batch_info = f"Batch {ijk + 1}/{batch_size}"
        
        # 为每个batch设置不同的随机种子，确保不同batch产生不同的拓扑变化
        batch_seed = BASE_RANDOM_SEED + ijk
        random.seed(batch_seed)
        np.random.seed(batch_seed)
        
        # 生成对比拓扑（使用统一的日志格式）
        # 1. BA Network
        start_time = time.time()
        with algorithm_timer("BA Network", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
            G_BA = construct_ba(node_num, edge_num)
        execution_times.setdefault("Construct_BA", []).append(time.time() - start_time)
        
        # 2. RA Network (Random)
        start_time = time.time()
        with algorithm_timer("RA Network", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
            G_RA = construct_random(node_num, edge_num)
        execution_times.setdefault("Construct_Random", []).append(time.time() - start_time)
        
        # 3. ONION
        start_time = time.time()
        with algorithm_timer("ONION", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
            G_ONION = nc.construct(G, algorithm='Onion', max_iter=200, verbose=(ijk == 0))  # 减少迭代次数
        execution_times.setdefault("Construct_Onion", []).append(time.time() - start_time)
        
        # 4. ROMEN
        start_time = time.time()
        with algorithm_timer("ROMEN", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
            G_ROMEM = nc.construct(G, algorithm='ROMEN', seed=batch_seed, verbose=(ijk == 0))
        execution_times.setdefault("Construct_ROMEN", []).append(time.time() - start_time)

        # 5. UNITY
        start_time = time.time()
        with algorithm_timer("UNITY", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
            G_UNITY = nc.construct(G, algorithm='UNITY')
        execution_times.setdefault("Construct_UNITY", []).append(time.time() - start_time)
        
        # 6. FRED-ABL
        start_time = time.time()
        try:
            with algorithm_timer("FRED-ABL", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
                # 使用优化参数：减少迭代次数和初始采样数
                G_FRED_ABL = nc.construct(G, algorithm='FRED_ABL', seed=batch_seed, iterations=15, initial_samples=3, verbose=(ijk == 0))
            execution_times.setdefault("Construct_FRED_ABL", []).append(time.time() - start_time)
        except Exception as e:
            logger.warning(f"FRED-ABL构建失败: {e}")
            G_FRED_ABL = None
            execution_times.setdefault("Construct_FRED_ABL", []).append(time.time() - start_time)
        
        # 7. QDLM
        start_time = time.time()
        try:
            with algorithm_timer("QDLM", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
                # 使用优化参数：减少种群大小和迭代次数
                G_QDLM = nc.construct(G, algorithm='QDLM', seed=batch_seed, pop_size=20, max_iterations=25, verbose=(ijk == 0))
            execution_times.setdefault("Construct_QDLM", []).append(time.time() - start_time)
        except Exception as e:
            logger.warning(f"QDLM构建失败: {e}")
            G_QDLM = None
            execution_times.setdefault("Construct_QDLM", []).append(time.time() - start_time)
        
        # 8. Bimodal
        start_time = time.time()
        with algorithm_timer("Bimodal", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
            if args.bimodal_strategy == 'adaptive_robust':
                bimodal_attacks = [m.strip() for m in args.bimodal_attacks.split(',') if m.strip()]
                if args.bimodal_score_profile == 'auto':
                    if current_attack_mode in ('random', 'wgcc'):
                        score_profile = 'wgcc'
                    elif current_attack_mode in ('degree', 'target', 'betweenness'):
                        score_profile = 'targeted'
                    else:
                        score_profile = 'balanced'
                else:
                    score_profile = args.bimodal_score_profile
                G_BiT = create_bimodal_adaptive_robust(
                    G,
                    seed=batch_seed,
                    max_hub_ratio=args.bimodal_max_hub_ratio,
                    num_samples=args.bimodal_samples,
                    attack_methods=bimodal_attacks,
                    score_profile=score_profile,
                    verbose=(ijk == 0 and args.debug),
                )
            elif args.bimodal_strategy == 'exact':
                hub_num = max(1, int(round(node_num * args.hub_ratio)))
                G_BiT = create_bimodal_network_exact(G, hub_num=hub_num, seed=batch_seed)
            else:
                G_BiT = create_bimodal_theoretical(node_num, edge_num, seed=batch_seed)
        execution_times.setdefault("Construct_bimodal", []).append(time.time() - start_time)
        
        # 9. Unified-PPO (按 batch seed 保持可复现)
        G_unified_ppo = None
        unified_ppo_centers = None
        if args.unified_ppo_model is not None and os.path.exists(args.unified_ppo_model):
            start_time = time.time()
            try:
                with algorithm_timer("Unified-PPO", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
                    ck = torch.load(args.unified_ppo_model, map_location='cpu', weights_only=False)
                    cfg = ck.get('model_config', {})
                    in_channels = cfg.get('in_channels', 259)
                    hidden_channels = cfg.get('hidden_channels', 128)
                    heads = cfg.get('heads', 4)
                    num_layers = cfg.get('num_layers', 3)
                    M_candidates = cfg.get('M_candidates', args.unified_ppo_M)
                    ppo_model = TopologyPolicy(
                        in_channels=in_channels, hidden_channels=hidden_channels,
                        heads=heads, num_layers=num_layers, M_candidates=M_candidates
                    )
                    ppo_model.load_state_dict(ck['model_state_dict'])
                    ppo_model = ppo_model.cuda() if torch.cuda.is_available() else ppo_model
                    result = unified_solve(
                        G, ppo_model, B_budget=args.unified_ppo_B, k_ratio=cover_rate,
                        embed_dim=in_channels - 3, seed=batch_seed, deterministic=True
                    )
                    G_unified_ppo = result['G_final']
                    unified_ppo_centers = result['controllers']
            except Exception as e:
                logger.warning(f"Unified-PPO 推理失败: {e}")
                G_unified_ppo = None
                unified_ppo_centers = None
            execution_times.setdefault("Construct_Unified_PPO", []).append(time.time() - start_time)
        
        # 其它方法
        start_time = time.time()
        with algorithm_timer("GA", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
            G_GA = nc.construct(G, algorithm='GA')
        execution_times.setdefault("Construct_GA", []).append(time.time() - start_time)
        
        # TEAM 方法已禁用
        # start_time = time.time()
        # try:
        #     with algorithm_timer("TEAM", verbose=(ijk == 0), batch_info=batch_info if ijk == 0 else None):
        #         # 使用优化参数：减少迭代次数
        #         G_TEAM = nc.construct(G, algorithm='TEAM', seed=batch_seed, verbose=(ijk == 0))
        #     execution_times.setdefault("Construct_TEAM", []).append(time.time() - start_time)
        # except Exception as e:
        #     logger.warning(f"TEAM构建失败: {e}")
        #     G_TEAM = None
        #     execution_times.setdefault("Construct_TEAM", []).append(time.time() - start_time)
        G_TEAM = None  # TEAM 方法已禁用
        
        # 将拓扑组织成字典
        all_G = {
            "G": G,   # 原始
            "G_BA": G_BA, # 1. BA Network
            "G_RA": G_RA, # 2. RA Network (Random)
            "G_ONION": G_ONION, # 3. ONION
            "G_ROMEM": G_ROMEM, # 4. ROMEN
            "G_UNITY": G_UNITY,  # 5. UNITY
            "G_FRED_ABL": G_FRED_ABL, # 6. FRED-ABL
            "G_QDLM": G_QDLM, # 7. QDLM
            "G_BiT": G_BiT, # 8. Bimodal
            "G_unified_ppo": G_unified_ppo, # 9. Unified-PPO
            # 其它方法
            "G_GA": G_GA, # GA 优化
            # "G_TEAM": G_TEAM, # TEAM 优化（已禁用）
        }
        
        # 输出每种方法重构前后的拓扑信息（仅在第一个batch输出；alltest_main 下跳过以简化日志）
        if ijk == 0 and os.environ.get("ALLTEST_QUIET") != "1":
            print("\n" + "=" * 80)
            print("拓扑重构统计信息")
            print("=" * 80)
            print(f"{'方法':<15} {'节点数(前→后)':<20} {'边数(前→后)':<20} {'控制器数':<10}")
            print("-" * 80)
            
            # 定义方法与图的映射
            method_graph_map = [
                ("Baseline", G, G),           # 原始→原始
                ("GA+RCP", G, G),              # 原始→原始（仅控制器选择不同）
                # 其它方法
                ("GA+RL", G, G_GA),            # 原始→GA优化
                # 按顺序：BA Network, RA Network, ONION, ROMEN, UNITY, FRED-ABL, QDLM, Bimodal
                ("Onion+RL", G, G_ONION),      # 原始→ONION优化
                ("ROMEN+RL", G, G_ROMEM),      # 原始→ROMEN优化
                ("UNITY+RL", G, G_UNITY),      # 原始→UNITY优化
                ("FRED-ABL", G, G_FRED_ABL),   # 原始→FRED-ABL优化
                ("QDLM", G, G_QDLM),           # 原始→QDLM优化
                ("BiT-HyRL", G, G_BiT),        # 原始→Bimodal拓扑
                ("Unified-PPO", G, G_unified_ppo), # 原始→Unified-PPO拓扑微调
                # ("TEAM", G, G_TEAM),         # 原始→TEAM优化（已禁用）
            ]
            
            controller_num = ceil(node_num * cover_rate)
            
            for method_name, g_before, g_after in method_graph_map:
                if g_before is not None and g_after is not None:
                    n_before = g_before.number_of_nodes()
                    e_before = g_before.number_of_edges()
                    n_after = g_after.number_of_nodes()
                    e_after = g_after.number_of_edges()
                    
                    # 判断节点/边是否变化
                    n_change = "" if n_before == n_after else f" (Δ{n_after - n_before:+d})"
                    e_change = "" if e_before == e_after else f" (Δ{e_after - e_before:+d})"
                    
                    print(f"{method_name:<15} {n_before:>4} → {n_after:<4}{n_change:<8} {e_before:>4} → {e_after:<4}{e_change:<8} {controller_num}")
                else:
                    print(f"{method_name:<15} {'N/A':<20} {'N/A':<20} {controller_num}")
            
            print("=" * 80 + "\n")
        
        # 传递BiT-HyRL参数到ControllerManager
        manager = ControllerManager(
            all_G, cover_rate, node_num, dataset_name, experiment_id,
            unified_ppo_centers=unified_ppo_centers,
            bit_hyrl_episodes=args.episodes,
            bit_hyrl_metric_type=args.metric_type,
            bit_hyrl_use_node2vec=args.use_node2vec,
            bit_hyrl_use_stepwise=args.use_stepwise,
            bit_hyrl_use_ci=args.use_ci,
            bit_hyrl_test_mode=args.test_mode,
            bit_hyrl_use_gnn=args.use_gnn,
            bit_hyrl_gnn_model=args.gnn_model
        )
        
        # 记录仿真时间（包含攻击和评估）
        sim_start_time = time.time()
        
        if is_wgcc_run:
            # WGCC 作为攻击方式：degree 与 random 各跑一次，对 GCC/CSA/CCE/WCP 四种指标求加权 (0.5*degree+0.5*random)，保存并画指标-攻击步数图
            internal_metric_types = ['gcc', 'csa', 'cce', 'wcp']
            method_names = ["Baseline", "GA+RCP", "GA+RL", "Onion+RL", "ROMEN+RL", "UNITY+RL", "BiT-HyRL", "FRED-ABL", "QDLM", "Unified-PPO"]
            x_degree = {mt: {m: [] for m in method_names} for mt in internal_metric_types}
            r_degree = {mt: {m: [] for m in method_names} for mt in internal_metric_types}
            x_random = {mt: {m: [] for m in method_names} for mt in internal_metric_types}
            r_random = {mt: {m: [] for m in method_names} for mt in internal_metric_types}
            curves_degree = {mt: {} for mt in internal_metric_types}
            collapse_degree = {mt: {} for mt in internal_metric_types}
            area_degree = {mt: {} for mt in internal_metric_types}
            attack_steps_degree = {mt: {} for mt in internal_metric_types}
            curves_random = {mt: {} for mt in internal_metric_types}
            collapse_random = {mt: {} for mt in internal_metric_types}
            area_random = {mt: {} for mt in internal_metric_types}
            attack_steps_random = {mt: {} for mt in internal_metric_types}
            manager._run_specific_attack_multi_metric("degree", x_degree, r_degree, ijk, plot_flag, internal_metric_types, metric_dirs)
            process_metrics_from_manager(manager, internal_metric_types, curves_degree, collapse_degree, area_degree, attack_steps_degree, node_num)
            manager._run_specific_attack_multi_metric("random", x_random, r_random, ijk, plot_flag, internal_metric_types, metric_dirs)
            process_metrics_from_manager(manager, internal_metric_types, curves_random, collapse_random, area_random, attack_steps_random, node_num)
            for metric in internal_metric_types:
                methods_d = set(attack_steps_degree[metric].keys()) if attack_steps_degree[metric] else set()
                methods_r = set(attack_steps_random[metric].keys()) if attack_steps_random[metric] else set()
                for method in (methods_d | methods_r):
                    batch_d = attack_steps_degree[metric].get(method, [])
                    batch_r = attack_steps_random[metric].get(method, [])
                    (x_d, y_d, steps_d, r_d) = batch_d[-1] if batch_d else ([0, 1], [0, 0], [0, node_num], 0.0)
                    (x_r, y_r, steps_r, r_r) = batch_r[-1] if batch_r else ([0, 1], [0, 0], [0, node_num], 0.0)
                    x_w, y_w, steps_w = merge_curves_weighted(x_d, y_d, steps_d, x_r, y_r, steps_r, node_num)
                    r_w = 0.5 * r_d + 0.5 * r_r
                    # 崩溃点也保存 degree 与 random 的加权值：0.5*collapse_degree + 0.5*collapse_random
                    c_d = calculate_collapse_point(x_d, y_d)
                    c_r = calculate_collapse_point(x_r, y_r)
                    c_w = 0.5 * c_d + 0.5 * c_r
                    if method not in attack_steps_by_metric[metric]:
                        attack_steps_by_metric[metric][method] = []
                    attack_steps_by_metric[metric][method].append((x_w, y_w, steps_w, r_w))
                    if method not in r_values_by_metric[metric]:
                        r_values_by_metric[metric][method] = []
                    r_values_by_metric[metric][method].append(r_w)
                    if method not in area_values_by_metric[metric]:
                        area_values_by_metric[metric][method] = []
                    area_values_by_metric[metric][method].append(r_w)
                    if method not in collapse_points_by_metric[metric]:
                        collapse_points_by_metric[metric][method] = []
                    collapse_points_by_metric[metric][method].append(c_w)
        else:
            # 运行攻击仿真，为所有指标类型计算数据
            manager._run_specific_attack_multi_metric(
                current_attack_mode, 
                x_values_by_metric, 
                r_values_by_metric, 
                ijk, 
                plot_flag, 
                metric_types,
                metric_dirs
            )
            # 收集当前迭代的曲线数据、崩溃点数据和面积数据
            process_metrics_from_manager(
                manager, metric_types, curves_data_by_metric,
                collapse_points_by_metric, area_values_by_metric, attack_steps_by_metric,
                node_num
            )
        
        execution_times.setdefault("Simulation_Run", []).append(time.time() - sim_start_time)
        execution_times.setdefault("Total_Iteration", []).append(time.time() - iter_start_time)

        # 额外保存每次迭代的曲线数据（wgcc 模式不保存单次曲线，只保存聚合后的 wgcc 指标）
        if not is_wgcc_run:
            save_iteration_curves(manager, metric_types, ijk, current_attack_mode, extra_data_dir)
        
        # 收集本轮的综合指标
        if hasattr(manager, 'metrics_summary'):
            for name, metrics_list in manager.metrics_summary.items():
                if name not in global_metrics_summary:
                    global_metrics_summary[name] = []
                global_metrics_summary[name].extend(metrics_list)
        
        # 收集控制器部署时间
        if hasattr(manager, 'timing_summary'):
            for method_name, times in manager.timing_summary.items():
                key = f"Deploy_{method_name}"
                if key not in execution_times:
                    execution_times[key] = []
                execution_times[key].extend(times)

    save_comprehensive_metrics(global_metrics_summary, experiment_base_dir, current_attack_mode)
    save_execution_times(execution_times, common_dir)

    for metric_type in metric_types:
        subdirs = metric_subdirs[metric_type]

        if current_attack_mode == 'wgcc' and collapse_points_by_metric[metric_type]:
            collapse_points_to_save = {m: list(collapse_points_by_metric[metric_type][m]) for m in collapse_points_by_metric[metric_type] if collapse_points_by_metric[metric_type][m]}
            if collapse_points_to_save:
                save_collapse_point_data(
                    collapse_points_to_save,
                    subdirs['collapse_point'],
                    metric_type,
                    current_attack_mode
                )
                from src.utils.visualization.result_plotter import plot_collapse_point_charts
                plot_collapse_point_charts(
                    collapse_points_to_save,
                    subdirs['collapse_point'],
                    metric_type,
                    current_attack_mode
                )
        elif attack_steps_by_metric[metric_type]:
            collapse_points_from_area = {}
            for method, batch_data_list in attack_steps_by_metric[metric_type].items():
                collapse_points_from_area[method] = []
                for batch_data in batch_data_list:
                    if len(batch_data) >= 3:
                        x_list, y_list, steps_list = batch_data[0], batch_data[1], batch_data[2]
                        removed_ratio_list = [step / node_num for step in steps_list] if steps_list else x_list
                        if y_list and removed_ratio_list:
                            max_y = max(y_list)
                            threshold_y = max_y * 0.2
                            collapse_x = 1.0
                            for i, y_val in enumerate(y_list):
                                if y_val <= threshold_y:
                                    collapse_x = removed_ratio_list[i] if i < len(removed_ratio_list) else 1.0
                                    break
                            collapse_points_from_area[method].append(collapse_x)
            if collapse_points_from_area:
                save_collapse_point_data(
                    collapse_points_from_area,
                    subdirs['collapse_point'],
                    metric_type,
                    current_attack_mode
                )
                from src.utils.visualization.result_plotter import plot_collapse_point_charts
                plot_collapse_point_charts(
                    collapse_points_from_area,
                    subdirs['collapse_point'],
                    metric_type,
                    current_attack_mode
                )
        if area_values_by_metric[metric_type]:
            from src.utils.visualization.result_plotter import plot_collapse_r_charts
            # 直接使用area_values_by_metric中的R值
            r_values_for_collapse = {k: v for k, v in area_values_by_metric[metric_type].items() if v}
            if r_values_for_collapse:
                save_r_values_data(
                    r_values_for_collapse,
                    subdirs['collapse_R'],
                    metric_type,
                    current_attack_mode
                )
                plot_collapse_r_charts(
                    r_values_for_collapse,
                    subdirs['collapse_R'],
                    metric_type,
                    current_attack_mode
                )
        if area_values_by_metric[metric_type]:
            save_area_data(
                area_values_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                current_attack_mode
            )
        if attack_steps_by_metric[metric_type]:
            save_attack_steps_data(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                current_attack_mode
            )
            plot_attack_steps_for_batches(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                current_attack_mode,
                batch_size,
                dataset_name
            )
            plot_metric_vs_attack_steps(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                current_attack_mode,
                batch_size,
                dataset_name,
                node_num
            )
            from src.utils.visualization.result_plotter import plot_metric_vs_attack_steps_raw
            plot_metric_vs_attack_steps_raw(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                current_attack_mode,
                batch_size,
                dataset_name,
                node_num
            )
    return x_values_by_metric['gcc'], r_values_by_metric['gcc']

def main():
    args = parse_arguments()
    alltest_quiet = os.environ.get("ALLTEST_QUIET") == "1"
    log_level = logging.DEBUG if args.debug else (logging.WARNING if alltest_quiet else logging.WARNING)
    setup_logger(log_dir="logs", log_filename="latest_run.log", level=log_level, console_level=log_level)

    G, dataset_name = load_or_generate_graph(args)
    if G is None:
        logger.error("加载图数据失败，程序退出")
        return
    if not os.environ.get("ALLTEST_QUIET") == "1":
        logger.info(f"已加载数据集: {dataset_name} (节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()})")

    experiment_id = setup_experiment_dir(dataset_name, args.attack)
    _, _ = run_simulation_batch(G, args, experiment_id, dataset_name)
    if not os.environ.get("ALLTEST_QUIET") == "1":
        logger.info("实验完成")

if __name__ == "__main__":
    main()
