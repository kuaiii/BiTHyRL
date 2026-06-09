# -*- coding: utf-8 -*-
"""
基础拓扑消融实验

对比 BiT-HyRL 使用不同基础拓扑（Baseline、GA、ONION、ROMEN、UNITY）时的性能变化
用于消融实验，验证 Bimodal 拓扑的重要性
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import network_construction as nc  # 统一重构接口
from src.topology.reconstruction import create_bimodal_theoretical
from src.topology.generators import load_graph, construct_random, construct_ba
import networkx as nx
from src.controller.manager import ControllerManager
from src.utils.io import save_comprehensive_metrics, save_execution_times, save_collapse_point_data, save_area_data, save_attack_steps_data, save_r_values_data, save_decomposition_results
from src.utils.visualization.result_plotter import plot_collapse_point_charts, plot_attack_steps_for_batches, plot_metric_vs_attack_steps, plot_collapse_r_charts
from src.utils.visualization.result_plotter import plot_metric_vs_attack_steps_raw
import os
import random
import numpy as np
from math import ceil
from tqdm import tqdm
import time
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

# 设置基础随机种子
BASE_RANDOM_SEED = 42
random.seed(BASE_RANDOM_SEED)
np.random.seed(BASE_RANDOM_SEED)


def load_or_generate_graph(dataset_name):
    """
    从 GML 文件加载图。
    训练用的数据集在 dataset/all 中；测试的数据集在 dataset/testdata 中。
    """
    gml_path = os.path.join('dataset', 'testdata', f'{dataset_name}.gml')
    if not os.path.exists(gml_path):
        gml_path = os.path.join('dataset', 'all', f'{dataset_name}.gml')
    
    if os.path.exists(gml_path):
        G, _ = load_graph(gml_path)
    else:
        # 如果文件不存在，生成一个 BA 网络
        logger.warning(f"未找到 {gml_path}，生成 BA 网络")
        n = int(dataset_name.split('-')[-1]) if '-' in dataset_name else 100
        G = nx.barabasi_albert_graph(n, 4)
    
    return G


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


def _nm_compute('r_value_interpolated', x_curve, y_curve, num_points=101):
    """计算R值（曲线下面积）"""
    from network_metrics import compute as _nm_compute as calc_r
    return calc_r(x_curve, y_curve, num_points)


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
        logger.warning(f"manager没有metrics_summary属性")
        return
    
    if not manager.metrics_summary:
        logger.warning(f"manager.metrics_summary为空")
        return
    
    for name, metrics_list in manager.metrics_summary.items():
        if not metrics_list:
            logger.warning(f"方法 {name} 的metrics_list为空")
            continue
        
        current_metric = metrics_list[-1]
        
        for metric_type in metric_types:
            curve_key = f'{metric_type.upper()}_Curve'
            attack_steps_key = f'{metric_type.upper()}_AttackSteps'
            curve_data = current_metric.get(curve_key, [])
            attack_steps_data = current_metric.get(attack_steps_key, [])
            
            if not curve_data:
                logger.debug(f"方法 {name} 的 {metric_type} 曲线数据为空")
                continue
            
            x_curve, y_curve, steps_curve = extract_curve_data(curve_data, attack_steps_data)
            
            if not x_curve or not y_curve:
                logger.warning(f"方法 {name} 的 {metric_type} 提取后的曲线数据为空")
                continue
            
            # 计算removed_nodes_ratio = attack_step/initial_nodes
            removed_ratio_list = [step / initial_nodes for step in steps_curve] if steps_curve else x_curve
            
            # 计算R值（面积）：统一使用removed_nodes_ratio
            r_val = _nm_compute('r_value_interpolated', removed_ratio_list, y_curve, num_points=101)
            
            # 保存曲线数据（虽然不绘制，但保留数据结构）
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
            
            logger.debug(f"成功处理 {name} 的 {metric_type} 数据: R={r_val:.4f}, 数据点={len(x_curve)}")


def compare_basic_topology(dataset_name="BA-200", attack_mode="degree", cover_rate=0.1, 
                          batch_size=5, experiment_id=None, 
                          bit_hyrl_episodes=100, bit_hyrl_metric_type='robustness',
                          bit_hyrl_use_node2vec=True, bit_hyrl_use_stepwise=True, 
                          bit_hyrl_use_ci=True, bit_hyrl_test_mode=True, bit_hyrl_use_gnn=True):
    """
    基础拓扑消融实验：对比 BiT-HyRL 使用不同基础拓扑时的性能
    
    注意：此函数默认只做推理（bit_hyrl_test_mode=True），不进行训练。
          如果模型不存在，会自动降级为CI算法。
    """
    """
    基础拓扑消融实验：对比 BiT-HyRL 使用不同基础拓扑时的性能
    
    Args:
        dataset_name: 数据集名称
        attack_mode: 攻击模式 ('degree', 'random', 等)
        cover_rate: 控制器部署比例
        batch_size: 批次数
        experiment_id: 实验ID（如果为None，自动生成）
        bit_hyrl_*: BiT-HyRL相关参数
    """
    import networkx as nx
    
    print("\n" + "="*70)
    print(f"基础拓扑消融实验: BiT-HyRL with Different Base Topologies")
    print(f"数据集: {dataset_name}, 攻击模式: {attack_mode}")
    print("="*70)
    
    # 加载图
    logger.info(f"加载图: {dataset_name}")
    G = load_or_generate_graph(dataset_name)
    logger.info(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()
    
    # 设置实验ID
    if experiment_id is None:
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
    
    # 创建实验目录
    experiment_base_dir = os.path.join('results', dataset_name, attack_mode, f'ablation_{experiment_id}')
    if not os.path.exists(experiment_base_dir):
        os.makedirs(experiment_base_dir)
    
    # 定义所有指标类型
    metric_types = ['gcc', 'csa', 'cce', 'wcp']
    
    # 为每个指标类型创建子目录
    metric_dirs = {}
    metric_subdirs = {}
    for metric_type in metric_types:
        metric_dir = os.path.join(experiment_base_dir, metric_type)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
        metric_dirs[metric_type] = metric_dir
        
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
    
    # 定义基础拓扑类型
    base_topology_types = {
        'Baseline': None,  # 使用原始图G
        'GA': 'GA',
        'ONION': 'ONION',
        'ROMEN': 'ROMEN',
        'UNITY': 'UNITY',
        'FRED-ABL': 'FRED-ABL',
        'TEAM': 'TEAM',
        'QDLM': 'QDLM',
        'Bimodal': 'Bimodal'  # 作为对比基准
    }
    
    # 初始化数据存储
    x_values_by_metric = {}
    r_values_by_metric = {}
    curves_data_by_metric = {}
    collapse_points_by_metric = {}
    area_values_by_metric = {}
    attack_steps_by_metric = {}
    
    for metric_type in metric_types:
        x_values_by_metric[metric_type] = {}
        r_values_by_metric[metric_type] = {}
        curves_data_by_metric[metric_type] = {}
        collapse_points_by_metric[metric_type] = {}
        area_values_by_metric[metric_type] = {}
        attack_steps_by_metric[metric_type] = {}
        
        for base_type in base_topology_types.keys():
            x_values_by_metric[metric_type][f'BiT-HyRL-{base_type}'] = []
            r_values_by_metric[metric_type][f'BiT-HyRL-{base_type}'] = []
            curves_data_by_metric[metric_type][f'BiT-HyRL-{base_type}'] = []
            collapse_points_by_metric[metric_type][f'BiT-HyRL-{base_type}'] = []
            area_values_by_metric[metric_type][f'BiT-HyRL-{base_type}'] = []
            attack_steps_by_metric[metric_type][f'BiT-HyRL-{base_type}'] = []
    
    execution_times = {}
    
    # 运行batch_size次实验
    for batch_idx in tqdm(range(batch_size), desc=f"消融实验 {dataset_name}"):
        batch_seed = BASE_RANDOM_SEED + batch_idx
        random.seed(batch_seed)
        np.random.seed(batch_seed)
        
        # 构建所有基础拓扑（每个batch构建一次）
        logger.info(f"构建基础拓扑 (批次 {batch_idx + 1})...")
        
        start_time = time.time()
        G_GA = nc.construct(G, algorithm='GA')
        execution_times.setdefault("Construct_GA", []).append(time.time() - start_time)
        
        start_time = time.time()
        G_ONION = nc.construct(G, algorithm='Onion', max_iter=300)
        execution_times.setdefault("Construct_Onion", []).append(time.time() - start_time)
        
        start_time = time.time()
        G_ROMEN = nc.construct(G, algorithm='ROMEN', seed=batch_seed)
        execution_times.setdefault("Construct_ROMEN", []).append(time.time() - start_time)
        
        start_time = time.time()
        G_UNITY = nc.construct(G, algorithm='UNITY')
        execution_times.setdefault("Construct_UNITY", []).append(time.time() - start_time)
        
        start_time = time.time()
        G_Bimodal = create_bimodal_theoretical(node_num, edge_num, seed=batch_seed)
        execution_times.setdefault("Construct_Bimodal", []).append(time.time() - start_time)
        
        # 构建三种新算法
        start_time = time.time()
        try:
            G_FRED_ABL = nc.construct(G, algorithm='FRED_ABL', seed=batch_seed, iterations=15, initial_samples=3, verbose=False)
            execution_times.setdefault("Construct_FRED_ABL", []).append(time.time() - start_time)
        except Exception as e:
            logger.warning(f"FRED-ABL构建失败: {e}")
            G_FRED_ABL = None
            execution_times.setdefault("Construct_FRED_ABL", []).append(time.time() - start_time)
        
        start_time = time.time()
        try:
            G_TEAM = nc.construct(G, algorithm='TEAM', seed=batch_seed, verbose=False)
            execution_times.setdefault("Construct_TEAM", []).append(time.time() - start_time)
        except Exception as e:
            logger.warning(f"TEAM构建失败: {e}")
            G_TEAM = None
            execution_times.setdefault("Construct_TEAM", []).append(time.time() - start_time)
        
        start_time = time.time()
        try:
            G_QDLM = nc.construct(G, algorithm='QDLM', seed=batch_seed, pop_size=20, max_iterations=25, verbose=False)
            execution_times.setdefault("Construct_QDLM", []).append(time.time() - start_time)
        except Exception as e:
            logger.warning(f"QDLM构建失败: {e}")
            G_QDLM = None
            execution_times.setdefault("Construct_QDLM", []).append(time.time() - start_time)
        
        # 构建 G_RA 和 G_BA（仅用于保持 all_G 字典的完整性，BiT-HyRL 不使用它们）
        start_time = time.time()
        G_RA = construct_random(node_num, edge_num)
        execution_times.setdefault("Construct_Random", []).append(time.time() - start_time)
        
        start_time = time.time()
        G_BA = construct_ba(node_num, edge_num)
        execution_times.setdefault("Construct_BA", []).append(time.time() - start_time)
        
        # 为每个基础拓扑类型运行 BiT-HyRL
        for base_type, topology_key in base_topology_types.items():
            # 选择基础拓扑
            if base_type == 'Baseline':
                base_topology = G
            elif base_type == 'GA':
                base_topology = G_GA
            elif base_type == 'ONION':
                base_topology = G_ONION
            elif base_type == 'ROMEN':
                base_topology = G_ROMEN
            elif base_type == 'UNITY':
                base_topology = G_UNITY
            elif base_type == 'FRED-ABL':
                base_topology = G_FRED_ABL
            elif base_type == 'TEAM':
                base_topology = G_TEAM
            elif base_type == 'QDLM':
                base_topology = G_QDLM
            elif base_type == 'Bimodal':
                base_topology = G_Bimodal
            else:
                continue
            
            if base_topology is None:
                logger.warning(f"跳过 {base_type}（拓扑为空）")
                continue
            
            logger.info(f"处理基础拓扑: {base_type} (批次 {batch_idx + 1}/{batch_size})")
            
            # 构建 all_G 字典（为了保持一致性，包含所有拓扑）
            all_G = {
                'G': G,
                'G_BiT': base_topology,  # 使用选定的基础拓扑替代 Bimodal
                'G_RA': G_RA,  # 添加以保持一致性（BiT-HyRL 不使用）
                'G_BA': G_BA,  # 添加以保持一致性（BiT-HyRL 不使用）
                'G_GA': G_GA,
                'G_ONION': G_ONION,
                'G_ROMEM': G_ROMEN,
                'G_UNITY': G_UNITY,
                'G_FRED_ABL': G_FRED_ABL,
                'G_TEAM': G_TEAM,
                'G_QDLM': G_QDLM,
            }
            
            # 创建 ControllerManager（仅用于获取配置，不运行所有方法）
            manager = ControllerManager(
                all_G, cover_rate, node_num, dataset_name, experiment_id,
                bit_hyrl_episodes=bit_hyrl_episodes,
                bit_hyrl_metric_type=bit_hyrl_metric_type,
                bit_hyrl_use_node2vec=bit_hyrl_use_node2vec,
                bit_hyrl_use_stepwise=bit_hyrl_use_stepwise,
                bit_hyrl_use_ci=bit_hyrl_use_ci,
                bit_hyrl_test_mode=bit_hyrl_test_mode,
                bit_hyrl_use_gnn=bit_hyrl_use_gnn
            )
            
            # 直接调用 BiT-HyRL 的选择函数，使用自定义方法名
            method_name = f'BiT-HyRL-{base_type}'
            
            # 选择控制器（只做推理，不训练）
            from src.bit_hyrl.selection import hybrid_gnn_select
            from src.bit_hyrl.config import MODEL_DIR
            
            # 尝试加载预训练模型（只做推理）
            gnn_model_path = os.path.join(MODEL_DIR, 'gnn_ppo_agent_survival.pth')
            if not os.path.exists(gnn_model_path):
                gnn_model_path = os.path.join(MODEL_DIR, 'gnn_ppo_agent.pth')
            if not os.path.exists(gnn_model_path):
                gnn_model_path = None  # 如果模型不存在，使用CI算法
            
            deployment_start = time.time()
            rl_centers = hybrid_gnn_select(
                base_topology, manager.controller_num,
                model_path=gnn_model_path,  # 使用预训练模型路径，只做推理
                use_ci=bit_hyrl_use_ci
            )
            deployment_time = time.time() - deployment_start
            
            # 记录时间（如果需要）
            if not hasattr(manager, 'timing_summary'):
                manager.timing_summary = {}
            if method_name not in manager.timing_summary:
                manager.timing_summary[method_name] = []
            manager.timing_summary[method_name].append(deployment_time)
            
            # 直接处理该方法的数据（使用自定义方法名）
            try:
                manager._process_method_multi_metric(
                    base_topology, rl_centers, attack_mode, method_name,
                    x_values_by_metric, r_values_by_metric
                )
                
                # 检查manager是否有metrics_summary
                if not hasattr(manager, 'metrics_summary') or method_name not in manager.metrics_summary:
                    logger.error(f"方法 {method_name} 的数据未正确生成到 metrics_summary")
                    continue
                
                if not manager.metrics_summary[method_name]:
                    logger.error(f"方法 {method_name} 的 metrics_summary 为空")
                    continue
                
                # 处理指标数据
                process_metrics_from_manager(
                    manager, metric_types, curves_data_by_metric,
                    collapse_points_by_metric, area_values_by_metric, attack_steps_by_metric,
                    node_num
                )
                
                logger.info(f"成功处理 {method_name} 的数据")
            except Exception as e:
                logger.error(f"处理 {method_name} 时出错: {e}", exc_info=True)
                continue
    
    # 保存运行时间
    if execution_times:
        save_execution_times(execution_times, experiment_base_dir)
    
    # 调试：打印数据统计
    logger.info("\n数据统计:")
    for metric_type in metric_types:
        logger.info(f"  {metric_type}:")
        logger.info(f"    attack_steps_by_metric: {len(attack_steps_by_metric[metric_type])} 个方法")
        logger.info(f"    area_values_by_metric: {len(area_values_by_metric[metric_type])} 个方法")
        for method_name, data_list in attack_steps_by_metric[metric_type].items():
            logger.info(f"      {method_name}: {len(data_list)} 个batch")
        for method_name, data_list in area_values_by_metric[metric_type].items():
            logger.info(f"      {method_name}: {len(data_list)} 个batch")
    
    # 为每个指标类型保存结果并绘制图表
    for metric_type in metric_types:
        subdirs = metric_subdirs[metric_type]
        
        # 从area数据计算并保存崩溃点数据
        if attack_steps_by_metric[metric_type] and len(attack_steps_by_metric[metric_type]) > 0:
            collapse_points_from_area = {}
            for method, batch_data_list in attack_steps_by_metric[metric_type].items():
                if not batch_data_list:
                    logger.warning(f"方法 {method} 的 {metric_type} batch_data_list 为空")
                    continue
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
                logger.info(f"保存 {metric_type} 的崩溃点数据: {len(collapse_points_from_area)} 个方法")
                save_collapse_point_data(
                    collapse_points_from_area,
                    subdirs['collapse_point'],
                    metric_type,
                    attack_mode
                )
                
                plot_collapse_point_charts(
                    collapse_points_from_area,
                    subdirs['collapse_point'],
                    metric_type,
                    attack_mode
                )
            else:
                logger.warning(f"{metric_type} 的崩溃点数据为空，跳过保存")
        else:
            logger.warning(f"{metric_type} 的 attack_steps_by_metric 为空，跳过崩溃点计算")
        
        # 保存R值数据并绘制图表
        if area_values_by_metric[metric_type] and len(area_values_by_metric[metric_type]) > 0:
            r_values_for_collapse = {k: v for k, v in area_values_by_metric[metric_type].items() if v and len(v) > 0}
            
            if r_values_for_collapse:
                logger.info(f"保存 {metric_type} 的 R值数据: {len(r_values_for_collapse)} 个方法")
                save_r_values_data(
                    r_values_for_collapse,
                    subdirs['collapse_R'],
                    metric_type,
                    attack_mode
                )
                
                plot_collapse_r_charts(
                    r_values_for_collapse,
                    subdirs['collapse_R'],
                    metric_type,
                    attack_mode
                )
            else:
                logger.warning(f"{metric_type} 的 R值数据为空，跳过保存")
        else:
            logger.warning(f"{metric_type} 的 area_values_by_metric 为空，跳过保存")
        
        # 保存面积数据
        if area_values_by_metric[metric_type] and len(area_values_by_metric[metric_type]) > 0:
            logger.info(f"保存 {metric_type} 的面积数据: {len(area_values_by_metric[metric_type])} 个方法")
            save_area_data(
                area_values_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                attack_mode
            )
        else:
            logger.warning(f"{metric_type} 的 area_values_by_metric 为空，跳过保存")
        
        # 保存攻击步数数据并绘制图表
        if attack_steps_by_metric[metric_type] and len(attack_steps_by_metric[metric_type]) > 0:
            logger.info(f"保存 {metric_type} 的攻击步数数据: {len(attack_steps_by_metric[metric_type])} 个方法")
            
            # 保存攻击步数数据（汇总统计）
            save_attack_steps_data(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                attack_mode
            )
            
            # 保存分解结果（详细的攻击过程数据）
            save_decomposition_results(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                attack_mode,
                node_num
            )
            
            # 绘制折线图
            plot_attack_steps_for_batches(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                attack_mode,
                batch_size,
                dataset_name
            )
            
            plot_metric_vs_attack_steps(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                attack_mode,
                batch_size,
                dataset_name,
                node_num
            )
            
            plot_metric_vs_attack_steps_raw(
                attack_steps_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                attack_mode,
                batch_size,
                dataset_name,
                node_num
            )
        else:
            logger.warning(f"{metric_type} 的 attack_steps_by_metric 为空，跳过保存和绘图")
    
    # 打印总结
    print("\n" + "="*70)
    print("消融实验完成!")
    print("="*70)
    print(f"结果保存在: {experiment_base_dir}")
    print("\n对比方法:")
    for base_type in base_topology_types.keys():
        print(f"  - BiT-HyRL-{base_type}")
    print("\n保存的内容:")
    print("  - 分解结果CSV文件（每种算法的攻击过程详细数据）")
    print("  - R值数据（面积数据）")
    print("  - 折线图（指标值 vs 移除节点比例）")
    print("  - 崩溃点数据")
    print("="*70 + "\n")
    
    return experiment_base_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='基础拓扑消融实验')
    parser.add_argument('--dataset', type=str, default='BA-200', help='数据集名称')
    parser.add_argument('--attack', type=str, default='degree', help='攻击模式')
    parser.add_argument('--cover_rate', type=float, default=0.1, help='控制器部署比例')
    parser.add_argument('--batch_size', type=int, default=5, help='批次数')
    parser.add_argument('--experiment_id', type=int, default=None, help='实验ID（可选）')
    parser.add_argument('--episodes', type=int, default=100, help='BiT-HyRL训练轮数')
    parser.add_argument('--metric_type', type=str, default='robustness', help='BiT-HyRL优化目标类型')
    parser.add_argument('--use_node2vec', action='store_true', default=True, help='使用Node2Vec特征')
    parser.add_argument('--no_node2vec', dest='use_node2vec', action='store_false')
    parser.add_argument('--use_stepwise', action='store_true', default=True, help='使用Step-wise Reward')
    parser.add_argument('--no_stepwise', dest='use_stepwise', action='store_false')
    parser.add_argument('--use_ci', action='store_true', default=True, help='使用CI算法')
    parser.add_argument('--no_ci', dest='use_ci', action='store_false')
    parser.add_argument('--test_mode', action='store_true', default=True, help='测试模式（仅推理，不训练）')
    parser.add_argument('--use_gnn', action='store_true', default=True, help='使用GNN模型（推理模式）')
    parser.add_argument('--no_test_mode', dest='test_mode', action='store_false', help='禁用测试模式（允许训练）')
    parser.add_argument('--no_gnn', dest='use_gnn', action='store_false', help='不使用GNN模型')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger(log_dir="logs", log_filename="ablation_basic_topology.log")
    
    compare_basic_topology(
        dataset_name=args.dataset,
        attack_mode=args.attack,
        cover_rate=args.cover_rate,
        batch_size=args.batch_size,
        experiment_id=args.experiment_id,
        bit_hyrl_episodes=args.episodes,
        bit_hyrl_metric_type=args.metric_type,
        bit_hyrl_use_node2vec=args.use_node2vec,
        bit_hyrl_use_stepwise=args.use_stepwise,
        bit_hyrl_use_ci=args.use_ci,
        bit_hyrl_test_mode=args.test_mode,
        bit_hyrl_use_gnn=args.use_gnn
    )
