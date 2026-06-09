# -*- coding: utf-8 -*-
import numpy as np
from math import ceil
from src.controller.strategies import random_select_controllers, RCP
# 导入旧的RL优化器用于其他方法（GA+RL, Onion+RL等）
from src.controller.rl_optimizer import train_and_select
from src.bit_hyrl.training import train_and_select as bit_hyrl_train_and_select
from src.simulation.dismantling import node_attack
from src.utils.logger import get_logger
from src.bit_hyrl.selection import hybrid_gnn_select, gnn_predict
from network_metrics import compute as _nm_compute
logger = get_logger(__name__)

class ControllerManager:
    def __init__(self, all_G, rate, nodes_num, dataset_name, experiment_id, 
                 bit_hyrl_episodes=100, bit_hyrl_metric_type='robustness',
                 bit_hyrl_use_node2vec=True, bit_hyrl_use_stepwise=True, bit_hyrl_use_ci=True,
                 bit_hyrl_test_mode=False, bit_hyrl_use_gnn=True, bit_hyrl_gnn_model=None):
        """
        初始化控制器管理器，用于管理不同拓扑结构和控制器部署策略的仿真实验。

        Args:
            all_G (dict): 包含所有待评估拓扑的字典，键为拓扑名称（如 'G', 'G1'），值为 networkx 图对象。
            rate (float): 控制器部署比例（例如 0.1 表示 10% 的节点作为控制器）。
            nodes_num (int): 网络中的总节点数。
            dataset_name (str): 数据集名称，用于结果保存路径。
            experiment_id (int): 实验 ID，用于结果保存路径的区分。
            bit_hyrl_episodes (int): [已弃用] BiT-HyRL训练轮数，默认100
            bit_hyrl_metric_type (str): [已弃用] BiT-HyRL优化目标类型，默认'robustness'
            bit_hyrl_use_node2vec (bool): [已弃用] BiT-HyRL是否使用Node2Vec特征，默认True
            bit_hyrl_use_stepwise (bool): [已弃用] BiT-HyRL是否使用Step-wise Reward，默认True
            bit_hyrl_use_ci (bool): BiT-HyRL是否使用CI算法（作为降级方案），默认True
            bit_hyrl_test_mode (bool): [已弃用] BiT-HyRL测试模式，默认False
            bit_hyrl_use_gnn (bool): BiT-HyRL是否使用GNN模型，默认True
            bit_hyrl_gnn_model (str): GNN模型路径，None时自动搜索最佳模型。
                优先级: curriculum_final > curriculum_phase3 > deep_gat > ...
        """
        self.all_G = all_G
        self.G = all_G.get('G')
        self.G_BiT = all_G.get('G_BiT')
        self.G_GA = all_G.get('G_GA')
        self.G_ONION = all_G.get('G_ONION')
        self.G_ROMEM = all_G.get('G_ROMEM')
        self.G_UNITY = all_G.get('G_UNITY')
        self.G_FRED_ABL = all_G.get('G_FRED_ABL')
        self.G_TEAM = all_G.get('G_TEAM')
        self.G_QDLM = all_G.get('G_QDLM')
        
        self.rate = rate
        self.nodes_num = nodes_num
        self.dataset_name = dataset_name
        self.experiment_id = experiment_id
        
        self.bit_hyrl_episodes = bit_hyrl_episodes
        self.bit_hyrl_metric_type = bit_hyrl_metric_type
        self.bit_hyrl_use_node2vec = bit_hyrl_use_node2vec
        self.bit_hyrl_use_stepwise = bit_hyrl_use_stepwise
        self.bit_hyrl_use_ci = bit_hyrl_use_ci
        self.bit_hyrl_test_mode = bit_hyrl_test_mode
        self.bit_hyrl_use_gnn = bit_hyrl_use_gnn
        self.bit_hyrl_gnn_model = bit_hyrl_gnn_model
        
        self.controller_num = ceil(nodes_num * rate)
        
        self.colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD', '#8C8C8C', '#E377C2', '#BCBD22', '#17BECF', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
        self.linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']
        self.markers = ['o', 's', '^', 'D', 'v', '+', '*', 'x', 'p', 'h', '1', '2', '3', '4', '8']

    def _process_method_multi_metric(self, G_target, centers, attack_mode, name, x_values_by_metric, r_values_by_metric):
        """
        处理单个部署策略的评估（多指标版本）：执行攻击仿真、计算所有指标并记录数据。
        
        Args:
            G_target: 目标拓扑
            centers: 控制器集合
            attack_mode: 攻击模式
            name: 方法名称
            x_values_by_metric: 按指标类型组织的崩溃点数据字典 {metric_type: {method: [values]}}
            r_values_by_metric: 按指标类型组织的鲁棒性数据字典 {metric_type: {method: [values]}}
        """
        if not G_target:
            return
        
        if not hasattr(self, 'metrics_summary'):
            self.metrics_summary = {}
            
        if name not in self.metrics_summary:
            self.metrics_summary[name] = []
            
        num_nodes = G_target.number_of_nodes()
        num_edges = G_target.number_of_edges()
        
        degrees = sorted([d for n, d in G_target.degree()], reverse=True)
        top_10_degrees = degrees[:10] if len(degrees) >= 10 else degrees
        top_10_str = str(top_10_degrees).replace(',', ';')
            
        metrics_dict, x_gcc_20, attack_steps_dict = node_attack(G_target.copy(), self.nodes_num, set(centers), attack_mode)
        
        metric_collapse_points = {}
        metric_r_values = {}
        
        for metric_type in ['gcc', 'csa', 'cce', 'wcp']:
            curve_key = metric_type
            curve_data = metrics_dict.get(curve_key, [])
            
            if not curve_data:
                metric_collapse_points[metric_type] = None
                metric_r_values[metric_type] = 0.0
                continue
            
            x_curve = [point[1] for point in curve_data]
            y_curve = [point[0] for point in curve_data]
            
            r_val = _nm_compute('r_value_interpolated', x_curve, y_curve, num_points=101)
            metric_r_values[metric_type] = r_val
            
            if len(y_curve) > 0:
                initial_value = y_curve[0] if y_curve[0] > 0 else 1.0
                threshold = initial_value * 0.2
                
                collapse_point = None
                for i, y_val in enumerate(y_curve):
                    if y_val <= threshold:
                        collapse_point = x_curve[i]
                        break
                
                if collapse_point is None and len(x_curve) > 0:
                    collapse_point = x_curve[-1]
                
                metric_collapse_points[metric_type] = collapse_point
            else:
                metric_collapse_points[metric_type] = None
        
        self.metrics_summary[name].append({
            'GCC_Curve': metrics_dict['gcc'],
            'CSA_Curve': metrics_dict['csa'],
            'CCE_Curve': metrics_dict['cce'],
            'WCP_Curve': metrics_dict['wcp'],
            'GCC_AttackSteps': attack_steps_dict['gcc'],
            'CSA_AttackSteps': attack_steps_dict['csa'],
            'CCE_AttackSteps': attack_steps_dict['cce'],
            'WCP_AttackSteps': attack_steps_dict['wcp'],
            'R': metric_r_values['gcc'],
            'Collapse_Point': x_gcc_20,
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Top10_Degrees': top_10_str
        })
        
        for metric_type in ['gcc', 'csa', 'cce', 'wcp']:
            if name not in x_values_by_metric[metric_type]:
                x_values_by_metric[metric_type][name] = []
            if name not in r_values_by_metric[metric_type]:
                r_values_by_metric[metric_type][name] = []
            
            collapse_point = metric_collapse_points[metric_type]
            r_val = metric_r_values[metric_type]
            
            if collapse_point is not None:
                x_values_by_metric[metric_type][name].append(collapse_point)
            else:
                x_values_by_metric[metric_type][name].append(1.0)
            
            r_values_by_metric[metric_type][name].append(r_val)

    def _run_specific_attack_multi_metric(self, attack_mode_str, x_values_by_metric, r_values_by_metric, ijk, plot_flag, metric_types, metric_dirs):
        """
        执行特定攻击模式下的所有对比方案仿真（多指标版本）。
        
        Args:
            attack_mode_str (str): 攻击模式名称。
            x_values_by_metric (dict): 按指标类型组织的崩溃点数据 {metric_type: {method: [values]}}
            r_values_by_metric (dict): 按指标类型组织的鲁棒性数据 {metric_type: {method: [values]}}
            ijk (int): 当前迭代次数。
            plot_flag (bool): 是否绘图。
            metric_types (list): 指标类型列表。
            metric_dirs (dict): 按指标类型组织的保存目录 {metric_type: save_dir}
        """
        import time
        attack_code = attack_mode_str
        
        if not hasattr(self, 'timing_summary'):
            self.timing_summary = {}
        
        # 用于收集控制器信息
        controller_info = []
        
        # 1. Baseline: 原始拓扑 + 随机控制器（使用ijk作为seed确保不同batch结果不同）
        deployment_start = time.time()
        # 使用ijk计算seed，确保不同batch结果不同
        batch_seed = 42 + ijk  # BASE_RANDOM_SEED = 42
        ori_random_centers = random_select_controllers(self.G, self.rate, seed=batch_seed)
        deployment_time = time.time() - deployment_start
        if 'Baseline' not in self.timing_summary:
            self.timing_summary['Baseline'] = []
        self.timing_summary['Baseline'].append(deployment_time)
        self._process_method_multi_metric(self.G, ori_random_centers, attack_code, "Baseline", x_values_by_metric, r_values_by_metric)
        controller_info.append(("Baseline", self.G, len(ori_random_centers)))
        
        # 2. GA+RCP: 原始拓扑 + E-RCP
        deployment_start = time.time()
        ori_ECA_centers = RCP(self.G, self.rate, 1)
        deployment_time = time.time() - deployment_start
        if 'GA+RCP' not in self.timing_summary:
            self.timing_summary['GA+RCP'] = []
        self.timing_summary['GA+RCP'].append(deployment_time)
        self._process_method_multi_metric(self.G, ori_ECA_centers, attack_code, "GA+RCP", x_values_by_metric, r_values_by_metric)
        controller_info.append(("GA+RCP", self.G, len(ori_ECA_centers)))
        
        # 3. BiT-HyRL: Bimodal + GNN 选择（模型路径 None 时自动搜索最佳模型）
        deployment_start = time.time()
        rl_centers = hybrid_gnn_select(
            self.G_BiT, self.controller_num,
            model_path=self.bit_hyrl_gnn_model,  # 支持指定模型路径
            use_ci=self.bit_hyrl_use_ci
        )
        deployment_time = time.time() - deployment_start
        if 'BiT-HyRL' not in self.timing_summary:
            self.timing_summary['BiT-HyRL'] = []
        self.timing_summary['BiT-HyRL'].append(deployment_time)
        self._process_method_multi_metric(self.G_BiT, rl_centers, attack_code, "BiT-HyRL", x_values_by_metric, r_values_by_metric)
        controller_info.append(("BiT-HyRL", self.G_BiT, len(rl_centers) if rl_centers else 0))
        
        # 4. GA+RL: GA优化拓扑 + RL选择
        deployment_start = time.time()
        ran_rl_centers = train_and_select(self.G_GA, self.controller_num, metric_type="robustness")
        deployment_time = time.time() - deployment_start
        if 'GA+RL' not in self.timing_summary:
            self.timing_summary['GA+RL'] = []
        self.timing_summary['GA+RL'].append(deployment_time)
        self._process_method_multi_metric(self.G_GA, ran_rl_centers, attack_code, "GA+RL", x_values_by_metric, r_values_by_metric)
        controller_info.append(("GA+RL", self.G_GA, len(ran_rl_centers) if ran_rl_centers else 0))
        
        # 5. Onion+RL: Onion优化拓扑 + RL选择
        deployment_start = time.time()
        onion_rl_centers = train_and_select(self.G_ONION, self.controller_num, metric_type="robustness")
        deployment_time = time.time() - deployment_start
        if 'Onion+RL' not in self.timing_summary:
            self.timing_summary['Onion+RL'] = []
        self.timing_summary['Onion+RL'].append(deployment_time)
        self._process_method_multi_metric(self.G_ONION, onion_rl_centers, attack_code, "Onion+RL", x_values_by_metric, r_values_by_metric)
        controller_info.append(("Onion+RL", self.G_ONION, len(onion_rl_centers) if onion_rl_centers else 0))
 
        # 6. ROMEN+RL: ROMEN优化拓扑 + RL选择
        deployment_start = time.time()
        romen_rl_centers = train_and_select(self.G_ROMEM, self.controller_num, metric_type="robustness")
        deployment_time = time.time() - deployment_start
        if 'ROMEN+RL' not in self.timing_summary:
            self.timing_summary['ROMEN+RL'] = []
        self.timing_summary['ROMEN+RL'].append(deployment_time)
        self._process_method_multi_metric(self.G_ROMEM, romen_rl_centers, attack_code, "ROMEN+RL", x_values_by_metric, r_values_by_metric)
        controller_info.append(("ROMEN+RL", self.G_ROMEM, len(romen_rl_centers) if romen_rl_centers else 0))

        # 7. UNITY+RL: UNITY优化拓扑 + RL选择
        deployment_start = time.time()
        unity_rl_centers = train_and_select(self.G_UNITY, self.controller_num, metric_type="robustness")
        deployment_time = time.time() - deployment_start
        if 'UNITY+RL' not in self.timing_summary:
            self.timing_summary['UNITY+RL'] = []
        self.timing_summary['UNITY+RL'].append(deployment_time)
        self._process_method_multi_metric(self.G_UNITY, unity_rl_centers, attack_code, "UNITY+RL", x_values_by_metric, r_values_by_metric)
        controller_info.append(("UNITY+RL", self.G_UNITY, len(unity_rl_centers) if unity_rl_centers else 0))
        
        # 8. FRED-ABL: FRED-ABL优化拓扑 + RL选择
        if self.G_FRED_ABL is not None:
            deployment_start = time.time()
            fred_abl_rl_centers = train_and_select(self.G_FRED_ABL, self.controller_num, metric_type="robustness")
            deployment_time = time.time() - deployment_start
            if 'FRED-ABL' not in self.timing_summary:
                self.timing_summary['FRED-ABL'] = []
            self.timing_summary['FRED-ABL'].append(deployment_time)
            self._process_method_multi_metric(self.G_FRED_ABL, fred_abl_rl_centers, attack_code, "FRED-ABL", x_values_by_metric, r_values_by_metric)
            controller_info.append(("FRED-ABL", self.G_FRED_ABL, len(fred_abl_rl_centers) if fred_abl_rl_centers else 0))
        
        # 9. TEAM: TEAM优化拓扑 + RL选择
        if self.G_TEAM is not None:
            deployment_start = time.time()
            team_rl_centers = train_and_select(self.G_TEAM, self.controller_num, metric_type="robustness")
            deployment_time = time.time() - deployment_start
            if 'TEAM' not in self.timing_summary:
                self.timing_summary['TEAM'] = []
            self.timing_summary['TEAM'].append(deployment_time)
            self._process_method_multi_metric(self.G_TEAM, team_rl_centers, attack_code, "TEAM", x_values_by_metric, r_values_by_metric)
            controller_info.append(("TEAM", self.G_TEAM, len(team_rl_centers) if team_rl_centers else 0))
        
        # 10. QDLM: QDLM优化拓扑 + RL选择
        if self.G_QDLM is not None:
            deployment_start = time.time()
            qdlm_rl_centers = train_and_select(self.G_QDLM, self.controller_num, metric_type="robustness")
            deployment_time = time.time() - deployment_start
            if 'QDLM' not in self.timing_summary:
                self.timing_summary['QDLM'] = []
            self.timing_summary['QDLM'].append(deployment_time)
            self._process_method_multi_metric(self.G_QDLM, qdlm_rl_centers, attack_code, "QDLM", x_values_by_metric, r_values_by_metric)
            controller_info.append(("QDLM", self.G_QDLM, len(qdlm_rl_centers) if qdlm_rl_centers else 0))
        
        # 输出控制器选择信息（仅第一个batch）
        if ijk == 0:
            print("\n" + "-" * 80)
            print("控制器选择结果")
            print("-" * 80)
            print(f"{'方法':<15} {'拓扑节点数':<12} {'拓扑边数':<12} {'控制器数':<10} {'控制器比例':<12}")
            print("-" * 80)
            for method_name, g, num_controllers in controller_info:
                if g is not None:
                    n = g.number_of_nodes()
                    e = g.number_of_edges()
                    ratio = num_controllers / n * 100 if n > 0 else 0
                    print(f"{method_name:<15} {n:<12} {e:<12} {num_controllers:<10} {ratio:.1f}%")
                else:
                    print(f"{method_name:<15} {'N/A':<12} {'N/A':<12} {num_controllers:<10} {'N/A'}")
            print("-" * 80 + "\n")