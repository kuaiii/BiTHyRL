# -*- coding: utf-8 -*-
"""
BiT-HyRL: Bi-modal Topology Hybrid Reinforcement Learning

模块结构:
- config: 设备、路径、Node2Vec 可用性
- topology: 双峰拓扑 create_bimodal_network_exact
- features: Node2Vec + 统计特征 get_node_features, get_node2vec_embeddings
- model: MLPPolicy (传统 MLP 策略网络)
- gnn_model: GATPolicy, DynamicGATPolicy, GraphTransformerPolicy (GNN 策略网络)
- reward: calculate_stepwise_reward, calculate_reward, calculate_gcc_focused_reward
         + 课程学习奖励函数 (Phase 1/2/3)
- selection: CI / K-Median, hybrid_rl_select, gnn_predict, hybrid_gnn_select
- training: train_offline_single, train_and_select, train_gnn_ppo_optimized
- ppo_trainer: PPOTrainer, CurriculumPPOTrainer (PPO 训练器)
- attack_plot: simulate_attack_and_plot
- reconstruct: bit_hyrl_reconstruct

新增功能 (v2.1):
1. 动态覆盖特征 (Dynamic State Representation)
   - compute_coverage_features: 计算节点到最近控制器的距离特征
   - compute_dispersion_features: 计算分散度相关特征
   
2. Graph Transformer 编码器
   - GraphTransformerPolicy: 全图注意力机制，捕捉长距离依赖
   - LaplacianPositionalEncoding: 拉普拉斯位置编码
   
3. 课程学习 (Curriculum Learning)
   - Phase 1: 只优化 GCC (连通性)
   - Phase 2: GCC + 覆盖率
   - Phase 3: GCC + 覆盖率 + 分散度 + 抗攻击
"""
from . import config
from .topology import create_bimodal_network_exact
from .features import get_node_features, get_node2vec_embeddings
from .model import MLPPolicy
from .reward import (
    calculate_stepwise_reward,
    calculate_reward,
    calculate_gcc_focused_reward,
    calculate_gcc_stepwise_reward,
    calculate_multi_attack_reward,
    # 课程学习奖励函数
    calculate_phase1_reward,
    calculate_phase2_reward,
    calculate_phase3_reward,
    get_curriculum_reward_fn,
    calculate_curriculum_reward,
    CurriculumRewardPhase,
)
from .selection import (
    hybrid_rl_select,
    predict,
    load_or_train_model,
    ci_select_subset,
    k_median_vectorized_subset,
    gnn_predict,
    hybrid_gnn_select,
    load_gnn_model,
)
from .training import (
    train_offline_single,
    train_and_select,
    train_offline_optimized,
    evaluate_model,
    train_gnn_ppo_optimized,
    evaluate_gnn_model,
)
from .attack_plot import simulate_attack_and_plot
from .reconstruct import bit_hyrl_reconstruct

# GNN 模块（可选导入，因为需要 torch-geometric）
try:
    from .gnn_model import (
        # 原有模型
        GATPolicy,
        UnifiedGATPolicy,
        graph_to_pyg_data,
        # 新增：动态特征计算
        compute_coverage_features,
        compute_dispersion_features,
        # 新增：动态 GAT 模型
        DynamicGATPolicy,
        # 新增：Graph Transformer
        GraphTransformerPolicy,
        GraphTransformerLayer,
        LaplacianPositionalEncoding,
        get_laplacian_pe,
    )
    from .ppo_trainer import PPOTrainer, train_gnn_ppo, UnifiedPPOTrainer
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    GATPolicy = None
    UnifiedGATPolicy = None
    DynamicGATPolicy = None
    GraphTransformerPolicy = None
    PPOTrainer = None

__all__ = [
    # 配置
    "config",
    "GNN_AVAILABLE",
    # 拓扑
    "create_bimodal_network_exact",
    # 特征
    "get_node_features",
    "get_node2vec_embeddings",
    # 模型 - 原有
    "MLPPolicy",
    "GATPolicy",
    "UnifiedGATPolicy",
    # 模型 - 新增
    "DynamicGATPolicy",           # 动态覆盖特征 GAT
    "GraphTransformerPolicy",     # Graph Transformer
    "GraphTransformerLayer",
    "LaplacianPositionalEncoding",
    # 动态特征计算
    "compute_coverage_features",
    "compute_dispersion_features",
    "get_laplacian_pe",
    # 奖励函数 - 原有
    "calculate_stepwise_reward",
    "calculate_reward",
    "calculate_gcc_focused_reward",
    "calculate_gcc_stepwise_reward",
    "calculate_multi_attack_reward",
    # 奖励函数 - 课程学习
    "calculate_phase1_reward",
    "calculate_phase2_reward",
    "calculate_phase3_reward",
    "get_curriculum_reward_fn",
    "calculate_curriculum_reward",
    "CurriculumRewardPhase",
    # 选择
    "hybrid_rl_select",
    "predict",
    "load_or_train_model",
    "ci_select_subset",
    "k_median_vectorized_subset",
    "gnn_predict",
    "hybrid_gnn_select",
    "load_gnn_model",
    # 训练
    "train_offline_single",
    "train_and_select",
    "train_offline_optimized",
    "evaluate_model",
    "train_gnn_ppo_optimized",
    "evaluate_gnn_model",
    "PPOTrainer",
    "UnifiedPPOTrainer",
    # 攻击与重构
    "simulate_attack_and_plot",
    "bit_hyrl_reconstruct",
]
