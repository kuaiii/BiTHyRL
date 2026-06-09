# -*- coding: utf-8 -*-
"""BiT-HyRL 配置与常量。"""
import os
import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"[BiT-HyRL] 使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("[BiT-HyRL] 使用 CPU")

# 模型默认路径（相对于项目根）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "train", "v1", "checkpoints")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "rl_agent.pth")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "bit_hyrl")

# Node2Vec
try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False
    Node2Vec = None
    logger.warning("Node2Vec not available. Install with: pip install node2vec")

# ============================================================
# 统一模型配置
# ============================================================# 网络规模阈值
SCALE_TINY = 30        # 极小规模 (< 30 节点)
SCALE_SMALL = 50       # 小规模 (30-50 节点)
SCALE_MEDIUM_LOW = 100 # 中小规模 (50-100 节点)
SCALE_MEDIUM = 200     # 中等规模 (100-200 节点)
SCALE_LARGE = 500      # 大规模 (200-500 节点)

# 统一GAT模型默认配置
UNIFIED_MODEL_CONFIG = {
    'in_channels': 64,           # Node2Vec投影后的特征维度
    'hidden_channels': 96,       # 隐藏层维度
    'scale_encoding_dim': 8,     # 规模编码维度
    'heads': 4,                  # GAT注意力头数
    'num_layers': 3,             # GAT层数
    'dropout': 0.15,             # Dropout比例
}# 规模自适应Node2Vec参数
SCALE_ADAPTIVE_NODE2VEC_PARAMS = {
    'tiny': dict(dimensions=32, walk_length=6, num_walks=100, p=1.0, q=2.5),
    'small': dict(dimensions=32, walk_length=8, num_walks=80, p=1.0, q=2.0),
    'medium_low': dict(dimensions=48, walk_length=12, num_walks=60, p=1.0, q=1.5),
    'medium': dict(dimensions=64, walk_length=15, num_walks=50, p=1.0, q=1.0),
    'large': dict(dimensions=64, walk_length=20, num_walks=40, p=1.0, q=0.8),
}

# 训练默认配置
UNIFIED_TRAINING_CONFIG = {
    'epochs': 200,
    'lr': 3e-4,
    'k_ratio': 0.1,
    'collect_per_epoch': 30,
    'use_curriculum': True,
    'use_scale_balance': True,
    'focus_range': (50, 150),
    'focus_weight': 2.0,
}

# 统一模型默认保存路径
UNIFIED_MODEL_PATH = os.path.join(MODEL_DIR, "unified_gat_policy.pth")
