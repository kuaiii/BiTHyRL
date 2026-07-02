# -*- coding: utf-8 -*-
"""BiT-HyRL 控制器选择（CI 算法、K-Median、混合 RL 选择）。"""
import os
import numpy as np
import torch
import networkx as nx

from src.utils.logger import get_logger
from . import config
from .model import MLPPolicy
from . import features

logger = get_logger(__name__)
DEVICE = config.DEVICE


def calculate_collective_influence(G, node, radius=2):
    """CI_l(v) = (k_v - 1) * sum(k_u - 1) for u in Ball(v, l)。"""
    if not G.has_node(node):
        return 0.0
    k_v = G.degree(node)
    if k_v == 0:
        return 0.0
    ball = {node}
    layer = {node}
    for _ in range(radius):
        next_layer = set()
        for n in layer:
            next_layer.update(G.neighbors(n))
        ball.update(next_layer)
        layer = next_layer
    s = sum(max(0, G.degree(u) - 1) for u in ball if u != node)
    return (k_v - 1) * s


def ci_select_subset(G, points, num_to_select, pre_selected, radius=2):
    """使用 CI 算法选择节点。"""
    if num_to_select <= 0:
        return []
    cand = [p for p in points if p not in pre_selected and G.has_node(p)]
    if not cand:
        return []
    selected = []
    remaining = set(cand)
    for _ in range(num_to_select):
        if not remaining:
            break
        scores = {n: calculate_collective_influence(G, n, radius=radius) for n in remaining}
        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.discard(best)
    return selected


def k_median_vectorized_subset(G, points, num_to_select, pre_selected, is_weight=0, use_ci=False):
    """K-Median 或 CI 选择剩余节点。"""
    if use_ci:
        return ci_select_subset(G, points, num_to_select, pre_selected, radius=2)
    if num_to_select <= 0:
        return []
    nodes_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    n_nodes = len(nodes_list)
    cand = [p for p in points if p not in pre_selected]
    n_cand = len(cand)
    if n_cand == 0:
        return []
    use_gpu = torch.cuda.is_available() and n_cand > 50
    dist = np.full((n_cand, n_nodes), np.inf, dtype=np.float32)
    for i, c in enumerate(cand):
        d = nx.single_source_dijkstra_path_length(G, c, weight='weight') if is_weight else nx.single_source_shortest_path_length(G, c)
        for t, v in d.items():
            if t in node_to_idx:
                dist[i, node_to_idx[t]] = v
    cur_min = np.full(n_nodes, np.inf, dtype=np.float32)
    for c in pre_selected:
        d = nx.single_source_dijkstra_path_length(G, c, weight='weight') if is_weight else nx.single_source_shortest_path_length(G, c)
        for t, v in d.items():
            if t in node_to_idx:
                idx = node_to_idx[t]
                if v < cur_min[idx]:
                    cur_min[idx] = v
    if use_gpu:
        dt = torch.tensor(dist, dtype=torch.float32, device=DEVICE)
        cm = torch.tensor(cur_min, dtype=torch.float32, device=DEVICE)
        mask = torch.ones(n_cand, dtype=torch.bool, device=DEVICE)
        sel = []
        for _ in range(num_to_select):
            if not mask.any():
                break
            new_d = torch.minimum(dt, cm.unsqueeze(0))
            tot = new_d.sum(dim=1)
            tot[~mask] = float('inf')
            idx = torch.argmin(tot).item()
            sel.append(idx)
            mask[idx] = False
            cm = torch.minimum(cm, dt[idx])
        return [cand[i] for i in sel]
    sel = []
    rem = set(range(n_cand))
    for _ in range(num_to_select):
        if not rem:
            break
        rl = list(rem)
        new_d = np.minimum(dist[rl], cur_min)
        tot = np.sum(new_d, axis=1)
        i_min = np.argmin(tot)
        idx = rl[i_min]
        sel.append(idx)
        rem.discard(idx)
        cur_min = np.minimum(cur_min, dist[idx])
    return [cand[i] for i in sel]


def load_or_train_model(model_path=None, num_features=5, strict=True):
    """加载或初始化 RL 模型，支持新旧两种 checkpoint 格式。特征维度不匹配时跳过加载，避免 input_proj 等 shape 报错。"""
    if model_path is None:
        model_path = config.DEFAULT_MODEL_PATH
    model = MLPPolicy(num_features=num_features, hidden_dim=64, output_dim=1)
    if os.path.exists(model_path):
        try:
            ck = torch.load(model_path, map_location=DEVICE, weights_only=False)
            
            if isinstance(ck, dict) and 'model_state_dict' in ck:
                state_dict = ck['model_state_dict']
            else:
                state_dict = ck
            
            # 检查特征维度：MLPPolicy 使用 input_proj，旧版可能用 fc1
            cf = None
            if 'input_proj.weight' in state_dict:
                cf = state_dict['input_proj.weight'].shape[1]
            elif 'fc1.weight' in state_dict:
                cf = state_dict['fc1.weight'].shape[1]
            if cf is not None and cf != num_features:
                logger.warning(f"Model feature dim mismatch: checkpoint {cf} vs current {num_features}. Using init.")
                return model
                    
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.warning(f"Load model failed: {e}. Using init.")
    return model


def predict(G, k, model=None, model_path=None, use_node2vec=True):
    """RL 预测控制器。"""
    x, node_list = features.get_node_features(G, use_node2vec=use_node2vec)
    if not node_list:
        return []
    nf = x.shape[1]
    if model is None:
        strict = not use_node2vec
        model = load_or_train_model(model_path, num_features=nf, strict=strict)
    model = model.to(DEVICE)
    model.eval()
    x = x.to(DEVICE)
    mask = torch.zeros(len(node_list), dtype=torch.bool, device=DEVICE)
    sel = []
    with torch.no_grad():
        for _ in range(k):
            probs = model(x, mask)
            a = torch.argmax(probs).item()
            sel.append(a)
            mask = mask.clone()
            mask[a] = True
    return [node_list[i] for i in sel]


def hybrid_rl_select(G, total_controllers_budget, model_path=None, use_ci=True):
    """混合 RL + CI/K-Median 选择。"""
    if model_path is None:
        model_path = config.DEFAULT_MODEL_PATH
    comps = [list(c) for c in nx.connected_components(G) if len(c) > 0]
    if not comps:
        return []
    comps.sort(key=len, reverse=True)
    alloc = []
    used = 0
    for _ in comps:
        if used < total_controllers_budget:
            alloc.append(1)
            used += 1
        else:
            alloc.append(0)
    rem = total_controllers_budget - used
    if rem > 0:
        n_tot = G.number_of_nodes()
        for i, nds in enumerate(comps):
            if rem <= 0:
                break
            share = int((len(nds) / n_tot) * total_controllers_budget)
            extra = max(0, share - alloc[i])
            take = min(rem, extra)
            alloc[i] += take
            rem -= take
    while rem > 0:
        for i in range(len(comps)):
            if rem <= 0:
                break
            if alloc[i] < len(comps[i]):
                alloc[i] += 1
                rem -= 1
    out = []
    for i, nds in enumerate(comps):
        k = alloc[i]
        if k <= 0:
            continue
        sub = G.subgraph(nds).copy()
        rl = predict(sub, 1, model=None, model_path=model_path, use_node2vec=True)
        out.extend(rl)
        need = k - 1
        if need > 0:
            extra = k_median_vectorized_subset(sub, nds, need, rl, is_weight=0, use_ci=use_ci)
            out.extend(extra)
    return out


# ============================================================
# GNN 预测函数
# ============================================================

def _resolve_gnn_model_path(model_path=None, prefer_curriculum=True):
    """
    解析 GNN 模型路径：若指定路径不存在，则依次尝试多个候选路径。
    
    优先级顺序 (prefer_curriculum=True):
    1. 指定的 model_path
    2. curriculum_final_dynamic_gat.pth (课程学习最终模型 - 推荐)
    3. curriculum_phase3_dynamic_gat.pth (课程学习第3阶段)
    4. deep_gat_gcc_L5_H8_D128.pth (深层 GAT)
    5. 其他 GNN 模型...
    
    Args:
        model_path: 指定的模型路径
        prefer_curriculum: 是否优先使用课程学习模型
    """
    candidates = []
    if model_path and os.path.exists(model_path):
        return model_path, _detect_model_type(model_path)
    if model_path:
        candidates.append(model_path)
        # 若为纯文件名，同时在 MODEL_DIR 下查找
        if os.path.basename(model_path) == model_path:
            candidates.append(os.path.join(config.MODEL_DIR, model_path))
    
    # 按优先级添加候选路径
    if prefer_curriculum:
        # 课程学习模型优先
        candidates.extend([
            os.path.join(config.MODEL_DIR, 'curriculum_final_dynamic_gat.pth'),       # 课程学习最终模型
            os.path.join(config.MODEL_DIR, 'curriculum_phase3_dynamic_gat.pth'),      # 课程学习第3阶段
            os.path.join(config.MODEL_DIR, 'curriculum_phase2_dynamic_gat.pth'),      # 课程学习第2阶段
        ])
    
    # 其他 GNN 模型
    candidates.extend([
        os.path.join(config.MODEL_DIR, 'deep_gat_gcc_L5_H8_D128.pth'),               # 深层 GAT GCC 模型
        os.path.join(config.MODEL_DIR, 'gnn_ppo_agent_robustness_focus50-150.pth'),   # 针对50-150节点优化
        os.path.join(config.MODEL_DIR, 'gnn_ppo_agent_robustness_improved.pth'),     # 改进版
        os.path.join(config.MODEL_DIR, 'gnn_ppo_agent_robustness.pth'),              # 鲁棒性模式
        os.path.join(config.MODEL_DIR, 'gnn_ppo_agent_survival.pth'),                # 生存模式
        os.path.join(config.MODEL_DIR, 'gnn_ppo_agent.pth'),                         # 通用
    ])
    
    for p in candidates:
        if os.path.exists(p):
            model_type = _detect_model_type(p)
            logger.info(f"使用 GNN 模型: {os.path.basename(p)} (类型: {model_type})")
            return p, model_type
    
    if model_path:
        logger.warning(f"GNN 模型文件不存在: {model_path}，已尝试: {candidates}")
    else:
        logger.warning(f"GNN 模型文件不存在，已尝试: {candidates}")
    return None, None


def _detect_model_type(model_path):
    """
    检测模型类型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        model_type: 'dynamic_gat', 'graph_transformer', 'unified_gat', 'gat', 'legacy_gat'
    """
    filename = os.path.basename(model_path).lower()
    
    # 根据文件名判断
    if 'dynamic_gat' in filename or 'curriculum' in filename:
        return 'dynamic_gat'
    if 'graph_transformer' in filename or 'transformer' in filename:
        return 'graph_transformer'
    if 'unified' in filename:
        return 'unified_gat'
    
    # 尝试从 checkpoint 判断
    try:
        ck = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(ck, dict):
            if 'model_type' in ck:
                return ck['model_type']
            state_dict = ck.get('model_state_dict', ck)
            # 检查特征层的键来判断
            keys = set(state_dict.keys())
            if 'dynamic_encoder.0.weight' in keys:
                return 'dynamic_gat'
            if 'transformer_layers.0.q_proj.weight' in keys:
                return 'graph_transformer'
            if 'feature_projector.projectors.32.weight' in keys:
                return 'unified_gat'
            if 'gat1.lin.weight' in keys or any('gat1.' in k for k in keys):
                return 'legacy_gat'
            if 'gat_layers.0.lin.weight' in keys:
                return 'gat'
    except Exception:
        pass
    
    # 默认返回 gat
    return 'gat'


def load_gnn_model(model_path=None, prefer_curriculum=True):
    """
    加载 GNN 模型（自动检测模型类型）
    
    支持的模型类型:
    - DynamicGATPolicy: 课程学习训练的动态 GAT 模型
    - GATPolicy: 深层 GAT 模型
    - GATPolicyLegacy: 旧版 2 层 GAT 模型
    
    Args:
        model_path: 模型路径，None 时自动搜索
        prefer_curriculum: 是否优先使用课程学习模型
        
    Returns:
        model: 加载的模型
        checkpoint: 模型检查点信息
        model_type: 模型类型字符串
    """
    resolved, model_type = _resolve_gnn_model_path(model_path, prefer_curriculum)
    if resolved is None:
        return None, None, None
    
    try:
        checkpoint = torch.load(resolved, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            state_dict = checkpoint

        # 若存在 DataParallel 的 "module." 前缀，去掉以便与当前模型匹配
        def _strip_module_prefix(sd):
            if not sd:
                return sd
            has_prefix = any(k.startswith('module.') for k in sd.keys())
            if not has_prefix:
                return sd
            return {k.replace('module.', '', 1): v for k, v in sd.items()}
        state_dict = _strip_module_prefix(state_dict)

        # 从 checkpoint 获取配置
        model_config = checkpoint.get('model_config', {})
        
        # 加载 DynamicGATPolicy (课程学习模型)
        if model_type == 'dynamic_gat':
            from .gnn_model import DynamicGATPolicy
            
            # 默认配置
            cfg = {
                'in_channels': model_config.get('in_channels', 64),
                'hidden_channels': model_config.get('hidden_channels', 96),
                'scale_encoding_dim': model_config.get('scale_encoding_dim', 8),
                'dynamic_feature_dim': model_config.get('dynamic_feature_dim', 6),
                'heads': model_config.get('num_heads', model_config.get('heads', 4)),
                'num_layers': model_config.get('num_layers', 3),
                'dropout': model_config.get('dropout', 0.15),
            }
            
            model = DynamicGATPolicy(**cfg).to(DEVICE)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded DynamicGATPolicy from {resolved}")
            model.eval()
            return model, checkpoint, 'dynamic_gat'
        
        # 加载 GraphTransformerPolicy
        if model_type == 'graph_transformer':
            from .gnn_model import GraphTransformerPolicy
            
            cfg = {
                'in_channels': model_config.get('in_channels', 64),
                'hidden_channels': model_config.get('hidden_channels', 96),
                'scale_encoding_dim': model_config.get('scale_encoding_dim', 8),
                'dynamic_feature_dim': model_config.get('dynamic_feature_dim', 6),
                'num_heads': model_config.get('num_heads', 4),
                'num_layers': model_config.get('num_layers', 3),
                'dropout': model_config.get('dropout', 0.15),
                'use_laplacian_pe': model_config.get('use_laplacian_pe', True),
            }
            
            model = GraphTransformerPolicy(**cfg).to(DEVICE)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded GraphTransformerPolicy from {resolved}")
            model.eval()
            return model, checkpoint, 'graph_transformer'
        
        # 加载 UnifiedGATPolicy
        if model_type == 'unified_gat':
            from .gnn_model import UnifiedGATPolicy
            
            cfg = {
                'in_channels': model_config.get('in_channels', 64),
                'hidden_channels': model_config.get('hidden_channels', 96),
                'scale_encoding_dim': model_config.get('scale_encoding_dim', 8),
                'heads': model_config.get('num_heads', model_config.get('heads', 4)),
                'num_layers': model_config.get('num_layers', 3),
                'dropout': model_config.get('dropout', 0.15),
            }
            
            model = UnifiedGATPolicy(**cfg).to(DEVICE)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded UnifiedGATPolicy from {resolved}")
            model.eval()
            return model, checkpoint, 'unified_gat'
        
        # 加载 GATPolicy 或 GATPolicyLegacy
        from .gnn_model import GATPolicy, GATPolicyLegacy
        
        keys_set = set(state_dict.keys())
        is_legacy = (
            'gat1.lin.weight' in keys_set
            or 'gat1.att_src' in keys_set
            or any('gat1.' in k for k in keys_set)
        )
        if is_legacy and ('gat_layers.0.lin.weight' in keys_set or 'input_proj.0.weight' in keys_set):
            is_legacy = False

        in_channels = checkpoint.get('in_channels', 128)
        hidden_channels = checkpoint.get('hidden_channels', 128)
        heads = checkpoint.get('heads', 8)
        num_layers = checkpoint.get('num_layers', 5 if not is_legacy else 2)

        if is_legacy:
            model = GATPolicyLegacy(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                heads=heads,
                num_layers=2,
            ).to(DEVICE)
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded legacy 2-layer GNN model from {resolved}")
            model.eval()
            return model, checkpoint, 'legacy_gat'
        else:
            model = GATPolicy(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                heads=heads,
                num_layers=num_layers,
            ).to(DEVICE)
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded GATPolicy from {resolved}")
            model.eval()
            return model, checkpoint, 'gat'

    except Exception as e:
        logger.warning(f"加载 GNN 模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def gnn_predict(G, k, model=None, model_path=None, model_type=None, deterministic=True, embed_dim=128, dre_dim=0):
    """
    使用 GNN 模型预测控制器（自动适配模型类型）
    
    支持 DynamicGATPolicy、GATPolicy、GraphTransformerPolicy 等多种模型。
    
    Args:
        G: NetworkX 图
        k: 要选择的控制器数量
        model: 预加载的模型（可选）
        model_path: 模型路径（如果 model 为 None）
        model_type: 模型类型（如果 model 不为 None 且需要指定类型）
        deterministic: 是否使用确定性策略（贪婪选择）
        embed_dim: Node2Vec 嵌入维度 (默认128，仅用于旧版 GAT 模型)
        dre_dim: DRE 嵌入维度 (默认0)
        
    Returns:
        centers: 选择的控制器列表
    """
    # #region agent log - 初始化调试日志收集器
    debug_log_data = {"probs_entropy": [], "probs_max": [], "probs_std": [], "selected_degrees": []}
    # #endregion
    
    try:
        from .gnn_model import (
            graph_to_pyg_data, 
            get_gnn_node_features,
            get_unified_features,
            get_scale_encoding,
            compute_coverage_features,
            compute_dispersion_features,
            get_laplacian_pe,
        )
    except ImportError as e:
        logger.error(f"导入 GNN 模块失败: {e}")
        import random
        nodes = list(G.nodes())
        return random.sample(nodes, min(k, len(nodes)))
    
    num_nodes = G.number_of_nodes()
    if num_nodes < 2:
        return list(G.nodes())[:k] if k <= num_nodes else list(G.nodes())
    
    # 加载模型
    if model is None:
        model, checkpoint, model_type = load_gnn_model(model_path)
        if model is None:
            logger.warning("GNN 模型加载失败，使用随机选择")
            import random
            nodes = list(G.nodes())
            return random.sample(nodes, min(k, len(nodes)))
        if checkpoint:
            embed_dim = checkpoint.get('in_channels', embed_dim)
    
    # 根据模型类型选择特征和推理方式
    if model_type in ('dynamic_gat', 'graph_transformer', 'unified_gat'):
        # 使用统一特征（规模自适应 Node2Vec + 规模编码）
        node_features, scale_encoding, node_list, raw_dim = get_unified_features(G, device=DEVICE)
        _, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)
        
        # 特征投影
        if hasattr(model, 'feature_projector'):
            with torch.no_grad():
                x = model.feature_projector(node_features, raw_dim)
        else:
            x = node_features
        
        # 拉普拉斯位置编码（仅 Graph Transformer）
        laplacian_pe = None
        if model_type == 'graph_transformer':
            hidden_dim = model.hidden_channels
            laplacian_pe = get_laplacian_pe(G, hidden_dim=hidden_dim, device=DEVICE)
        
        selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        centers = []
        
        with torch.no_grad():
            for step in range(k):
                # 计算动态特征
                if model_type in ('dynamic_gat', 'graph_transformer'):
                    coverage_feat = compute_coverage_features(G, centers, node_list, device=DEVICE)
                    dispersion_feat = compute_dispersion_features(G, centers, node_list, device=DEVICE)
                    dynamic_features = torch.cat([coverage_feat, dispersion_feat], dim=-1)
                else:
                    dynamic_features = None
                
                # 获取动作
                if model_type == 'graph_transformer':
                    action, log_prob, entropy, _ = model.get_action(
                        x, edge_index, scale_encoding, dynamic_features, laplacian_pe,
                        selected_mask=selected_mask, deterministic=deterministic
                    )
                elif model_type == 'dynamic_gat':
                    action, log_prob, entropy, _ = model.get_action(
                        x, edge_index, scale_encoding, dynamic_features,
                        selected_mask=selected_mask, deterministic=deterministic
                    )
                else:  # unified_gat
                    action, log_prob, entropy, _ = model.get_action(
                        x, edge_index, scale_encoding,
                        selected_mask=selected_mask, deterministic=deterministic
                    )
                
                action_idx = action.item()
                centers.append(node_list[action_idx])
                selected_mask[action_idx] = True
                
                # #region agent log - 假设D: 记录GNN预测置信度
                if step == 0:  # 只记录第一步的详细信息
                    try:
                        import json, os, time as time_module
                        debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
                        entropy_val = entropy.item() if entropy is not None else -1
                        log_prob_val = log_prob.item() if log_prob is not None else 0
                        # 计算选中节点的度
                        selected_degree = G.degree(node_list[action_idx]) if node_list[action_idx] in G else 0
                        degrees = [d for n, d in G.degree()]
                        avg_degree = sum(degrees) / len(degrees) if degrees else 0
                        log_entry = {
                            "timestamp": int(time_module.time() * 1000),
                            "location": "selection.py:gnn_predict",
                            "message": "GNN prediction confidence for hypothesis D",
                            "hypothesisId": "D",
                            "sessionId": "debug-session",
                            "data": {
                                "model_type": model_type,
                                "num_nodes": num_nodes,
                                "num_controllers": k,
                                "entropy": round(entropy_val, 4),
                                "log_prob": round(log_prob_val, 4),
                                "selected_node_degree": selected_degree,
                                "avg_network_degree": round(avg_degree, 4),
                                "degree_ratio": round(selected_degree / avg_degree, 4) if avg_degree > 0 else 0
                            }
                        }
                        with open(debug_log_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                    except Exception as e:
                        pass
                # #endregion
        
        return centers
    
    else:
        # 旧版 GATPolicy / GATPolicyLegacy 模型
        x, node_list = get_gnn_node_features(G, device=DEVICE, embed_dim=embed_dim, dre_dim=dre_dim)
        _, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)
        
        selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        centers = []
        
        with torch.no_grad():
            for _ in range(k):
                action, _, _, _ = model.get_action(
                    x, edge_index, selected_mask=selected_mask, deterministic=deterministic
                )
                action_idx = action.item()
                centers.append(node_list[action_idx])
                selected_mask[action_idx] = True
        
        return centers


def hybrid_ci_select(G, total_controllers_budget, use_ci=True):
    """
    混合 CI 选择：按连通分量分配预算，每分量内用 CI 算法选控制器。
    GNN/MLP 不可用时的稳妥降级，避免特征维度不一致导致的加载失败。
    """
    comps = [list(c) for c in nx.connected_components(G) if len(c) > 0]
    if not comps:
        return []
    comps.sort(key=len, reverse=True)
    alloc = []
    used = 0
    for _ in comps:
        alloc.append(1 if used < total_controllers_budget else 0)
        used += 1
    rem = total_controllers_budget - used
    if rem > 0:
        n_tot = G.number_of_nodes()
        for i, nds in enumerate(comps):
            if rem <= 0:
                break
            share = int((len(nds) / n_tot) * total_controllers_budget)
            extra = max(0, share - alloc[i])
            take = min(rem, extra)
            alloc[i] += take
            rem -= take
    while rem > 0:
        for i in range(len(comps)):
            if rem <= 0:
                break
            if alloc[i] < len(comps[i]):
                alloc[i] += 1
                rem -= 1
    out = []
    for i, nds in enumerate(comps):
        k = alloc[i]
        if k <= 0:
            continue
        sub = G.subgraph(nds).copy()
        sel = ci_select_subset(sub, nds, k, [], radius=2)
        out.extend(sel)
    return out


def hybrid_gnn_select(G, total_controllers_budget, model_path=None, use_ci=False, gnn_ratio=1.0, prefer_curriculum=True):
    """
    使用 GNN 模型选择控制器（自动适配模型类型）
    
    支持 DynamicGATPolicy（课程学习）、GATPolicy 等多种模型。
    全部使用 GNN 模型进行控制器选择。
    
    Args:
        G: NetworkX 图
        total_controllers_budget: 控制器总预算
        model_path: GNN 模型路径，None 时自动搜索
        use_ci: 已废弃，保留参数兼容性
        gnn_ratio: 已废弃，全部由 GNN 选择
        prefer_curriculum: 是否优先使用课程学习模型
        
    Returns:
        centers: 选择的控制器列表
    """
    # 加载 GNN 模型
    gnn_model, checkpoint, model_type = load_gnn_model(model_path, prefer_curriculum)
    
    if gnn_model is None:
        # GNN 模型不可用时，使用 CI 算法作为降级方案
        logger.warning("GNN 模型不可用，使用 CI 算法选择控制器")
        return hybrid_ci_select(G, total_controllers_budget, use_ci=True)
    
    # 处理单连通图（常见情况，优化性能）
    if nx.is_connected(G):
        return gnn_predict(
            G, total_controllers_budget, 
            model=gnn_model, model_type=model_type, 
            deterministic=True
        )
    
    # 按连通分量分配控制器
    comps = [list(c) for c in nx.connected_components(G) if len(c) > 0]
    if not comps:
        return []
    
    comps.sort(key=len, reverse=True)
    
    # 分配预算
    alloc = []
    used = 0
    for _ in comps:
        if used < total_controllers_budget:
            alloc.append(1)
            used += 1
        else:
            alloc.append(0)
    
    # 分配剩余预算
    rem = total_controllers_budget - used
    if rem > 0:
        n_tot = G.number_of_nodes()
        for i, nds in enumerate(comps):
            if rem <= 0:
                break
            share = int((len(nds) / n_tot) * total_controllers_budget)
            extra = max(0, share - alloc[i])
            take = min(rem, extra)
            alloc[i] += take
            rem -= take
    
    while rem > 0:
        for i in range(len(comps)):
            if rem <= 0:
                break
            if alloc[i] < len(comps[i]):
                alloc[i] += 1
                rem -= 1
    
    # 选择控制器（全部使用 GNN）
    out = []
    for i, nds in enumerate(comps):
        k = alloc[i]
        if k <= 0:
            continue
        
        sub = G.subgraph(nds).copy()
        
        # 使用 GNN 选择控制器
        gnn_selected = gnn_predict(sub, k, model=gnn_model, model_type=model_type, deterministic=True)
        out.extend(gnn_selected)
    
    return out
