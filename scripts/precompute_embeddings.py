#!/usr/bin/env python3
"""
预计算所有图的 Node2Vec + DRE 嵌入，保存到缓存目录。

用法:
    python scripts/precompute_embeddings.py --data-dir dataset/all/syn --workers 8
"""
import sys
import os
import argparse
import hashlib
import numpy as np
from tqdm import tqdm
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.bit_hyrl.gnn_model import get_gnn_node_features


def graph_hash(G):
    """基于图结构生成稳定哈希（用于缓存文件名）"""
    nodes = sorted(G.nodes(), key=str)
    edges = sorted(tuple(sorted((str(u), str(v)))) for u, v in G.edges())
    content = f"nodes:{nodes},edges:{edges}"
    return hashlib.md5(content.encode()).hexdigest()


def precompute_single(args):
    """单图预计算（用于多进程）"""
    filepath, cache_dir, embed_dim, dre_dim = args
    try:
        G = nx.read_gml(filepath)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
            G = nx.convert_node_labels_to_integers(G)
        
        h = graph_hash(G)
        cache_path = os.path.join(cache_dir, f"{h}.npy")
        
        if os.path.exists(cache_path):
            return filepath, True, "cached"
        
        features, _ = get_gnn_node_features(
            G, device='cpu', embed_dim=embed_dim, dre_dim=dre_dim, seed=42
        )
        np.save(cache_path, features.numpy())
        return filepath, True, "computed"
    except Exception as e:
        return filepath, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="预计算图嵌入缓存")
    parser.add_argument('--data-dir', type=str, default='dataset/all/syn',
                        help='图数据目录')
    parser.add_argument('--cache-dir', type=str, default='dataset/all/syn_embedding_cache',
                        help='缓存目录')
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Node2Vec 维度')
    parser.add_argument('--dre-dim', type=int, default=256,
                        help='DRE 维度')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行进程数')
    args = parser.parse_args()
    
    data_dir = os.path.join(ROOT, args.data_dir)
    cache_dir = os.path.join(ROOT, args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    gml_files = [f for f in os.listdir(data_dir) if f.endswith('.gml')]
    gml_files.sort()
    print(f"共 {len(gml_files)} 个图，缓存目录: {cache_dir}")
    
    # 检查已缓存数量
    existing = set(os.listdir(cache_dir))
    todo = []
    for f in gml_files:
        G = nx.read_gml(os.path.join(data_dir, f))
        h = graph_hash(G)
        if f"{h}.npy" not in existing:
            todo.append((os.path.join(data_dir, f), cache_dir, args.embed_dim, args.dre_dim))
    
    print(f"已缓存: {len(gml_files) - len(todo)}，待计算: {len(todo)}")
    
    if not todo:
        print("全部已缓存，无需计算")
        return
    
    # 多进程预计算
    from multiprocessing import Pool
    
    computed = 0
    failed = 0
    with Pool(processes=args.workers) as pool:
        for filepath, success, msg in tqdm(
            pool.imap_unordered(precompute_single, todo),
            total=len(todo),
            desc="预计算嵌入"
        ):
            if success and msg == "computed":
                computed += 1
            elif not success:
                failed += 1
                print(f"失败: {os.path.basename(filepath)} - {msg}")
    
    print(f"完成: 新增 {computed} 个缓存，失败 {failed} 个")


if __name__ == '__main__':
    main()
