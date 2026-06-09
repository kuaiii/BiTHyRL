# -*- coding: utf-8 -*-
"""
生成 testdata：
1. 在 dataset/testdata 中生成节点数 400 的 BA 拓扑
2. 在 dataset/all/syn 中生成节点数 20-100 的 BA、RA(ER)、WS 拓扑
3. 将 testdata 中文件名里的 _ 替换为 -
"""
import os
import sys
import random
import numpy as np
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_TESTDATA = os.path.join(ROOT, 'dataset', 'testdata')
RAW_TESTDATA = os.path.join(ROOT, 'dataset', 'all', 'syn')


def generate_ba(n, m=4, seed=None):
    return nx.barabasi_albert_graph(n, m, seed=seed)


def generate_ra(n, seed=None):
    """RA = 随机图 (Erdos-Renyi)，保证连通"""
    p = min(0.99, max(0.01, 2.0 * np.log(n) / n))
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            G.add_edge(u, v)
    return G


def generate_ws(n, k=4, p=0.3, seed=None):
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def ensure_connected(G):
    if nx.is_connected(G):
        return G
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        u = random.choice(list(components[i]))
        v = random.choice(list(components[i + 1]))
        G.add_edge(u, v)
    return G


def generate_ba_400():
    """在 dataset/testdata 生成 400 节点的 BA 拓扑"""
    os.makedirs(DATA_TESTDATA, exist_ok=True)
    path = os.path.join(DATA_TESTDATA, 'BA-400.gml')
    G = generate_ba(400, m=4, seed=42)
    G = ensure_connected(G)
    nx.write_gml(G, path)
    print(f"已生成: {path} (节点数 {G.number_of_nodes()})")
    return path


def generate_raw_testdata(count_per_type=8):
    """在 dataset/all/syn 生成 20-100 节点的 BA、RA、WS，文件名用 - 不用 _"""
    os.makedirs(RAW_TESTDATA, exist_ok=True)
    types = [
        ('BA', generate_ba),
        ('RA', generate_ra),
        ('WS', generate_ws),
    ]
    created = []
    for label, gen_func in types:
        for i in range(count_per_type):
            n = random.randint(20, 100)
            seed = random.randint(0, 1000000)
            try:
                if label == 'BA':
                    m = random.randint(2, min(4, n // 2))
                    G = gen_func(n, m=m, seed=seed)
                elif label == 'WS':
                    k = random.randint(4, min(8, n // 2))
                    p = random.uniform(0.1, 0.5)
                    G = gen_func(n, k=k, p=p, seed=seed)
                else:
                    G = gen_func(n, seed=seed)
                G = ensure_connected(G)
                if not nx.is_connected(G):
                    continue
                # 文件名用连字符（避免后续需把 _ 替换为 -）
                fname = f"{label}-{n}-{i}.gml"
                path = os.path.join(RAW_TESTDATA, fname)
                nx.write_gml(G, path)
                created.append(path)
            except Exception as e:
                print(f"跳过 {label} n={n}: {e}")
    print(f"已在 dataset/all/syn 生成 {len(created)} 个图 (BA/RA/WS 各约 {count_per_type} 个)")
    return created


def replace_underscore_with_hyphen_in_testdata():
    """将 dataset/testdata 与 dataset/all/syn 中文件名里的 _ 替换为 -"""
    for dir_path in [DATA_TESTDATA, RAW_TESTDATA]:
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if '_' not in fname:
                continue
            old_path = os.path.join(dir_path, fname)
            if not os.path.isfile(old_path):
                continue
            new_fname = fname.replace('_', '-')
            if new_fname == fname:
                continue
            new_path = os.path.join(dir_path, new_fname)
            if os.path.exists(new_path):
                print(f"已存在，跳过重命名: {new_fname}")
                continue
            os.rename(old_path, new_path)
            print(f"重命名: {fname} -> {new_fname}")


def main():
    random.seed(42)
    np.random.seed(42)
    print("1. 生成 dataset/testdata 中 BA-400 ...")
    generate_ba_400()
    print("\n2. 生成 dataset/all/syn 中 20-100 节点 BA/RA/WS ...")
    generate_raw_testdata(count_per_type=8)
    print("\n3. 将 testdata 中文件名中的 _ 替换为 - ...")
    replace_underscore_with_hyphen_in_testdata()
    print("\n完成。")


if __name__ == '__main__':
    main()
