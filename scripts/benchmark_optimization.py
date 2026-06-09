# -*- coding: utf-8 -*-
"""
性能基准测试脚本
比较 V1 和 V2 优化算法的性能
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import os
import sys
import time
import argparse
import networkx as nx
import numpy as np
from tabulate import tabulate

# 设置环境变量以控制版本
def set_version(use_v2):
    os.environ['USE_OPTIMIZATION_V2'] = '1' if use_v2 else '0'

def load_test_graph(name, nodes=100):
    """加载或生成测试图"""
    if name.startswith('BA_'):
        n = int(name.split('_')[1])
        return nx.barabasi_albert_graph(n, 3)
    else:
        # 从文件加载
        gml_path = os.path.join('dataset', 'testdata', f'{name}.gml')
        if os.path.exists(gml_path):
            G = nx.read_gml(gml_path)
            # 转换节点标签为整数
            G = nx.convert_node_labels_to_integers(G)
            return G
    return None

def benchmark_genetic_algorithm(G, version='v2'):
    """测试遗传算法性能"""
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    
    if version == 'v2':
        from src.topology.genetic_optimization_v2 import solution_v2 as solution
    else:
        from src.topology.genetic_optimization import solution
    
    start_time = time.time()
    best_solution, best_fitness = solution(nodes, edges)
    elapsed = time.time() - start_time
    
    return elapsed, best_fitness

def benchmark_onion(G, version='v2'):
    """测试 ONION 算法性能"""
    if version == 'v2':
        from src.topology.onion_optimization_v2 import OnionOptimizerV2 as OnionOptimizer
        optimizer = OnionOptimizer(G)
    else:
        from src.topology.onion_optimization import OnionOptimizer
        optimizer = OnionOptimizer(G)
    
    start_time = time.time()
    G_optimized = optimizer.optimize_network(verbose=False)
    elapsed = time.time() - start_time
    
    # 计算优化后的鲁棒性（简化版）
    return elapsed, G_optimized.number_of_edges()

def run_benchmark(datasets, n_runs=3):
    """运行完整基准测试"""
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Testing: {dataset}")
        print('='*60)
        
        G = load_test_graph(dataset)
        if G is None:
            print(f"Failed to load {dataset}")
            continue
        
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # 测试遗传算法
        print("\n--- Genetic Algorithm ---")
        ga_results = {'v1': [], 'v2': []}
        
        for version in ['v1', 'v2']:
            for run in range(n_runs):
                try:
                    elapsed, fitness = benchmark_genetic_algorithm(G, version)
                    ga_results[version].append(elapsed)
                    print(f"  {version.upper()} Run {run+1}: {elapsed:.2f}s (fitness={fitness:.4f})")
                except Exception as e:
                    print(f"  {version.upper()} Run {run+1}: ERROR - {e}")
        
        # 测试 ONION
        print("\n--- ONION Algorithm ---")
        onion_results = {'v1': [], 'v2': []}
        
        for version in ['v1', 'v2']:
            for run in range(n_runs):
                try:
                    elapsed, edges = benchmark_onion(G, version)
                    onion_results[version].append(elapsed)
                    print(f"  {version.upper()} Run {run+1}: {elapsed:.2f}s (edges={edges})")
                except Exception as e:
                    print(f"  {version.upper()} Run {run+1}: ERROR - {e}")
        
        # 计算统计
        result = {
            'Dataset': dataset,
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
        }
        
        if ga_results['v1'] and ga_results['v2']:
            v1_mean = np.mean(ga_results['v1'])
            v2_mean = np.mean(ga_results['v2'])
            speedup = v1_mean / v2_mean if v2_mean > 0 else 0
            result['GA_V1 (s)'] = f"{v1_mean:.2f}"
            result['GA_V2 (s)'] = f"{v2_mean:.2f}"
            result['GA_Speedup'] = f"{speedup:.2f}x"
        
        if onion_results['v1'] and onion_results['v2']:
            v1_mean = np.mean(onion_results['v1'])
            v2_mean = np.mean(onion_results['v2'])
            speedup = v1_mean / v2_mean if v2_mean > 0 else 0
            result['Onion_V1 (s)'] = f"{v1_mean:.2f}"
            result['Onion_V2 (s)'] = f"{v2_mean:.2f}"
            result['Onion_Speedup'] = f"{speedup:.2f}x"
        
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark optimization algorithms')
    parser.add_argument('--datasets', nargs='+', 
                       default=['BA_100', 'BA_200', 'BA_500'],
                       help='Datasets to test')
    parser.add_argument('--runs', type=int, default=2,
                       help='Number of runs per test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (only 1 run)')
    args = parser.parse_args()
    
    n_runs = 1 if args.quick else args.runs
    
    print("="*70)
    print("  OPTIMIZATION ALGORITHM BENCHMARK")
    print("  Comparing V1 (Original) vs V2 (High Performance)")
    print("="*70)
    
    # 检查 GPU 状态
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n[GPU] CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("\n[GPU] CUDA not available, using CPU")
    except ImportError:
        print("\n[GPU] PyTorch not installed")
    
    # 运行基准测试
    results = run_benchmark(args.datasets, n_runs)
    
    # 打印结果表格
    if results:
        print("\n" + "="*70)
        print("  BENCHMARK RESULTS SUMMARY")
        print("="*70)
        print(tabulate(results, headers='keys', tablefmt='grid'))
        
        # 计算平均加速比
        ga_speedups = []
        onion_speedups = []
        for r in results:
            if 'GA_Speedup' in r:
                try:
                    ga_speedups.append(float(r['GA_Speedup'].replace('x', '')))
                except:
                    pass
            if 'Onion_Speedup' in r:
                try:
                    onion_speedups.append(float(r['Onion_Speedup'].replace('x', '')))
                except:
                    pass
        
        print("\n" + "-"*40)
        if ga_speedups:
            print(f"Average GA Speedup: {np.mean(ga_speedups):.2f}x")
        if onion_speedups:
            print(f"Average Onion Speedup: {np.mean(onion_speedups):.2f}x")
        print("-"*40)

if __name__ == '__main__':
    main()
