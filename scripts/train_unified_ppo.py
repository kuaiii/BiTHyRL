#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 PPO 训练脚本 (Topology Refinement + Controller Selection)

用法：
  python scripts/train_unified_ppo.py --epochs 20 --B-budget 3 --K-ratio 0.1 --max-nodes 100
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import csv
import math
import random
import datetime
import logging
import re
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('node2vec').setLevel(logging.WARNING)

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_network_exact
from src.bit_hyrl import config
from src.utils.logger import setup_logger, get_logger
from src.unified_ppo.topology_policy import TopologyPolicy
from src.unified_ppo.unified_trainer import UnifiedPPOTrainer
from src.unified_ppo.reward_unified import calculate_unified_reward

logger = get_logger(__name__)


class TeeOutput:
    def __init__(self, filepath, stream):
        self.file = open(filepath, 'a', encoding='utf-8')
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def init_logger():
    log_dir = os.path.join(ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'train_unified_ppo_{timestamp}.log'
    log_path = os.path.join(log_dir, log_filename)
    setup_logger(log_dir=log_dir, log_filename=log_filename, level=logging.INFO,
                 console_level=logging.INFO)
    tee_out = TeeOutput(log_path, sys.stdout)
    tee_err = TeeOutput(log_path, sys.stderr)
    sys.stdout = tee_out
    sys.stderr = tee_err
    logger.info(f"训练日志: {log_path}")
    return log_path


def _load_single_graph(args):
    filepath, name, min_nodes, max_nodes = args
    try:
        G_original, _ = load_graph(filepath, verbose=False)
        if G_original is None or G_original.number_of_nodes() < min_nodes:
            return None, 'skipped'
        if G_original.number_of_nodes() > max_nodes:
            return None, 'skipped'
        if not nx.is_connected(G_original):
            return None, 'skipped'
        return {'graph': G_original, 'name': name,
                'nodes': G_original.number_of_nodes(),
                'edges': G_original.number_of_edges()}, 'ok'
    except Exception as e:
        return None, f'failed:{e}'


def _extract_nodes_from_filename(filename):
    """从文件名如 'BA_n50_i10.gml' 中提取节点数，失败返回 None。"""
    m = re.search(r'[_\-]n(\d+)[_\-]', filename)
    if m:
        return int(m.group(1))
    return None


def load_training_graphs(data_dir=None, max_graphs=None, min_nodes=20, max_nodes=100,
                         seed=42, num_workers=8):
    if data_dir is None:
        data_dir = os.path.join(ROOT, 'dataset', 'all', 'syn')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"训练数据目录不存在: {data_dir}")

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.gml')]
    # 快速文件名过滤（如果文件名包含节点数）
    gml_files = []
    for f in all_files:
        n = _extract_nodes_from_filename(f)
        if n is not None and (n < min_nodes or n > max_nodes):
            continue
        gml_files.append(f)
    if not gml_files:
        gml_files = all_files  # 回退到全部文件

    gml_files.sort()
    random.seed(seed)
    random.shuffle(gml_files)
    if max_graphs and max_graphs < len(gml_files):
        gml_files = gml_files[:max_graphs]

    args_list = [(os.path.join(data_dir, f), os.path.splitext(f)[0], min_nodes, max_nodes)
                 for f in gml_files]

    graphs = []
    skipped = 0
    failed = 0
    from multiprocessing import Pool
    with Pool(processes=num_workers) as pool:
        for result, status in tqdm(pool.imap(_load_single_graph, args_list),
                                   total=len(args_list), desc='Loading graphs', ncols=80):
            if status == 'ok':
                graphs.append(result['graph'])
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1
                logger.warning(f"加载失败: {status}")
    print(f"加载完成: {len(graphs)} 个有效图，跳过 {skipped} 个，失败 {failed} 个")
    return graphs


def generate_synthetic_data(num_graphs=200, min_nodes=50, max_nodes=100, seed=42):
    """快速生成合成训练图（BA/WS/ER）。"""
    random.seed(seed)
    np.random.seed(seed)
    graphs = []
    graph_types = ['ba', 'ws', 'er']
    for i in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        gtype = random.choice(graph_types)
        try:
            if gtype == 'ba':
                m = random.randint(2, min(5, n // 2))
                G = nx.barabasi_albert_graph(n, m)
            elif gtype == 'ws':
                k = random.randint(4, min(10, n // 2))
                p = random.uniform(0.1, 0.5)
                G = nx.watts_strogatz_graph(n, k, p)
            else:
                p = random.uniform(0.05, 0.15)
                G = nx.erdos_renyi_graph(n, p)
                while not nx.is_connected(G):
                    p += 0.01
                    G = nx.erdos_renyi_graph(n, p)
            if G.number_of_nodes() >= min_nodes and nx.is_connected(G):
                graphs.append(G)
        except Exception:
            continue
    print(f"合成数据生成完成: {len(graphs)} 个有效图")
    return graphs


def save_checkpoint(model, optimizer, history, best_reward, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_reward': best_reward,
        'model_config': {
            'in_channels': model.in_channels,
            'hidden_channels': model.hidden_channels,
            'heads': model.heads,
            'num_layers': model.num_layers,
            'M_candidates': model.M_candidates,
        }
    }, save_path)
    print(f"模型已保存: {save_path}")


def save_history(history, save_path):
    hist_path = save_path.replace('.pth', '_training_history.csv')
    with open(hist_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg_Reward'])
        writer.writerows(history)
    print(f"训练历史已保存: {hist_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Unified PPO Training for BiT-HyRL')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='合成训练数据目录 (默认: dataset/all/syn)')
    parser.add_argument('--real-data-dir', type=str, default=None,
                        help='真实网络数据目录 (默认: dataset/testdata)，设为 "" 可禁用')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--B-budget', type=int, default=3, help='拓扑微调预算')
    parser.add_argument('--K-ratio', type=float, default=0.1, help='控制器比例')
    parser.add_argument('--phase2-warmup', type=int, default=5, help='Phase 2 预热 epoch 数')
    parser.add_argument('--M-candidates', type=int, default=30, help='候选边池大小')
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--dre-dim', type=int, default=0)
    parser.add_argument('--collect-per-epoch', type=int, default=20)
    parser.add_argument('--max-graphs', type=int, default=None)
    parser.add_argument('--min-nodes', type=int, default=20)
    parser.add_argument('--max-nodes', type=int, default=100)
    parser.add_argument('--attack-ratio', type=float, default=0.15)
    parser.add_argument('--attack-methods', type=str, default='degree,betweenness,pagerank,eigenvector,random',
                        help='训练时使用的攻击方法列表，逗号分隔')
    parser.add_argument('--sample-methods', type=int, default=4,
                        help='每次奖励计算随机采样的攻击方法数 (None=全部)')
    parser.add_argument('--aggregation', type=str, default='min', choices=['min', 'mean'])
    parser.add_argument('--use-curriculum', action='store_true')
    parser.add_argument('--use-adaptive-aggregation', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--use-synthetic', action='store_true',
                        help='使用合成数据训练，避免加载大数据集')
    return parser.parse_args()


def main():
    args = parse_args()
    init_logger()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.save_path is None:
        args.save_path = os.path.join(ROOT, 'src', 'train', 'v1', 'checkpoints', 'unified_ppo_agent.pth')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("=" * 60)
    print("Unified PPO Training for BiT-HyRL")
    print("=" * 60)
    print(f"Epochs: {args.epochs}, B-budget: {args.B_budget}, K-ratio: {args.K_ratio}")
    print(f"M-candidates: {args.M_candidates}, LR: {args.lr}")
    print(f"Data: nodes {args.min_nodes}-{args.max_nodes}, collect/epoch {args.collect_per_epoch}")
    print(f"Attack methods: {args.attack_methods}, sample: {args.sample_methods}")
    print("=" * 60)

    if args.use_synthetic:
        graphs = generate_synthetic_data(
            num_graphs=args.max_graphs or 200,
            min_nodes=args.min_nodes, max_nodes=args.max_nodes,
            seed=args.seed
        )
    else:
        # 合成数据
        graphs = load_training_graphs(
            args.data_dir, max_graphs=args.max_graphs,
            min_nodes=args.min_nodes, max_nodes=args.max_nodes,
            seed=args.seed, num_workers=args.num_workers
        )
        # 真实网络数据（若存在且未禁用）
        if args.real_data_dir != "":
            real_dir = args.real_data_dir or os.path.join(ROOT, 'dataset', 'testdata')
            if os.path.exists(real_dir):
                print(f"加载真实网络数据: {real_dir}")
                real_graphs = load_training_graphs(
                    real_dir, max_graphs=None,
                    min_nodes=args.min_nodes, max_nodes=args.max_nodes,
                    seed=args.seed, num_workers=args.num_workers
                )
                print(f"合成图: {len(graphs)}, 真实图: {len(real_graphs)}")
                graphs.extend(real_graphs)
            else:
                print(f"真实网络数据目录不存在: {real_dir}，跳过")
    if not graphs:
        print("没有加载到有效图，退出。")
        return

    # 模型输入维度：Node2Vec + 节点状态(3)
    in_channels = args.embed_dim + 3

    model = TopologyPolicy(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        M_candidates=args.M_candidates,
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    trainer = UnifiedPPOTrainer(
        model, lr=args.lr, topology_lr_ratio=0.3, batch_size=32, n_epochs=4
    )

    history = []
    best_reward = -float('inf')
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        try:
            ck = torch.load(args.resume, map_location=config.DEVICE, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            trainer.optimizer.load_state_dict(ck['optimizer_state_dict'])
            history = ck.get('history', [])
            best_reward = ck.get('best_reward', -float('inf'))
            start_epoch = len(history)
            print(f"恢复训练: {args.resume}, 已训练 {start_epoch} 轮")
        except Exception as e:
            print(f"恢复失败: {e}，从头训练")

    attack_methods = [m.strip() for m in args.attack_methods.split(',') if m.strip()]

    def reward_fn(G_final, controllers, G_original, epoch=None, total_epochs=None):
        return calculate_unified_reward(
            G_final, controllers, G_original,
            attack_ratio=args.attack_ratio,
            attack_methods=attack_methods,
            aggregation=args.aggregation,
            sample_methods=args.sample_methods,
            use_curriculum=args.use_curriculum,
            use_adaptive_aggregation=args.use_adaptive_aggregation,
            epoch=epoch, total_epochs=total_epochs,
        )

    total_epochs = start_epoch + args.epochs

    for epoch in range(start_epoch, total_epochs):
        # Phase 2 预热
        if epoch < args.phase2_warmup:
            trainer.set_phase2_warmup(True)
            print(f"[Epoch {epoch+1}/{total_epochs}] Phase 2 warmup")
        else:
            if trainer.phase2_only:
                trainer.set_phase2_warmup(False)
                trainer.optimizer = trainer._build_optimizer()
                print(f"[Epoch {epoch+1}/{total_epochs}] End warmup, start joint training")

        epoch_rewards = []
        sampled = random.sample(graphs, min(args.collect_per_epoch, len(graphs)))

        for i, G in enumerate(sampled):
            if G.number_of_nodes() < 3:
                continue
            # 定期输出进度，避免后台 worker 心跳超时
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [Epoch {epoch+1}/{total_epochs}] collect {i+1}/{len(sampled)}", flush=True)
            B = args.B_budget
            K = max(1, math.ceil(G.number_of_nodes() * args.K_ratio))
            try:
                _, reward = trainer.collect_unified_trajectory(
                    G, B_budget=B, K_budget=K, reward_fn=reward_fn,
                    embed_dim=args.embed_dim, dre_dim=args.dre_dim, seed=args.seed,
                    epoch=epoch, total_epochs=total_epochs
                )
                epoch_rewards.append(reward)
            except Exception as e:
                logger.warning(f"轨迹收集失败: {e}")
                continue

        stats = trainer.update()
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        history.append([epoch + 1, avg_reward])

        if avg_reward > best_reward:
            best_reward = avg_reward
            save_checkpoint(model, trainer.optimizer, history, best_reward, args.save_path)

        print(f"Epoch {epoch+1}/{total_epochs} | Avg Reward: {avg_reward:.4f} | "
              f"Best: {best_reward:.4f} | Policy Loss: {stats.get('policy_loss', 0):.4f} | "
              f"Value Loss: {stats.get('value_loss', 0):.4f} | Entropy: {stats.get('entropy', 0):.4f}")

    save_history(history, args.save_path)
    print("训练完成。")


if __name__ == '__main__':
    main()
