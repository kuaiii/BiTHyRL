# -*- coding: utf-8 -*-
"""
双峰网络构造过程诊断实验。

目的：把"构造阶段"的问题量化，而不是只看最终端到端排名。
对比 5 种构造：
  1. original       原始拓扑
  2. theory_1hub    create_bimodal_theoretical（旧默认：单超级枢纽）
  3. docx_kmin2     create_bimodal_network_docx（书中 k_min>=2 版本）
  4. exact_1hub     create_bimodal_network_exact（单 hub，带连通性/边数修正）
  5. adaptive_robust 当前默认（多 hub 候选 + 多攻击 AUC 筛选）

每个拓扑记录：
  - 约束是否保持：N / M / 连通性
  - 结构特征：k_min / k_max / 最大度占比 / hub 数 / 割点数 / 桥数 / 2-core 比例
  - 低度层自保：_low_layer_resilience_score（去掉 hub 后低度层 GCC）
  - 构造层韧性：R_degree / R_random / R_betweenness（GCC 曲线 AUC，x 轴为真实移除比例）

输出：
  results/bimodal_diagnosis/construction_diagnostics.csv
  results/bimodal_diagnosis/construction_summary.csv

用法：
  python scripts/diagnose_bimodal_construction.py
  python scripts/diagnose_bimodal_construction.py --datasets Colt,GtsCe
  python scripts/diagnose_bimodal_construction.py --skip-adaptive   # 跳过最慢的 adaptive
  python scripts/diagnose_bimodal_construction.py --seed-check      # 额外测 adaptive 的 seed 敏感性
"""
import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from src.topology.generators import load_graph
from src.topology.reconstruction import (
    create_bimodal_theoretical,
    create_bimodal_network_exact,
    create_bimodal_network_docx,
    create_bimodal_adaptive_robust,
    _attack_sequence_for_method,
    _attack_auc_lcc,
    _low_layer_resilience_score,
)

OUT_DIR = os.path.join(ROOT, "results", "bimodal_diagnosis")


def to_int_graph(G):
    """统一节点编号，保证不同构造之间可比。"""
    return nx.convert_node_labels_to_integers(G, label_attribute="orig_label")


def hub_threshold(G):
    n = G.number_of_nodes()
    degs = dict(G.degree())
    avg = sum(degs.values()) / n if n else 0.0
    return max(avg * 2.0, avg + 1.0), degs


def two_core_ratio(G):
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    try:
        return nx.k_core(G, k=2).number_of_nodes() / n
    except Exception:
        return 0.0


def gcc_ratio(G):
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return max((len(c) for c in nx.connected_components(G)), default=0) / n


def attack_curves(G, seed=42):
    """三种攻击下的 GCC 曲线 AUC（x 轴用真实移除比例，规避 attack_step 口径偏差）。"""
    out = {}
    out["R_degree"] = _attack_auc_lcc(
        G, _attack_sequence_for_method(G, "degree", seed=seed)
    )
    out["R_betweenness"] = _attack_auc_lcc(
        G, _attack_sequence_for_method(G, "betweenness", seed=seed)
    )
    random_aucs = [
        _attack_auc_lcc(G, _attack_sequence_for_method(G, "random", seed=seed + 131 * j))
        for j in range(3)
    ]
    # 与 _candidate_topology_score 一致：min 与 mean 各半，抑制随机攻击运气成分
    out["R_random"] = 0.5 * min(random_aucs) + 0.5 * float(np.mean(random_aucs))
    out["R_random_min"] = min(random_aucs)
    out["R_random_mean"] = float(np.mean(random_aucs))
    out["R_wgcc"] = 0.5 * out["R_degree"] + 0.5 * out["R_random"]
    return out


def describe(G, n0, m0, variant, dataset, elapsed=0.0):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    thr, degs = hub_threshold(G)
    hubs = [v for v, d in degs.items() if d >= thr]
    k_max = max(degs.values()) if degs else 0
    k_min = min(degs.values()) if degs else 0

    G_top1 = G.copy()
    if hubs:
        G_top1.remove_node(max(degs, key=degs.get))
    gcc_after_top1 = gcc_ratio(G_top1) if G_top1.number_of_nodes() else 0.0

    # 移除度数最高的 5% 节点后的 GCC，衡量"打掉头部即崩"的程度
    top_k = max(1, int(round(0.05 * n)))
    seq = sorted(degs, key=degs.get, reverse=True)[:top_k]
    G_top5 = G.copy()
    G_top5.remove_nodes_from(seq)
    gcc_after_top5 = gcc_ratio(G_top5) if G_top5.number_of_nodes() else 0.0

    row = {
        "dataset": dataset,
        "variant": variant,
        "n": n,
        "m": m,
        "n_ok": n == n0,
        "m_ok": m == m0,
        "connected": nx.is_connected(G) if n else False,
        "k_min": k_min,
        "k_max": k_max,
        "k_avg": round(2 * m / n, 3) if n else 0.0,
        "max_deg_share": round(k_max / (2 * m), 4) if m else 0.0,
        "num_hubs": len(hubs),
        "num_articulation": sum(1 for _ in nx.articulation_points(G)) if n else 0,
        "num_bridges": sum(1 for _ in nx.bridges(G)) if m else 0,
        "two_core_ratio": round(two_core_ratio(G), 4),
        "low_layer_score": round(_low_layer_resilience_score(G), 4),
        "gcc_after_top1": round(gcc_after_top1, 4),
        "gcc_after_top5pct": round(gcc_after_top5, 4),
        "build_sec": round(elapsed, 2),
    }
    row.update({k: round(v, 4) for k, v in attack_curves(G).items()})
    return row


def build_variants(G, args):
    """返回 [(variant_name, graph, elapsed_sec)]。"""
    n, m = G.number_of_nodes(), G.number_of_edges()
    out = [("original", G.copy(), 0.0)]

    t0 = time.time()
    out.append(("theory_1hub", create_bimodal_theoretical(n, m, seed=args.seed), time.time() - t0))

    t0 = time.time()
    try:
        res = create_bimodal_network_docx(G=G, seed=args.seed)
        G_docx = res.get("G") if isinstance(res, dict) else res
        if G_docx is not None:
            G_docx.graph["docx_meta"] = {
                "strategy": res.get("strategy") if isinstance(res, dict) else None,
                "k_min_target": res.get("k_min_target") if isinstance(res, dict) else None,
                "k_max_target": res.get("k_max_target") if isinstance(res, dict) else None,
            }
        out.append(("docx_kmin2", G_docx, time.time() - t0))
    except Exception as e:
        print(f"    [warn] docx 构造失败: {e}")
        out.append(("docx_kmin2", None, time.time() - t0))

    t0 = time.time()
    try:
        out.append(("exact_1hub", create_bimodal_network_exact(G, hub_num=1, seed=args.seed), time.time() - t0))
    except Exception as e:
        print(f"    [warn] exact 构造失败: {e}")
        out.append(("exact_1hub", None, time.time() - t0))

    t0 = time.time()
    if not args.skip_adaptive:
        try:
            G_ad = create_bimodal_adaptive_robust(
                G,
                seed=args.seed,
                max_hub_ratio=args.max_hub_ratio,
                num_samples=args.num_samples,
                attack_methods=tuple(args.attack_methods.split(",")),
                score_profile=args.profile,
                num_score_seeds=args.score_seeds,
            )
            out.append(("adaptive_robust", G_ad, time.time() - t0))
        except Exception as e:
            print(f"    [warn] adaptive 构造失败: {e}")
            out.append(("adaptive_robust", None, time.time() - t0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--skip-adaptive", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hub-ratio", type=float, default=0.25)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--score-seeds", type=int, default=3)
    parser.add_argument("--profile", type=str, default="balanced")
    parser.add_argument("--attack-methods", type=str, default="degree,betweenness,random")
    parser.add_argument("--seed-check", action="store_true",
                        help="额外验证 adaptive 对随机种子的敏感性")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    data_dir = os.path.join(ROOT, "dataset", "testdata")
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = sorted(p[:-4] for p in os.listdir(data_dir) if p.endswith(".gml"))

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    seed_rows = []

    for ds in datasets:
        path = os.path.join(data_dir, f"{ds}.gml")
        if not os.path.exists(path):
            print(f"[skip] 未找到 {ds}.gml")
            continue
        G_raw, _ = load_graph(path)
        G0 = to_int_graph(G_raw)
        n0, m0 = G0.number_of_nodes(), G0.number_of_edges()
        print(f"\n=== {ds}  N={n0}  M={m0}  k_avg={2 * m0 / n0:.2f} ===")

        for name, Gv, elapsed in build_variants(G0, args):
            if Gv is None:
                continue
            Gv = to_int_graph(Gv)
            row = describe(Gv, n0, m0, name, ds, elapsed)
            meta_ad = Gv.graph.get("adaptive_bimodal", {}) or {}
            meta_docx = Gv.graph.get("docx_meta", {}) or {}
            row["note"] = (
                f"hub_num={meta_ad.get('best_hub_num')}" if meta_ad else
                (f"strategy={meta_docx.get('strategy')},k_min_t={meta_docx.get('k_min_target')}"
                 if meta_docx else "")
            )
            rows.append(row)
            flag = "" if (row["n_ok"] and row["m_ok"] and row["connected"]) else "  <-- 约束未保持"
            print(
                f"  {name:<16} N/M={row['n']}/{row['m']} conn={row['connected']} "
                f"k_min={row['k_min']:<3} k_max={row['k_max']:<4} "
                f"maxshare={row['max_deg_share']:.3f} low={row['low_layer_score']:.3f} "
                f"R_deg={row['R_degree']:.4f} R_rnd={row['R_random']:.4f} "
                f"R_btw={row['R_betweenness']:.4f} ({row['build_sec']}s){flag}"
            )

        if args.seed_check and not args.skip_adaptive:
            for s in (0, 1, 2, 3, 4):
                t0 = time.time()
                Gs = create_bimodal_adaptive_robust(
                    G0, seed=s,
                    max_hub_ratio=args.max_hub_ratio,
                    num_samples=args.num_samples,
                    attack_methods=tuple(args.attack_methods.split(",")),
                    score_profile=args.profile,
                    num_score_seeds=args.score_seeds,
                )
                meta = Gs.graph.get("adaptive_bimodal", {})
                Gs = to_int_graph(Gs)
                rec = describe(Gs, n0, m0, f"adaptive_seed{s}", ds, time.time() - t0)
                rec["hub_num_selected"] = meta.get("best_hub_num")
                seed_rows.append(rec)
                print(
                    f"  seed={s} hub_num={meta.get('best_hub_num')} "
                    f"R_deg={rec['R_degree']:.4f} R_rnd={rec['R_random']:.4f} "
                    f"R_wgcc={rec['R_wgcc']:.4f}"
                )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "construction_diagnostics.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 汇总：各构造方式在数据集上的均值，以及相对原始图的增益
    metric_cols = ["R_degree", "R_random", "R_wgcc", "R_betweenness",
                   "low_layer_score", "two_core_ratio", "max_deg_share",
                   "gcc_after_top1", "gcc_after_top5pct", "build_sec"]
    summary = df.groupby("variant")[metric_cols].mean().round(4)
    summary["n_datasets"] = df.groupby("variant")["dataset"].nunique()
    summary["constraint_violations"] = df.groupby("variant").apply(
        lambda d: int((~d["n_ok"] | ~d["m_ok"] | ~d["connected"]).sum())
    )
    order = ["original", "theory_1hub", "docx_kmin2", "exact_1hub", "adaptive_robust"]
    summary = summary.reindex([v for v in order if v in summary.index])
    summary_csv = os.path.join(args.out_dir, "construction_summary.csv")
    summary.to_csv(summary_csv, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("汇总（各数据集均值）")
    print("=" * 100)
    print(summary.to_string())
    print(f"\n明细: {out_csv}\n汇总: {summary_csv}")

    if seed_rows:
        sdf = pd.DataFrame(seed_rows)
        scsv = os.path.join(args.out_dir, "adaptive_seed_sensitivity.csv")
        sdf.to_csv(scsv, index=False, encoding="utf-8-sig")
        agg = sdf.groupby("dataset")[["R_degree", "R_random", "R_wgcc"]].agg(["mean", "std"]).round(4)
        print("\n" + "=" * 100)
        print("adaptive_robust 的 seed 敏感性（5 个 seed 的均值/标准差）")
        print("=" * 100)
        print(agg.to_string())
        print(f"\n明细: {scsv}")


if __name__ == "__main__":
    main()
