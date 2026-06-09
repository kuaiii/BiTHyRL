"""
Pre-generate synthetic network datasets and save them to disk.

Usage:
    python dataset/generate_datasets.py --split train --count 500
    python dataset/generate_datasets.py --split test --count 100
    python dataset/generate_datasets.py --split all --category syn --count 300 \
        --sizes 20 50 100 150 200 300 500 800 --seed 42
"""
import argparse
from pathlib import Path

import networkx as nx

_PROJECT_ROOT = Path(__file__).parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "dataset"


def _dataset_dir(split: str, category: str) -> Path:
    """Return path to a dataset split/category directory."""
    return _DATASET_ROOT / split / category


def generate_ba(n: int, m: int = 4, seed: int = 0) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=seed)


def generate_er(n: int, p: float = None, seed: int = 0) -> nx.Graph:
    if p is None:
        p = 8.0 / (n - 1)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return nx.convert_node_labels_to_integers(G)


def generate_ws(n: int, k: int = 8, p: float = 0.3, seed: int = 0) -> nx.Graph:
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def generate_powerlaw_cluster(n: int, m: int = 4, p: float = 0.3, seed: int = 0) -> nx.Graph:
    """Holme-Kim powerlaw cluster graph: BA + tunable clustering."""
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def generate_random_geometric(n: int, radius: float = None, seed: int = 0) -> nx.Graph:
    """Random geometric graph in unit square."""
    if radius is None:
        # heuristic radius for connectivity
        radius = min(0.4, 2.0 / (n ** 0.5))
    G = nx.random_geometric_graph(n, radius, seed=seed)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return nx.convert_node_labels_to_integers(G)


def generate_complete(n: int) -> nx.Graph:
    return nx.complete_graph(n)


def save_graph(G: nx.Graph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(G, str(path))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic network datasets")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validate", "test", "all"])
    parser.add_argument("--category", type=str, default="syn", choices=["syn", "real"])
    parser.add_argument("--count", type=int, default=100, help="Number of instances per topology/size")
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = _dataset_dir(args.split, args.category)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for size in args.sizes:
        for i in range(args.count):
            seed = args.seed + i + size * 10000
            # 1. BA (standard)
            G = generate_ba(size, m=4, seed=seed)
            save_graph(G, out_dir / f"BA_n{size}_i{i}.gml")
            # 2. BA-dense (higher m)
            G = generate_ba(size, m=6, seed=seed + 100000)
            save_graph(G, out_dir / f"BAdense_n{size}_i{i}.gml")
            # 3. ER
            G = generate_er(size, seed=seed + 1)
            save_graph(G, out_dir / f"ER_n{size}_i{i}.gml")
            # 4. WS (standard p=0.3)
            G = generate_ws(size, k=8, p=0.3, seed=seed + 2)
            save_graph(G, out_dir / f"WS_n{size}_i{i}.gml")
            # 5. WS-low (p=0.1, high clustering)
            G = generate_ws(size, k=8, p=0.1, seed=seed + 3)
            save_graph(G, out_dir / f"WSlow_n{size}_i{i}.gml")
            # 6. WS-high (p=0.5, near-random)
            G = generate_ws(size, k=8, p=0.5, seed=seed + 4)
            save_graph(G, out_dir / f"WShigh_n{size}_i{i}.gml")
            # 7. Powerlaw cluster (Holme-Kim)
            if size >= 10:
                G = generate_powerlaw_cluster(size, m=4, p=0.3, seed=seed + 5)
                save_graph(G, out_dir / f"HK_n{size}_i{i}.gml")
            # 8. Random geometric
            if size >= 10:
                G = generate_random_geometric(size, seed=seed + 6)
                save_graph(G, out_dir / f"RG_n{size}_i{i}.gml")
            total += 1

    n_topos = 8 if args.sizes[0] >= 10 else 6
    print(f"Generated {args.count * len(args.sizes) * n_topos} graphs in {out_dir}")


if __name__ == "__main__":
    main()
