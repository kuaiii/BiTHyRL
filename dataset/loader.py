"""
Unified dataset loader for network dismantling experiments.

Organizes datasets into:
    dataset/{train,validate,test}/{syn,real}/

Supports formats: .gml, .graphml, .gt, .gt.xz
"""
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "dataset"


def _dataset_dir(split: str, category: str) -> Path:
    """Return path to a dataset split/category directory."""
    return _DATASET_ROOT / split / category


def _decompress_xz(path: Path) -> Path:
    """Decompress .xz to a temporary file and return the path."""
    if not str(path).endswith(".xz"):
        return path
    # Use xz to decompress to a temp file in the same directory
    tmp_path = path.with_suffix("")  # remove .xz
    # If the decompressed file already exists, just return it
    if tmp_path.exists():
        return tmp_path
    subprocess.run(["xz", "-dk", str(path)], check=True)
    return tmp_path


def load_graph(path: Path) -> Optional[nx.Graph]:
    """Load a single graph file (gml, graphml, gt, gt.xz)."""
    path = Path(path)
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    name = path.name.lower()

    try:
        if name.endswith(".gt.xz"):
            decompressed = _decompress_xz(path)
            import graph_tool.all as gt
            g = gt.load_graph(str(decompressed))
            # Convert graph-tool graph to NetworkX
            G = nx.Graph()
            G.add_nodes_from(range(int(g.num_vertices())))
            for e in g.edges():
                G.add_edge(int(e.source()), int(e.target()))
            if not decompressed.samefile(path):
                # Remove temporary decompressed file if it was created here
                pass  # keep it for future use
            return G
        elif suffix == ".gml":
            try:
                G = nx.read_gml(str(path), label="id")
            except Exception:
                G = nx.read_gml(str(path))
            return nx.Graph(G)
        elif suffix == ".graphml":
            G = nx.read_graphml(str(path))
            return nx.Graph(G)
        elif suffix == ".gt":
            import graph_tool.all as gt
            g = gt.load_graph(str(path))
            G = nx.Graph()
            G.add_nodes_from(range(int(g.num_vertices())))
            for e in g.edges():
                G.add_edge(int(e.source()), int(e.target()))
            return G
        elif suffix == ".txt" or suffix == ".edgelist":
            G = nx.read_edgelist(str(path))
            return nx.Graph(G)
    except Exception as e:
        print(f"  Warning: failed to load {path}: {e}")
        return None

    return None


def _process_file(
    fpath: Path,
    min_nodes: int,
    max_nodes: int,
    ensure_connected: bool,
) -> Optional[Tuple[str, nx.Graph]]:
    """Helper for parallel loading: load and filter a single graph file."""
    G = load_graph(fpath)
    if G is None:
        return None
    if ensure_connected and not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    if min_nodes <= n <= max_nodes:
        name = fpath.stem.replace(".gt", "").replace(".xz", "")
        return name, G
    return None


def load_split(
    split: str,
    category: str,
    min_nodes: int = 10,
    max_nodes: int = 100000,
    ensure_connected: bool = True,
    num_workers: int = 4,
) -> Dict[str, nx.Graph]:
    """
    Load all networks from a given split and category.

    Args:
        split: 'train', 'validate', or 'test'
        category: 'syn' or 'real'
        min_nodes: minimum number of nodes to include
        max_nodes: maximum number of nodes to include
        ensure_connected: if True, extract largest connected component
        num_workers: number of parallel workers for loading (0 = sequential)

    Returns:
        Dict mapping network name (filename without extension) to nx.Graph
    """
    d = _dataset_dir(split, category)
    networks = {}
    if not d.exists():
        return networks

    file_paths = [fpath for fpath in sorted(d.iterdir()) if fpath.is_file()]

    if num_workers <= 1 or len(file_paths) < 50:
        # Sequential loading for small batches
        for fpath in file_paths:
            result = _process_file(fpath, min_nodes, max_nodes, ensure_connected)
            if result is not None:
                networks[result[0]] = result[1]
    else:
        # Parallel loading for large datasets
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_file, fpath, min_nodes, max_nodes, ensure_connected): fpath
                for fpath in file_paths
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    networks[result[0]] = result[1]
    return networks


def load_network(name: str, split: Optional[str] = None, category: Optional[str] = None) -> Optional[nx.Graph]:
    """
    Load a single network by name. Searches all splits/categories if not specified.
    """
    if split and category:
        paths = [_dataset_dir(split, category) / name]
        # Try common extensions
        paths = [p.with_suffix(s) for p in paths for s in [".gml", ".graphml", ".gt", ".gt.xz", ".txt", ".edgelist"]]
        for p in paths:
            G = load_graph(p)
            if G is not None:
                return G
        return None

    for sp in ["train", "validate", "test"]:
        for cat in ["syn", "real"]:
            G = load_network(name, split=sp, category=cat)
            if G is not None:
                return G
    return None


def generate_synthetic_network(
    net_name: str,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate a standard synthetic test network on the fly.
    This is provided as a fallback; prefer pre-generated datasets.
    """
    if net_name.startswith("BA_"):
        n = int(net_name.split("_")[1])
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    elif net_name.startswith("ER_"):
        n = int(net_name.split("_")[1])
        p = 8.0 / (n - 1)
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
        return nx.convert_node_labels_to_integers(G)
    elif net_name.startswith("WS_"):
        n = int(net_name.split("_")[1])
        return nx.watts_strogatz_graph(n, 8, 0.3, seed=seed)
    elif net_name.startswith("complete_"):
        n = int(net_name.split("_")[1])
        return nx.complete_graph(n)
    else:
        raise ValueError(f"Unknown synthetic network: {net_name}")


def list_networks(split: Optional[str] = None, category: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    List all available networks.
    Returns list of (name, split, category).
    """
    results = []
    splits = [split] if split else ["train", "validate", "test"]
    categories = [category] if category else ["syn", "real"]
    for sp in splits:
        for cat in categories:
            d = _dataset_dir(sp, cat)
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.is_file():
                    results.append((f.stem.replace(".gt", "").replace(".xz", ""), sp, cat))
    return results
