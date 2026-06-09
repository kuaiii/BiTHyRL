"""
Networkx wrapper for CI (Collective Influence) dismantler.
Calls the precompiled CI.exe on Windows.
"""
import os
import tempfile
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np

# Cross-platform CI binary resolution
_CI_DIR = Path(__file__).parent
if (_CI_DIR / "CI").exists() and os.access(_CI_DIR / "CI", os.X_OK):
    CI_EXE = _CI_DIR / "CI"
else:
    CI_EXE = _CI_DIR / "CI.exe"


def _collective_influence_python(G: nx.Graph, l: int = 2, stop_condition: int = 1) -> List[int]:
    """Pure Python implementation of Collective Influence as fallback."""
    G = G.copy()
    seq = []
    while G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        ci_scores = {}
        for node in G.nodes():
            ball = nx.single_source_shortest_path_length(G, node, cutoff=l)
            ci = (degrees[node] - 1) * sum((degrees.get(j, 0) - 1) for j in ball if j != node)
            ci_scores[node] = ci
        if not ci_scores:
            break
        best = max(ci_scores, key=ci_scores.get)
        seq.append(best)
        G.remove_node(best)
        if G.number_of_nodes() == 0:
            break
        lcc = max(len(c) for c in nx.connected_components(G))
        if lcc <= stop_condition:
            break
    return seq


def ci_dismantle_nx(G: nx.Graph, l: int = 2, stop_condition: int = 1) -> List[int]:
    """
    Run CI dismantler on a networkx graph.
    Returns the dismantling sequence (node indices aligned with input G).
    Falls back to pure Python implementation if binary is unavailable.
    """
    # Try binary first
    use_binary = CI_EXE.exists() and str(CI_EXE).endswith("CI")
    if use_binary:
        try:
            # Ensure nodes are 0..n-1
            mapping = {node: i for i, node in enumerate(G.nodes())}
            reverse_mapping = {i: node for node, i in mapping.items()}
            G = nx.relabel_nodes(G, mapping)
            
            network_fd, network_path = tempfile.mkstemp(suffix=".txt")
            output_fd, output_path = tempfile.mkstemp(suffix=".txt")
            
            try:
                with open(network_fd, "w") as f:
                    for node in sorted(G.nodes()):
                        neighbors = sorted(G.neighbors(node))
                        line = f"{node + 1}"
                        for nb in neighbors:
                            line += f" {nb + 1}"
                        f.write(line + "\n")
                
                import subprocess
                result = subprocess.run(
                    [str(CI_EXE), network_path, str(l), str(stop_condition), output_path],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    nodes = []
                    with open(output_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                node = int(parts[1]) - 1
                                nodes.append(node)
                    return [reverse_mapping[n] for n in nodes]
            finally:
                try:
                    os.close(network_fd)
                    os.remove(network_path)
                except Exception:
                    pass
                try:
                    os.close(output_fd)
                    os.remove(output_path)
                except Exception:
                    pass
        except Exception:
            pass  # fallback to Python
    
    # Pure Python fallback
    return _collective_influence_python(G, l=l, stop_condition=stop_condition)
