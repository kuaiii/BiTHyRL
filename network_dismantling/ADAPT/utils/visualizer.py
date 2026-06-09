"""
Plotting utilities for dismantling curves.
"""
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_dismantling_curves(
    curves: Dict[str, List[Tuple[float, float]]],
    title: str = "Dismantling Curves",
    save_path: str = "dismantling_curves.png",
):
    """
    Plot GCC ratio vs fraction of removed nodes for multiple methods.

    Parameters
    ----------
    curves : dict
        Mapping method_name -> list of (q, sigma(q)).
    """
    plt.figure(figsize=(8, 6))
    for method_name, curve in curves.items():
        if not curve:
            continue
        qs = [c[0] for c in curve]
        sigmas = [c[1] for c in curve]
        plt.plot(qs, sigmas, marker="o", markersize=3, label=method_name, alpha=0.8)

    plt.xlabel("Fraction of removed nodes $q$")
    plt.ylabel("Giant Component Size Ratio $\\sigma(q)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")
