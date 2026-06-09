# BiT-HyRL Project Convention

Project-scope skill for the **BiT-HyRL** (Bi-modal Topology Hybrid Reinforcement Learning) codebase.
This document defines the canonical directory layout, module responsibilities, and coding conventions that every agent must follow when working on this project.

## 1. Project Overview

BiT-HyRL is a research framework for analyzing SDN (Software-Defined Network) topology resilience under various attack scenarios and controller placement strategies.

- **Core contribution**: Bimodal topology construction + GAT-based RL controller placement.
- **Stack**: Python 3.9+, PyTorch, PyTorch Geometric, NetworkX, Node2Vec, Matplotlib.
- **Entry points**: `main.py` (full comparison), `main_bit_hyrl.py` (BiT-HyRL only).

## 2. Directory Layout (Canonical)

The root directory **must** contain only the following essential files:

```
BiT-HyRL/
├── .gitignore
├── README.md
├── BiT-HyRL.md
├── requirements.txt
├── main.py                 # Main experiment entry
├── main_bit_hyrl.py        # BiT-HyRL standalone entry
│
├── dataset/                # Datasets (read-only after generation)
│   ├── all/
│   │   ├── real/           # ~260 real-world GML networks
│   │   └── syn/            # ~1000 synthetic BA networks
│   ├── testdata/           # Small test subsets
│   ├── generate_datasets.py
│   └── loader.py
│
├── docs/                   # Documentation & guides
│   ├── *.md
│   └── ...
│
├── logs/                   # Runtime logs (*.log)
│
├── network_construction/   # Topology reconstruction / optimisation algorithms
│   ├── __init__.py
│   ├── base.py             # NetworkReconstructionAlgorithm base class
│   ├── unified_interface.py# Unified construct(G, algorithm, ...) API
│   ├── random_attack.py
│   ├── high_degree_attack.py
│   ├── Daimo/
│   ├── FRED_ABL/
│   ├── ONION/
│   ├── QDLM/
│   ├── ROMEN/
│   ├── UNITY/
│   ├── other/              # Q_Robust, smartTRO, TEAM, wrappers
│   └── utils/              # attack_runner, metrics, iot_generator
│
├── network_dismantling/    # Attack / dismantling algorithms
│   ├── unified_interface.py# Unified dismantle(G, method, ...) API
│   ├── ADAPT/              # RL-based dismantling (PPO + GAT)
│   ├── CI/, CoreHD/, EGND/, EI/, FINDER_ND/, GDM/, GND/
│   ├── brute_force/, common/, heuristics/
│   ├── multiscale_entanglement/, vertex_entanglement/
│   └── ...
│
├── results/                # Experimental results (by dataset / attack / metric)
│
├── scripts/                # Auxiliary scripts, training entry points, tests
│   ├── train_*.py
│   ├── evaluate_*.py
│   ├── generate_*.py
│   ├── test_*.py
│   ├── compare_*.py
│   └── *.bat / *.sh
│
└── src/                    # Core source code
    ├── __init__.py
    ├── bit_hyrl/           # BiT-HyRL core (GAT, PPO, reward, selection)
    ├── controller/         # Controller placement strategies & manager
    ├── metrics/            # GCC, CSA, CCE, WCP, R-value calculations
    ├── simulation/         # Attack simulation (dismantling.py)
    ├── topology/           # Topology generators & BiT-HyRL reconstruction
    ├── train/              # Training artefacts per strategy/version
    │   └── [v1]/
    │       ├── checkpoints/# *.pth model weights
    │       └── *.png       # Training curves
    ├── utils/              # I/O, logging, GPU utilities
    │   └── visualization/  # Plots, result_plotter, training_history
    └── results/            # Aggregated result tables / figures
```

### 2.1 Rules

- **No `__pycache__/` in Git**. Always delete `__pycache__` directories after refactoring.
- **Root must stay lean**. Any new top-level script that is not a primary entry point belongs in `scripts/`.
- **One algorithm → one folder** inside `network_construction/` or `network_dismantling/`.
- **New reconstruction algorithms** must register themselves in `network_construction/unified_interface.py` via `@register_algorithm(name)`.

## 3. Model Weights & Training Checkpoints

All PyTorch model weights (`.pth`, `.pt`) **must** reside under:

```
src/train/[version]/checkpoints/
```

- `[version]` is a folder like `v1`, `curriculum`, `deep_gat`, etc.
- Training curves (`.png`) and CSV logs go in the parent `[version]/` folder, **not** inside `checkpoints/`.
- **Never** put `.pth` files directly under `src/bit_hyrl/` or the project root.
- If you create a new training strategy, create a new `src/train/[strategy]/` directory with its own `checkpoints/` sub-folder.

### 3.1 Updating Default Paths

When a new checkpoint directory is introduced, update the following locations:

1. `src/bit_hyrl/config.py` → `MODEL_DIR`
2. Any hard-coded defaults in `scripts/train_*.py`, `scripts/evaluate_*.py`, `main.py`
3. Docstrings that show CLI examples (`python main.py --model ...`)

## 4. Adding a New Reconstruction Algorithm

1. Create a sub-directory under `network_construction/` (e.g. `network_construction/MyAlgo/`).
2. Implement the core logic inside that directory.
3. Expose a thin wrapper in `network_construction/unified_interface.py`:

```python
@register_algorithm("MyAlgo")
def _my_algo(G: nx.Graph, param: int = 10, **kwargs) -> Dict[str, Any]:
    from .MyAlgo.my_algo import MyAlgoOptimizer
    opt = MyAlgoOptimizer(param=param)
    G_opt = opt.optimize(G)
    added = list(set(G_opt.edges()) - set(G.edges()))
    return {"G_constructed": G_opt, "added_edges": added, "algorithm": "MyAlgo"}
```

4. Import gracefully with `try/except` so that a missing dependency does not break the whole package.

## 5. Adding a New Dismantling Algorithm

1. Create a sub-directory under `network_dismantling/`.
2. Implement the dismantler so it accepts `nx.Graph` and returns a **list of node IDs** in removal order.
3. Register it in `network_dismantling/unified_interface.py`:

```python
@register_method("my_method")
def _my_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    ...
```

## 6. Python Import Conventions

- Use **absolute imports** starting from the project root (e.g. `from src.bit_hyrl.gnn_model import GATPolicy`).
- Scripts placed in `scripts/` must add the project root to `sys.path` if they are executed directly:

```python
import os, sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
```

- Do **not** use relative imports across top-level packages (e.g. `from ..src import xxx`).

## 7. Code Style

- UTF-8 encoding header: `# -*- coding: utf-8 -*-`
- Docstrings in Chinese for internal research code, English for public APIs.
- Type hints are encouraged for new functions (`nx.Graph`, `List[int]`, `Dict[str, Any]`).
- Keep functions under 60 lines when possible; split large training loops into helper methods.

## 8. Common Pitfalls

- `src/visualization/` **no longer exists**. It was merged into `src/utils/visualization/`. Update any stale imports.
- `src/training/` was removed because it was unused. The canonical training module path is now `src/train/` (for artefacts) and `src/bit_hyrl/training.py` (for training logic).
- `network_construction/other/wrappers.py` still contains legacy relative-import errors. Prefer using `network_construction.unified_interface.construct()` instead.

## 9. Quick Reference

| Task | Where to put it |
|------|-----------------|
| New topology optimizer | `network_construction/MyAlgo/` + register in `unified_interface.py` |
| New attack dismantler | `network_dismantling/MyDismantler/` + register in `unified_interface.py` |
| New training script | `scripts/train_my_strategy.py` |
| New model checkpoint | `src/train/[version]/checkpoints/` |
| New evaluation / plot script | `scripts/` |
| Core model change (GAT, reward) | `src/bit_hyrl/` |
| Controller strategy change | `src/controller/` |
| Metric computation change | `src/metrics/` |
