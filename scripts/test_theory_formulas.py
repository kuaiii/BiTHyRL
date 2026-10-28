# -*- coding: utf-8 -*-
"""自查：书中 7.67/7.69 的最优标度 k_max* = A r^(-2/3) 是否能被我们的解析实现复现。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.theory_range import fc_random, fc_random_kappa, ftot  # noqa: E402

A_TH = (2 * 9 * 4 / 5) ** (1 / 3)
print(f"A_theory(<k>=3) = {A_TH:.4f}")

print("\n[r]  k_max*   k*r^(2/3)   f_tot*   f_rand   f_target  case")
for r in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4]:
    best = (-9.0, 0.0)
    for km in np.linspace(2, 30 / r ** (2 / 3), 4000):
        f = ftot(3.0, float(r), float(km))["f_tot"]
        if f > best[0]:
            best = (f, float(km))
    k = best[1]
    d = ftot(3.0, float(r), k)
    print(f"{r:8.5f} {k:9.2f} {k*r**(2/3):10.3f} {best[0]:8.4f} {d['f_rand']:8.4f} "
          f"{d['f_target']:9.4f} {int(d['f_case']):5d}")

print("\n[check] 7.64 与 1-1/(kappa-1) 的差异（两者不是同一个量）")
for r, km in [(0.01, 50.0), (0.001, 240.0)]:
    print(f"  r={r} k_max={km}: f_7.64={fc_random(3.0, r, km):.4f}  "
          f"f_kappa={fc_random_kappa(3.0, r, km):.4f}  "
          f"diff={fc_random(3.0, r, km)-fc_random_kappa(3.0, r, km):+.4f}")

print("\n[check] r->0 时 f_tot -> 2-1/(<k>-1) =", 2 - 1 / 2)
