#!/usr/bin/env python3
"""Paper Figure 6 (§8.3.6) -- effective rank rises with k; Sharpe peaks near k0.

Two panels: erank vs k (left, with identity reference and true-k0 marker)
and Sharpe vs k (right), both for ``z in {0, 10, 50}`` at m=1024.

Output: results/figures/fig06_erank_collapse.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import apply_paper_style, load_sweep_csv, savefig, styled_grid


P_FIXED: int = 1024
K0: int = 12


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv(
        "sweep_factor_dim.csv", "all_synthetic_datarun_results.csv",
        "synthetic_v3_sweep_A_summary.csv",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    styles = [(0, "firebrick", "^"), (10, "steelblue", "o"), (50, "green", "s")]

    ax = axes[0]
    for zv, color, marker in styles:
        s = df[(df["P"] == P_FIXED) & (df["z"] == zv)].sort_values("k")
        if not s.empty:
            ax.plot(s["k"], s["mean_erank"], color=color, marker=marker,
                    ms=8, lw=2, label=f"$z={zv}$")
    kr = np.arange(df["k"].min(), df["k"].max() + 1)
    ax.plot(kr, kr, "grey", ls="--", lw=1, alpha=0.55, label="erank = $k$")
    ax.axvline(K0, color="orange", ls="--", lw=1.8, label=f"true $k_0={K0}$")
    ax.set_xlabel("Factor dimension $k$")
    ax.set_ylabel(r"Effective rank erank($\hat\Sigma_f$)")
    ax.set_title("Effective rank vs $k$")
    ax.legend(fontsize=8); styled_grid(ax)

    ax = axes[1]
    for zv, color, marker in styles:
        s = df[(df["P"] == P_FIXED) & (df["z"] == zv)].sort_values("k")
        if not s.empty:
            ax.plot(s["k"], s["sharpe"], color=color, marker=marker,
                    ms=8, lw=2, label=f"$z={zv}$")
    ax.axvline(K0, color="orange", ls="--", lw=1.8, label=f"true $k_0={K0}$")
    ax.set_xlabel("Factor dimension $k$"); ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Sharpe vs $k$")
    ax.legend(fontsize=8); styled_grid(ax)

    fig.suptitle(r"erank rises with $k$, Sharpe peaks near $k_0$ ($\tilde m=1024$)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig06_erank_collapse.png", dpi=200)


if __name__ == "__main__":
    main()
