#!/usr/bin/env python3
"""Paper Figure 2 (§8.3.2) -- paired regime separation at m=1024.

Two panels of d_proj (left) and Sharpe (right) versus factor dim k,
contrasting z=0 (illusory) with z=10 (virtuous). True synthetic rank k0=12.

Output: results/figures/fig02_paired_regime.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
    s0 = df[(df["P"] == P_FIXED) & (df["z"] == 0)].sort_values("k")
    s10 = df[(df["P"] == P_FIXED) & (df["z"] == 10)].sort_values("k")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(s0["k"], s0["mean_subspace_stability"], "r-^", ms=8, lw=2, label="$z=0$ (illusory)")
    ax.plot(s10["k"], s10["mean_subspace_stability"], "b-o", ms=8, lw=2, label="$z=10$ (virtuous)")
    ax.axvline(K0, color="orange", ls="--", lw=1.8, label=f"true $k_0={K0}$")
    ax.set_xlabel("Factor dimension $k$")
    ax.set_ylabel(r"Subspace stability $d_{\mathrm{proj}}$")
    ax.set_title(r"$d_{\mathrm{proj}}$: frozen vs active")
    ax.legend(fontsize=9)
    styled_grid(ax)

    ax = axes[1]
    ax.plot(s0["k"], s0["sharpe"], "r-^", ms=8, lw=2, label="$z=0$ (illusory)")
    ax.plot(s10["k"], s10["sharpe"], "b-o", ms=8, lw=2, label="$z=10$ (virtuous)")
    ax.axvline(K0, color="orange", ls="--", lw=1.8, label=f"true $k_0={K0}$")
    ax.set_xlabel("Factor dimension $k$")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Sharpe: illusory vs virtuous")
    ax.legend(fontsize=9)
    styled_grid(ax)

    fig.suptitle(r"Matched pairs at $\tilde m=1024$ -- regime separation", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig02_paired_regime.png", dpi=200)


if __name__ == "__main__":
    main()
