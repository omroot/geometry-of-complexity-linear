#!/usr/bin/env python3
"""Paper Figure 5 (§8.3.5) -- spectral gap cliff at the true rank k0.

Plots the per-window factor-covariance spectral gap ``lambda_k / lambda_1``
versus the fitted factor dimension k for three ridge values, with the cliff
band from k=k0 to k=16 shaded.

Output: results/figures/fig05_spectral_cliff.png
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

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.axvspan(K0, 16, alpha=0.22, color="orange",
               label=fr"cliff: $k_0={K0}\to k=16$ (gap $\to 0$)")
    for zv, color, marker, ls in [
        (0, "firebrick", "o", "-"),
        (10, "steelblue", "s", "-"),
        (50, "purple", "^", ":"),
    ]:
        s = df[(df["P"] == P_FIXED) & (df["z"] == zv)].sort_values("k")
        if not s.empty:
            ax.plot(s["k"], s["mean_spectral_gap"],
                    color=color, marker=marker, ms=8, lw=2.2, ls=ls,
                    label=f"$z={zv}$")
    ax.set_xlabel("Factor dimension $k$")
    ax.set_ylabel(r"Spectral gap $\lambda_k / \lambda_1$")
    ax.set_title(
        r"Spectral gap cliff: collapse just past the true rank ($\tilde m=1024$)"
    )
    ax.legend(fontsize=9, loc="upper right")
    styled_grid(ax)
    fig.tight_layout()
    savefig(fig, "fig05_spectral_cliff.png", dpi=200)


if __name__ == "__main__":
    main()
