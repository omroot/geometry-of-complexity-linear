#!/usr/bin/env python3
"""Paper Figure 14 (§9.6) -- historical spectral-gap cliff.

Plots ``lambda_k / lambda_1`` vs fitted factor dim k at m=1024 for four
ridge values; highlights the empirical signal boundary band (k0 ~ 16-20).

Output: results/figures/fig14_spectral_cliff.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import apply_paper_style, load_sweep_csv, savefig, styled_grid


P_FIXED: int = 1024


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv("real_baseline_grid.csv", "all_rff_results.csv")

    fig, ax = plt.subplots(figsize=(9, 5.8))
    styles = [(0, "firebrick", "o", "-"), (10, "steelblue", "s", "-"),
              (20, "green", "^", "--"), (50, "purple", "D", ":")]
    for zv, color, marker, ls in styles:
        s = df[(df["P"] == P_FIXED) & (df["z"] == zv)].sort_values("k")
        s = s.groupby("k", as_index=False).mean(numeric_only=True)
        if len(s) > 1:
            ax.plot(s["k"], s["mean_spectral_gap"], color=color, marker=marker,
                    ms=8, lw=2, ls=ls, label=f"$z={zv}$")
    ax.axvspan(16, 24, alpha=0.12, color="green", label=r"signal-boundary zone $k_0\!\approx\!16$-$20$")
    ax.set_xlabel("Factor dimension $k$"); ax.set_ylabel(r"Spectral gap $\lambda_k / \lambda_1$")
    ax.set_title(r"Spectral-gap cliff -- historical ($\tilde m=1024$)")
    ax.legend(fontsize=9); styled_grid(ax)
    fig.tight_layout()
    savefig(fig, "fig14_spectral_cliff.pdf")


if __name__ == "__main__":
    main()
