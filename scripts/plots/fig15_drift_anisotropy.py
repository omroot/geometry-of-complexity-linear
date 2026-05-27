#!/usr/bin/env python3
"""Paper Figure 15 (§9.7) -- historical drift anisotropy vs Sharpe.

rho_psi = bar psi / psi_max vs annualised Sharpe across all 88 historical
configurations; OLS fit + 95% CI on the virtuous (z>0) subset. Paper
reports within-virtuous r = -0.52.

Output: results/figures/fig15_drift_anisotropy.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import (
    apply_paper_style, load_sweep_csv, plot_ols_with_ci, savefig, styled_grid,
)


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv("real_baseline_grid.csv", "all_rff_results.csv").dropna(
        subset=["mean_anisotropy", "sharpe"]
    )
    virt = df[df["z"] > 0]
    ills = df[df["z"] == 0]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(virt["mean_anisotropy"], virt["sharpe"], c="steelblue", marker="o",
               s=55, alpha=0.75, edgecolors="k", linewidth=0.4,
               label=f"Virtuous ($z>0$, $n={len(virt)}$)")
    ax.scatter(ills["mean_anisotropy"], ills["sharpe"], c="firebrick", marker="^",
               s=55, alpha=0.75, edgecolors="k", linewidth=0.4,
               label=f"Illusory ($z=0$, $n={len(ills)}$)")
    _, _, r = plot_ols_with_ci(ax, virt["mean_anisotropy"].values, virt["sharpe"].values)

    handles, labels = ax.get_legend_handles_labels()
    labels = [
        (f"OLS fit on virtuous, $r = {r:.2f}$ (95% CI)" if "CI" in lbl and r is not None else lbl)
        for lbl in labels
    ]
    ax.legend(handles, labels, loc="lower left", fontsize=9, framealpha=0.95)
    ax.text(0.02, 0.98, r"low $\rho_\psi$ $\Rightarrow$ concentrated drift",
            fontsize=10, transform=ax.transAxes, ha="left", va="top",
            style="italic", color="dimgrey")
    ax.set_xlabel(r"Drift anisotropy $\rho_\psi = \bar\psi / \psi_{\max}$")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Drift anisotropy vs Sharpe -- historical")
    styled_grid(ax)
    fig.tight_layout()
    savefig(fig, "fig15_drift_anisotropy.pdf")


if __name__ == "__main__":
    main()
