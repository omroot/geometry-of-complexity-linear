#!/usr/bin/env python3
"""Paper Figure 1 (§8.3.1) -- synthetic drift anisotropy vs Sharpe.

Reads the combined broad-grid CSV (sweeps A+B+C+D) or, falling back, the
legacy ``all_synthetic_datarun_results.csv``. Plots rho_psi vs annualised
Sharpe for every (k, z, m) cell, with an OLS line + 95% CI band on the
virtuous (z > 0) configurations.

Output: results/figures/fig01_drift_anisotropy.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import (
    apply_paper_style, load_sweep_csv, plot_ols_with_ci, savefig, styled_grid,
)


def _load() -> pd.DataFrame:
    """Try the new combined CSV; fall back to legacy or concat per-sweep."""
    try:
        return load_sweep_csv("all_synthetic_datarun_results.csv")
    except FileNotFoundError:
        pass
    # Combine the four new per-sweep CSVs.
    pieces = []
    for name in ("sweep_factor_dim.csv", "sweep_pgrid.csv", "sweep_ridge.csv", "sweep_rff_dim.csv"):
        try:
            pieces.append(load_sweep_csv(name))
        except FileNotFoundError:
            continue
    if not pieces:
        raise FileNotFoundError(
            "Need one of: all_synthetic_datarun_results.csv OR any of "
            "sweep_factor_dim.csv, sweep_pgrid.csv, sweep_ridge.csv, sweep_rff_dim.csv"
        )
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    apply_paper_style()
    df = _load().dropna(subset=["mean_anisotropy", "sharpe"])
    virt = df[df["z"] > 0]
    ills = df[df["z"] == 0]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(virt["mean_anisotropy"], virt["sharpe"], c="steelblue", marker="o",
               s=60, alpha=0.75, edgecolors="k", linewidth=0.4,
               label=f"Virtuous ($z>0$, $n={len(virt)}$)")
    ax.scatter(ills["mean_anisotropy"], ills["sharpe"], c="firebrick", marker="^",
               s=60, alpha=0.75, edgecolors="k", linewidth=0.4,
               label=f"Illusory ($z=0$, $n={len(ills)}$)")
    _, _, r = plot_ols_with_ci(ax, virt["mean_anisotropy"].values, virt["sharpe"].values)
    handles, labels = ax.get_legend_handles_labels()
    labels = [
        f"OLS fit on virtuous, $r = {r:.2f}$ (95% CI)" if "CI" in lbl and r is not None else lbl
        for lbl in labels
    ]
    ax.legend(handles, labels, loc="lower left", fontsize=9, framealpha=0.95)
    ax.text(0.02, 0.98, r"low $\rho_\psi$ $\Rightarrow$ concentrated drift",
            fontsize=10, transform=ax.transAxes, ha="left", va="top",
            style="italic", color="dimgrey")
    ax.set_xlabel(r"Drift anisotropy $\rho_\psi = \bar{\psi}/\psi_{\max}$")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Synthetic: drift anisotropy vs Sharpe")
    styled_grid(ax)
    fig.tight_layout()
    savefig(fig, "fig01_drift_anisotropy.pdf")


if __name__ == "__main__":
    main()
