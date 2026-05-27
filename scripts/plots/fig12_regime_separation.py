#!/usr/bin/env python3
"""Paper Figure 12 (§9.4) -- historical regime separation: Sharpe vs d_proj.

Each marker is one (k, P, z) configuration from the real-data baseline
grid. Unregularised configurations (z=0) cluster in the frozen regime
(d_proj < 0.5); regularised configurations (z>0) populate the active
regime with higher Sharpes on average.

Output: results/figures/fig12_regime_separation.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import apply_paper_style, load_sweep_csv, savefig, styled_grid


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv("real_baseline_grid.csv", "all_rff_results.csv").dropna(
        subset=["mean_subspace_stability", "sharpe"]
    )
    virt = df[df["z"] > 0]
    ills = df[df["z"] == 0]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(virt["mean_subspace_stability"], virt["sharpe"], c="steelblue", marker="o",
               s=55, alpha=0.75, edgecolors="k", linewidth=0.4,
               label=f"Regularised ($z>0$, $n={len(virt)}$)")
    ax.scatter(ills["mean_subspace_stability"], ills["sharpe"], c="firebrick", marker="^",
               s=55, alpha=0.75, edgecolors="k", linewidth=0.4,
               label=f"Unregularised ($z=0$, $n={len(ills)}$)")
    ax.axvspan(0, 0.5, alpha=0.10, color="salmon",
               label=r"Frozen regime ($d_{\mathrm{proj}} < 0.5$)")
    r_all = float(np.corrcoef(df["mean_subspace_stability"], df["sharpe"])[0, 1])
    ax.text(0.97, 0.05, f"all-data $r = {r_all:.2f}$", fontsize=10,
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", alpha=0.95))
    ax.set_xlabel(r"Mean projection distance $d_{\mathrm{proj}}$ (radians)")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title(r"Historical: Sharpe vs $d_{\mathrm{proj}}$ -- regime separation")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    styled_grid(ax)
    fig.tight_layout()
    savefig(fig, "fig12_regime_separation.pdf")


if __name__ == "__main__":
    main()
