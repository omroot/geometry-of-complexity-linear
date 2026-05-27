#!/usr/bin/env python3
"""Paper Figure 16 (§9.8) -- Goldilocks zone: Sharpe vs d_proj colored by P.

Marker color encodes log10(m), marker size proportional to ridge z; the
Goldilocks band d_proj in [0.3, 1.0] is shaded.

Output: results/figures/fig16_goldilocks.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import apply_paper_style, load_sweep_csv, savefig, styled_grid


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv("real_baseline_grid.csv", "all_rff_results.csv").dropna(
        subset=["mean_subspace_stability", "sharpe"]
    ).copy()

    norm = Normalize(vmin=np.log10(64), vmax=np.log10(10000))

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df["mean_subspace_stability"], df["sharpe"],
                    c=np.log10(df["P"]), cmap="viridis", norm=norm,
                    s=df["z"].clip(1, 100) * 1.5 + 25,
                    alpha=0.65, edgecolors="k", linewidth=0.35)
    plt.colorbar(sc, ax=ax, label=r"$\log_{10}(\tilde m)$")

    ax.axvspan(0.3, 1.0, alpha=0.10, color="green", label="Goldilocks zone")
    ax.axvline(0.3, color="green", ls=":", lw=1, alpha=0.5)
    ax.axvline(1.0, color="green", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel(r"$d_{\mathrm{proj}}$ (subspace stability)")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title(r"Sharpe vs $d_{\mathrm{proj}}$ -- Goldilocks zone (size $\propto z$)")
    ax.legend(loc="lower right", fontsize=9)
    styled_grid(ax)
    fig.tight_layout()
    savefig(fig, "fig16_goldilocks.pdf")


if __name__ == "__main__":
    main()
