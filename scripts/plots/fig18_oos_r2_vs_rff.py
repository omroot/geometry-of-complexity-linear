#!/usr/bin/env python3
"""Paper Figure 18 (§9.10.1) -- OOS R^2 across the extended (k, T) grid.

Plots OOS R^2 (%) vs RFF dim for each (k, T) configuration in the extended
backtest. The (k=18, T=36) curve reaches the paper's best ~0.88% at m=4096.

Output: results/figures/fig18_oos_r2_vs_rff.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import apply_paper_style, load_sweep_csv, savefig, styled_grid


STYLES: dict[tuple[int, int], tuple[str, str]] = {
    (12, 24): ("steelblue", "o"),
    (14, 36): ("forestgreen", "s"),
    (18, 24): ("darkorange", "v"),
    (18, 36): ("crimson", "D"),
}


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv("real_extended.csv")
    df = df[df["P"] > 0]  # drop linear baseline

    fig, ax = plt.subplots(figsize=(9, 6))
    best_label = None
    best_val = -float("inf")
    for (k, T), (color, marker) in STYLES.items():
        sub = df[(df["k"] == k) & (df["T"] == T)].sort_values("P")
        if sub.empty:
            continue
        ax.plot(sub["P"], sub["oos_r2"] * 100, color=color, marker=marker,
                ms=7, lw=2, label=fr"$k={k},\ T={T}$")
        if float(sub["oos_r2"].max()) > best_val:
            best_val = float(sub["oos_r2"].max()) * 100
            best_label = fr"new best: {best_val:.2f}%"

    if best_label is not None:
        ax.text(0.55, 0.95, best_label, transform=ax.transAxes,
                fontsize=11, color="crimson", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", alpha=0.95))
    ax.set_xlabel(r"RFF components $\tilde m$")
    ax.set_ylabel(r"OOS $R^2$ (%)")
    ax.set_title(r"OOS $R^2$ vs RFF dimension across extended (k, T) cells")
    ax.axhline(0, color="grey", ls=":", lw=1, alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    styled_grid(ax)
    fig.tight_layout()
    savefig(fig, "fig18_oos_r2_vs_rff.pdf")


if __name__ == "__main__":
    main()
