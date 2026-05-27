#!/usr/bin/env python3
"""Paper Figure 4 (§8.3.4) -- Sharpe vs k: z=0 (erratic) vs z>=10 (clean peak).

Two panels at m=1024: left shows the unregularised run's erratic profile;
right shows regularised runs (z=10, z=50) peaking near the true rank k0.

Output: results/figures/fig04_sharpe_vs_k.png
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    ax = axes[0]
    s = df[(df["P"] == P_FIXED) & (df["z"] == 0)].sort_values("k")
    ax.plot(s["k"], s["sharpe"], "r-o", ms=8, lw=2, label=fr"$\tilde m={P_FIXED},\ z=0$")
    ax.axvline(K0, color="orange", ls="--", lw=1.8, label=f"true $k_0={K0}$")
    if not s.empty:
        peak = s.loc[s["sharpe"].idxmax()]
        ax.annotate(
            f"peak at $k={int(peak['k'])}$\n(not at $k_0$)",
            xy=(peak["k"], peak["sharpe"]),
            xytext=(peak["k"] - 8, peak["sharpe"] + 0.05),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9),
        )
    ax.set_xlabel("Factor dimension $k$"); ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("$z=0$ (no ridge): erratic profile")
    ax.legend(loc="lower right", fontsize=9); styled_grid(ax)

    ax = axes[1]
    for zv, color, marker, ls in [(10, "steelblue", "o", "-"), (50, "green", "s", "--")]:
        s = df[(df["P"] == P_FIXED) & (df["z"] == zv)].sort_values("k")
        ax.plot(s["k"], s["sharpe"], color=color, marker=marker, ms=8, lw=2, ls=ls,
                label=fr"$\tilde m={P_FIXED},\ z={zv}$")
    ax.axvline(K0, color="orange", ls="--", lw=1.8, label=f"true $k_0={K0}$")
    s10 = df[(df["P"] == P_FIXED) & (df["z"] == 10)].sort_values("k")
    if not s10.empty:
        peak = s10.loc[s10["sharpe"].idxmax()]
        ax.annotate("peaks at $k_0$",
                    xy=(peak["k"], peak["sharpe"]),
                    xytext=(peak["k"] + 8, peak["sharpe"] - 0.12),
                    fontsize=9, ha="left",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9))
    ax.set_xlabel("Factor dimension $k$")
    ax.set_title(r"$z\geq 10$ (with ridge): clean peak at $k_0$")
    ax.legend(loc="lower right", fontsize=9); styled_grid(ax)

    fig.suptitle(r"Sharpe vs. $k$: $z=0$ never finds the true rank, $z>0$ does",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig04_sharpe_vs_k.png", dpi=200)


if __name__ == "__main__":
    main()
