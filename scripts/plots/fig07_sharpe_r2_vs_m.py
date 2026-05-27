#!/usr/bin/env python3
"""Paper Figure 7 (§8.3.7) -- Sharpe and OOS R^2 rise with RFF dim m.

Two panels (left: Sharpe; right: OOS R^2 in %) versus the RFF dimension on
a log axis, for k in {12, 24} at fixed z=10. Sweep D.

Output: results/figures/fig07_sharpe_r2_vs_m.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import apply_paper_style, load_sweep_csv, savefig, styled_grid


Z_FIXED: float = 10.0


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv(
        "sweep_rff_dim.csv", "all_synthetic_datarun_results.csv",
        "synthetic_v3_sweep_D_summary.csv",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    style = {12: ("steelblue", "o", "-"), 24: ("darkorange", "s", "--")}
    for kv, (color, marker, ls) in style.items():
        s = df[(df["k"] == kv) & (df["z"] == Z_FIXED)].sort_values("P")
        if len(s) < 2:
            continue
        axes[0].plot(s["P"], s["sharpe"], color=color, marker=marker, ms=8, lw=2.2, ls=ls,
                     label=fr"$k={kv},\ z={Z_FIXED:g}$")
        axes[1].plot(s["P"], s["oos_r2"] * 100, color=color, marker=marker,
                     ms=8, lw=2.2, ls=ls, label=fr"$k={kv},\ z={Z_FIXED:g}$")

    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"RFF dimension $\tilde m$"); axes[0].set_ylabel("Annualised Sharpe ratio")
    axes[0].set_title(r"Sharpe vs $\tilde m$"); axes[0].legend(fontsize=9, loc="lower right")
    styled_grid(axes[0])

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"RFF dimension $\tilde m$"); axes[1].set_ylabel(r"OOS $R^2$ (%)")
    axes[1].set_title(r"OOS $R^2$ vs $\tilde m$")
    axes[1].axhline(0, color="grey", ls=":", lw=1, alpha=0.6)
    axes[1].legend(fontsize=9, loc="lower right"); styled_grid(axes[1])

    fig.suptitle(
        r"Sharpe and OOS $R^2$ vs $\tilde m$ -- Sweep D ($k\in\{12,24\}$, $z=10$)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "fig07_sharpe_r2_vs_m.png", dpi=200)


if __name__ == "__main__":
    main()
