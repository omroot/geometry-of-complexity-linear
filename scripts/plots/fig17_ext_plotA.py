#!/usr/bin/env python3
"""Paper Figure 17 (§9.10.1) -- OOS R^2 crossover at p/n ~ 0.48.

For the strongest extended configuration (k=18, T=36, z=100), plots OOS R^2
vs the parameter ratio p/n = 2*m*k/n_train. Shaded regions mark the
under/over-parameterised regimes; the dashed vertical is the interpolation
threshold p/n=1. Crossover from negative to positive R^2 lands at
p/n ~ 0.48 (m=24), well before the threshold.

This is the file historically known as ``ext_plotA.pdf``.

Output: results/figures/fig17_ext_plotA.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import apply_paper_style, load_sweep_csv, savefig


K_TARGET: int = 18
T_TARGET: int = 36
Z_TARGET: float = 100.0


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv("real_extended.csv")
    sub = df[(df["k"] == K_TARGET) & (df["T"] == T_TARGET) & (df["z"] == Z_TARGET)].sort_values("P")
    if sub.empty:
        raise ValueError(
            f"No rows for k={K_TARGET}, T={T_TARGET}, z={Z_TARGET}. "
            "Run scripts/experiments/real_data/04_extended_grid.py first."
        )

    n_train = T_TARGET * 50  # top-50 universe, T months -> n_train obs per fit
    sub = sub.assign(p_over_n=2 * sub["P"] * K_TARGET / n_train)
    baseline = sub[sub["P"] == 0]
    rff = sub[sub["P"] > 0]
    crossover_idx = rff["oos_r2"].gt(0).idxmax() if (rff["oos_r2"] > 0).any() else None

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.05},
    )
    x_max = max(1.1, float(2 * sub["P"].max() * K_TARGET / n_train) * 1.1)
    for ax in (ax_top, ax_bot):
        ax.axvspan(0, 1, color="#a5d6a7", alpha=0.10, lw=0)
        ax.axvspan(1, x_max, color="#ef9a9a", alpha=0.10, lw=0)
        ax.axvline(1, color="#9575cd", linestyle="--", linewidth=1.8, alpha=0.95)
        ax.grid(True, alpha=0.25)

    if not baseline.empty:
        b = baseline.iloc[0]
        ax_top.plot(b["p_over_n"], b["oos_r2"] * 100, marker="X", markersize=14,
                    color="#e53935", linewidth=0)
        ax_top.annotate(fr"$\tilde m=0$: {b['oos_r2']*100:.2f}%",
                        xy=(b["p_over_n"], b["oos_r2"] * 100),
                        xytext=(10, -2), textcoords="offset points",
                        color="#e53935", fontsize=13, va="center")
    ax_top.set_ylim(-4.0, -2.0)

    ax_bot.plot(rff["p_over_n"], rff["oos_r2"] * 100, color="#e53935", linewidth=3,
                marker="D", markersize=10, markerfacecolor="white",
                markeredgecolor="#e53935", markeredgewidth=2.5)
    for m_label in (24, 64, 256, 1024, 4096):
        row = rff[rff["P"] == m_label]
        if not row.empty:
            ax_bot.annotate(fr"$\tilde m={m_label}$",
                            xy=(row["p_over_n"].iloc[0], row["oos_r2"].iloc[0] * 100),
                            xytext=(8, 8), textcoords="offset points", fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.7))
    if crossover_idx is not None:
        cx = sub.loc[crossover_idx, "p_over_n"]; cy = sub.loc[crossover_idx, "oos_r2"] * 100
        ax_bot.annotate(fr"crossover at $p/n \approx {cx:.2f}$",
                        xy=(cx, cy), xytext=(cx + 0.16 * x_max, cy + 0.15),
                        color="#9575cd", fontsize=14,
                        arrowprops=dict(arrowstyle="-", color="#9575cd", lw=1.8, alpha=0.75))

    y_low = min(0.0, float(rff["oos_r2"].min() * 100) - 0.1) if not rff.empty else 0.0
    y_high = max(1.05, float(rff["oos_r2"].max() * 100) + 0.1) if not rff.empty else 1.05
    ax_bot.set_ylim(y_low, y_high)
    ax_bot.set_xlim(-0.5, x_max)
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)

    d = 0.015
    kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kw); ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw); ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    ax_top.set_ylabel(r"OOS $R^2$ (%)", fontsize=16)
    ax_bot.set_ylabel(r"OOS $R^2$ (%)", fontsize=16)
    ax_bot.set_xlabel(r"$p/n = 2\tilde{m}k / n_{\mathrm{train}}$", fontsize=16)
    fig.suptitle(fr"OOS $R^2$ crossover: $k={K_TARGET}, T={T_TARGET}, z={Z_TARGET:g}$",
                 fontsize=20, y=0.98)
    fig.tight_layout()
    savefig(fig, "fig17_ext_plotA.pdf")


if __name__ == "__main__":
    main()
