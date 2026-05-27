#!/usr/bin/env python3
"""Paper Figure 19 (§9.10.2) -- effective rank + spectral gap across extended grid.

1x2 panel: effective rank (left) and lambda_k/lambda_1 gap ratio (right)
vs RFF dim, both on log-m axes, one line per (k, T) config in the extended
backtest.

Output: results/figures/fig19_spectral_diagnostics.pdf
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
    (16, 24): ("purple", "^"),
    (18, 24): ("darkorange", "v"),
    (18, 36): ("crimson", "D"),
}


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv("real_extended.csv")
    df = df[df["P"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (k, T), (color, marker) in STYLES.items():
        sub = df[(df["k"] == k) & (df["T"] == T)].sort_values("P")
        if sub.empty:
            continue
        label = fr"$k={k},\ T={T}$"
        axes[0].plot(sub["P"], sub["mean_erank"], color=color, marker=marker,
                     ms=6, lw=1.8, label=label)
        axes[1].plot(sub["P"], sub["mean_spectral_gap"], color=color, marker=marker,
                     ms=6, lw=1.8, label=label)

    for ax, title, ylabel in [
        (axes[0], "Effective rank vs RFF dim", r"erank($\hat\Sigma_f$)"),
        (axes[1], "Gap ratio vs RFF dim", r"$\lambda_{\min}/\lambda_{\max}$"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel(r"RFF components $\tilde m$")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); styled_grid(ax)

    fig.suptitle("Spectral diagnostics in the extended backtest", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig19_spectral_diagnostics.pdf")


if __name__ == "__main__":
    main()
