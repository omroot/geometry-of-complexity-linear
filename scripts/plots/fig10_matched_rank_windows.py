#!/usr/bin/env python3
"""Paper Figure 10 (§8.5) -- matched-rank comparison at T=24 vs T=36.

2x2 panel of OOS R^2, Sharpe, effective rank, spectral gap versus RFF dim
for (k=18, T=24) and (k=18, T=36). True synthetic rank k0=18.

Output: results/figures/fig10_matched_rank_windows.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import (
    FIG_DPI, FIGSIZE_2X2, SHORT_WINDOW_STYLES,
    add_right_legend, add_rff_threshold, apply_paper_style,
    load_short_window_csv, plot_short_window_run, savefig, set_rff_axis,
)


CONFIGS: tuple[tuple[int, int], ...] = ((18, 24), (18, 36))


def main() -> None:
    apply_paper_style(short_window=True)
    df = load_short_window_csv()

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2, dpi=FIG_DPI)
    panels = [
        (axes[0, 0], "oos_r2", r"Average OOS $R^2$", mticker.MultipleLocator(0.001)),
        (axes[0, 1], "sharpe", "Sharpe ratio", mticker.MultipleLocator(0.1)),
        (axes[1, 0], "mean_erank", "Effective rank", mticker.MultipleLocator(0.5)),
        (axes[1, 1], "mean_spectral_gap", "Spectral gap", mticker.MultipleLocator(0.1)),
    ]
    for ax, y_col, title, locator in panels:
        for cfg in CONFIGS:
            sub = df[(df["k"] == cfg[0]) & (df["T"] == cfg[1])].sort_values("P")
            if not sub.empty:
                plot_short_window_run(ax, sub, cfg, y_col)
        add_rff_threshold(ax, 24, color=SHORT_WINDOW_STYLES[(18, 24)]["color"], alpha=0.85)
        add_rff_threshold(ax, 36, color=SHORT_WINDOW_STYLES[(18, 36)]["color"], alpha=0.85)
        set_rff_axis(ax)
        ax.set_title(title)
        ax.yaxis.set_major_locator(locator)

    add_right_legend(fig, axes[0, 0])
    fig.suptitle(r"Matched-rank extension: $k=18$ at $T=24$ and $T=36$",
                 y=0.965, fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.80, top=0.88, bottom=0.135,
                        wspace=0.28, hspace=0.43)
    savefig(fig, "fig10_matched_rank_windows.png", dpi=FIG_DPI)


if __name__ == "__main__":
    main()
