#!/usr/bin/env python3
"""Paper Figure 11 (§8.5) -- stability extension matched-rank at two windows.

1x2 panel of projection distance and mean principal angle versus RFF dim
for (k=18, T=24) and (k=18, T=36).

Output: results/figures/fig11_matched_rank_stability.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import (
    FIG_DPI, FIGSIZE_1X2, SHORT_WINDOW_STYLES,
    add_right_legend, add_rff_threshold, apply_paper_style,
    load_short_window_csv, plot_short_window_run, savefig, set_rff_axis,
)


CONFIGS: tuple[tuple[int, int], ...] = ((18, 24), (18, 36))


def main() -> None:
    apply_paper_style(short_window=True)
    df = load_short_window_csv()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_1X2, dpi=FIG_DPI)
    panels = [
        (axes[0], "mean_subspace_stability", r"Projection distance $d_{\mathrm{proj}}$",
         mticker.MultipleLocator(0.05)),
        (axes[1], "mean_mean_principal_angle", r"Mean principal angle $\bar\psi$",
         mticker.MultipleLocator(0.02)),
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

    add_right_legend(fig, axes[0])
    fig.suptitle(r"Stability extension: matched rank at two window lengths",
                 y=0.955, fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.80, top=0.80, bottom=0.255, wspace=0.30)
    savefig(fig, "fig11_matched_rank_stability.png", dpi=FIG_DPI)


if __name__ == "__main__":
    main()
