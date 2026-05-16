# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,scripts///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Synthetic Graphs: Figures 8-11
#
# This notebook is plotting-only. It reads the exported synthetic Excel files from the repo root and recreates the four synthetic figures used in `VirtueOfComplGeometry.pdf`:
#
# - Figure 8: two `T=24` runs, fitted-rank comparison
# - Figure 9: stability summaries for the two `T=24` runs
# - Figure 10: matched-rank comparison across window lengths
# - Figure 11: matched-rank stability comparison across window lengths
#
# The generated images are saved to `plots/synthetic/` with the same figure basenames used in the paper.

# %%
from __future__ import annotations
import os

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("XLSX_DIR", "results/data/xlsx")).resolve()
PLOTS_DIR = Path(os.environ.get("FIGURES_DIR", "results/figures")).resolve()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RFF_GRID = [0, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096]
RUNS = {
    (17, 24): {
        "label": r"$k=17, T=24$",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    (18, 24): {
        "label": r"$k=18, T=24$",
        "color": "#d62728",
        "marker": "s",
        "linestyle": "--",
    },
    (18, 36): {
        "label": r"$k=18, T=36$",
        "color": "#9467bd",
        "marker": "^",
        "linestyle": "-",
    },
}

FIGSIZE_2X2 = (8.90, 6.612)  # 2225 x 1653 px at 250 dpi, matching the paper PNGs.
FIGSIZE_1X2 = (8.90, 3.268)  # 2225 x 817 px at 250 dpi.
FIG_DPI = 250

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

print(f"Reading Excel files from: {DATA_DIR}")
print(f"Saving figures to: {PLOTS_DIR}")


# %%
def find_excel(kind: str, k: int, T: int) -> Path:
    """Find the synthetic Excel file for one artifact/configuration."""
    candidates = [
        DATA_DIR / f"{kind}_synthetic_{k}_{T}.xlsx",
        DATA_DIR / f"{kind}_synthetic_{k}_months_{T}.xlsx",
        DATA_DIR / f"{kind}_synthetic_{k}_month_{T}.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {kind} Excel for k={k}, T={T}. Tried:\n{tried}")


def read_features(k: int, T: int) -> pd.DataFrame:
    path = find_excel("features_data", k, T)
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"rff_n_component", "column", "mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    out = (
        df.assign(
            rff_n_component=lambda d: pd.to_numeric(d["rff_n_component"], errors="coerce"),
            mean=lambda d: pd.to_numeric(d["mean"], errors="coerce"),
        )
        .dropna(subset=["rff_n_component", "column", "mean"])
        .pivot_table(index="rff_n_component", columns="column", values="mean", aggfunc="mean")
        .reset_index()
    )
    out.columns = [str(c).strip() for c in out.columns]
    return out


def read_portfolio(k: int, T: int) -> pd.DataFrame:
    path = find_excel("portfolio_performance", k, T)
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "rff_n_component" not in df.columns:
        df = df.rename(columns={df.columns[0]: "rff_n_component"})
    for col in ["rff_n_component", "sharpe"]:
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["rff_n_component", "sharpe"]].dropna()


def read_oos_r2(k: int, T: int) -> pd.DataFrame:
    path = find_excel("oos_r2", k, T)
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["rff_n_component", "oos_r2"]:
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["rff_n_component", "oos_r2"]].dropna()


def load_run(k: int, T: int) -> pd.DataFrame:
    df = read_features(k, T)
    df = df.merge(read_portfolio(k, T), on="rff_n_component", how="outer")
    df = df.merge(read_oos_r2(k, T), on="rff_n_component", how="outer")
    df["k"] = k
    df["T"] = T
    return df.sort_values("rff_n_component").reset_index(drop=True)


data = {cfg: load_run(*cfg) for cfg in RUNS}
pd.concat(data, names=["config", "row"]).reset_index().head()


# %%
def x_position(values, grid=RFF_GRID):
    """Place RFF values on an evenly spaced paper-style axis."""
    index = {value: i for i, value in enumerate(grid)}
    positions = []
    for value in values:
        value = int(value)
        if value in index:
            positions.append(float(index[value]))
            continue
        # Interpolate threshold lines such as m=36 between neighboring grid ticks.
        lower = max(v for v in grid if v < value)
        upper = min(v for v in grid if v > value)
        positions.append(index[lower] + (value - lower) / (upper - lower))
    return np.asarray(positions, dtype=float)


def set_rff_axis(ax):
    ax.set_xlim(-0.35, len(RFF_GRID) - 0.65)
    ax.set_xticks(range(len(RFF_GRID)))
    ax.set_xticklabels([str(v) for v in RFF_GRID], rotation=28, ha="right")
    ax.set_xlabel(r"RFF components $\tilde{m}$")
    ax.grid(True, alpha=0.28, color="#cfcfcf", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4)


def plot_run(ax, cfg, y_col, *, label=True):
    frame = data[cfg].dropna(subset=["rff_n_component", y_col]).copy()
    style = RUNS[cfg]
    ax.plot(
        x_position(frame["rff_n_component"]),
        frame[y_col],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2.2,
        markersize=5.8,
        markeredgewidth=0.0,
        label=style["label"] if label else None,
    )


def add_threshold(ax, m_value, *, color="0.25", alpha=0.9):
    ax.axvline(
        x_position([m_value])[0],
        color=color,
        linestyle="--",
        linewidth=1.45,
        alpha=alpha,
        zorder=0,
    )


def add_right_legend(fig, ax, *, y=0.52):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    return fig.legend(
        unique.values(),
        unique.keys(),
        loc="center right",
        bbox_to_anchor=(0.985, y),
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.0,
    )


def finish_2x2(fig, title, out_name):
    fig.suptitle(title, y=0.965, fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.80, top=0.88, bottom=0.135, wspace=0.28, hspace=0.43)
    path = PLOTS_DIR / out_name
    fig.savefig(path, dpi=FIG_DPI)
    print(f"Saved {path}")
    return path


def finish_1x2(fig, title, out_name):
    fig.suptitle(title, y=0.955, fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.80, top=0.80, bottom=0.255, wspace=0.30)
    path = PLOTS_DIR / out_name
    fig.savefig(path, dpi=FIG_DPI)
    print(f"Saved {path}")
    return path


# %% [markdown]
# ## Figure 8
#
# Two `T=24` synthetic runs: fitted-rank comparison across average OOS `R^2`, Sharpe ratio, effective rank, and gap ratio.

# %%
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2, dpi=FIG_DPI)
fig8_runs = [(17, 24), (18, 24)]
panels = [
    (axes[0, 0], "oos_r2", r"Average OOS $R^2$", (-0.0034, 0.00125), mticker.MultipleLocator(0.001)),
    (axes[0, 1], "sharpe", "Sharpe ratio", (0.0, 0.95), mticker.MultipleLocator(0.1)),
    (axes[1, 0], "erank", "Effective rank", (15.1, 17.95), mticker.MultipleLocator(0.5)),
    (axes[1, 1], "gap_ratio", "Gap ratio", (0.10, 0.60), mticker.MultipleLocator(0.1)),
]

for ax, y_col, title, ylim, locator in panels:
    for cfg in fig8_runs:
        plot_run(ax, cfg, y_col)
    add_threshold(ax, 24)
    set_rff_axis(ax)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(locator)

add_right_legend(fig, axes[0, 0])
finish_2x2(fig, r"Two $T=24$ synthetic runs: fitted-rank comparison", "figure_08_synthetic_t24_fitted_rank_comparison.png")
plt.show()

# %% [markdown]
# ## Figure 9
#
# Stability summaries for the two `T=24` synthetic runs.

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_1X2, dpi=FIG_DPI)
fig9_runs = [(17, 24), (18, 24)]
panels = [
    (axes[0], "d_proj_norm", "Normalized projection distance", (0.278, 0.308), mticker.MultipleLocator(0.005)),
    (axes[1], "principal_angle_mean", "Mean principal angle", (0.105, 0.195), mticker.MultipleLocator(0.02)),
]

for ax, y_col, title, ylim, locator in panels:
    for cfg in fig9_runs:
        plot_run(ax, cfg, y_col)
    add_threshold(ax, 24)
    set_rff_axis(ax)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(locator)

add_right_legend(fig, axes[0])
finish_1x2(fig, r"Stability summaries for the two $T=24$ runs", "figure_09_synthetic_t24_stability_summaries.png")
plt.show()

# %% [markdown]
# ## Figure 10
#
# Matched-rank comparison across window lengths.

# %%
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2, dpi=FIG_DPI)
fig10_runs = [(18, 24), (18, 36)]
panels = [
    (axes[0, 0], "oos_r2", r"Average OOS $R^2$", (-0.0032, 0.00185), mticker.MultipleLocator(0.001)),
    (axes[0, 1], "sharpe", "Sharpe ratio", (0.0, 1.00), mticker.MultipleLocator(0.1)),
    (axes[1, 0], "erank", "Effective rank", (15.2, 18.05), mticker.MultipleLocator(0.5)),
    (axes[1, 1], "gap_ratio", "Gap ratio", (0.10, 0.62), mticker.MultipleLocator(0.1)),
]

for ax, y_col, title, ylim, locator in panels:
    for cfg in fig10_runs:
        plot_run(ax, cfg, y_col)
    add_threshold(ax, 24, color=RUNS[(18, 24)]["color"], alpha=0.85)
    add_threshold(ax, 36, color=RUNS[(18, 36)]["color"], alpha=0.85)
    set_rff_axis(ax)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(locator)

add_right_legend(fig, axes[0, 0])
finish_2x2(fig, r"Matched-rank extension: $k=18$ at $T=24$ and $T=36$", "figure_10_synthetic_rank_aligned_window_length_comparison.png")
plt.show()

# %% [markdown]
# ## Figure 11
#
# Matched-rank stability comparison across window lengths.

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_1X2, dpi=FIG_DPI)
fig11_runs = [(18, 24), (18, 36)]
panels = [
    (axes[0], "d_proj_norm", "Normalized projection distance", (0.262, 0.307), mticker.MultipleLocator(0.005)),
    (axes[1], "principal_angle_mean", "Mean principal angle", (0.10, 0.205), mticker.MultipleLocator(0.02)),
]

for ax, y_col, title, ylim, locator in panels:
    for cfg in fig11_runs:
        plot_run(ax, cfg, y_col)
    add_threshold(ax, 24, color=RUNS[(18, 24)]["color"], alpha=0.85)
    add_threshold(ax, 36, color=RUNS[(18, 36)]["color"], alpha=0.85)
    set_rff_axis(ax)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(locator)

add_right_legend(fig, axes[0])
finish_1x2(fig, r"Stability extension: matched rank at two window lengths", "figure_11_synthetic_rank_aligned_stability_window_comparison.png")
plt.show()

# %%
sorted(path.name for path in PLOTS_DIR.glob("figure_*.png"))
