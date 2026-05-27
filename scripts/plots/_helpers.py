"""Shared style, CSV loader, and axis helpers for paper-figure scripts.

Co-located with the figure scripts (not under src/) to keep the active
``src/`` surface minimal. Every script under ``scripts/plots/fig*.py``
imports from here.

Schema canonicalization: this module's :func:`load_sweep_csv` and
:func:`load_short_window_csv` translate both the **legacy** column names
(``avg_sharpe``, ``avg_subspace_stability``, ``result_set``, ...) and the
**new** ones produced by ``scripts/experiments/{synthetic,real_data}/*.py``
(``sharpe``, ``mean_subspace_stability``, ``sweep``, ...) into a single
canonical schema using the **new** names. Plot scripts read canonical names
only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.legend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.environ.get("GOCL_DATA_DIR", REPO_ROOT / "results" / "data"))
FIGURES_DIR: Path = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "results" / "figures"))


def ensure_figures_dir() -> Path:
    """Create ``FIGURES_DIR`` if missing and return the resolved path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PAPER_RC: dict[str, Any] = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
}

PAPER_RC_SHORT_WINDOW: dict[str, Any] = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


def apply_paper_style(short_window: bool = False) -> None:
    """Apply the paper-figure matplotlib rcParams."""
    plt.rcParams.update(PAPER_RC)
    if short_window:
        plt.rcParams.update(PAPER_RC_SHORT_WINDOW)


def styled_grid(ax: matplotlib.axes.Axes) -> None:
    """Apply the standard light-grey grid used across §8 / §9 figures."""
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# CSV loader (schema canonicalization)
# ---------------------------------------------------------------------------

_RENAME_MAP: dict[str, str] = {
    "avg_sharpe": "sharpe",
    "avg_r2_oos": "oos_r2",
    "avg_subspace_stability": "mean_subspace_stability",
    "avg_max_principal_angle": "mean_max_principal_angle",
    "avg_mean_principal_angle": "mean_mean_principal_angle",
    "avg_erank": "mean_erank",
    "avg_spectral_gap": "mean_spectral_gap",
    "avg_geodesic_accel": "mean_geodesic_accel",
    "anisotropy": "mean_anisotropy",
    "result_set": "sweep",
    "run_label": "sweep",
}


def _canonicalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns to the new canonical names; coerce ``z`` to float.

    The legacy CSV stores ``z`` as ``"z=1e+01"``; the new schema stores it as
    a float. We coerce here so plot scripts can compare ``df.z == 10``.
    """
    df = df.rename(columns={k: v for k, v in _RENAME_MAP.items() if k in df.columns})
    # Derive rho_psi when missing but components are present.
    if "mean_anisotropy" not in df.columns:
        if {"mean_mean_principal_angle", "mean_max_principal_angle"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["mean_anisotropy"] = np.where(
                    df["mean_max_principal_angle"] > 1e-8,
                    df["mean_mean_principal_angle"] / df["mean_max_principal_angle"],
                    np.nan,
                )
    if "z" in df.columns and df["z"].dtype == object:
        df["z"] = df["z"].astype(str).str.replace("z=", "", regex=False).astype(float)
    return df


def _aggregate_seeds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Average per-seed rows when a ``seed`` column is present; otherwise pass-through."""
    if "seed" not in df.columns:
        return df
    return df.groupby(group_cols, as_index=False).mean(numeric_only=True)


def load_sweep_csv(
    *candidate_names: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a sweep CSV, trying multiple candidate basenames in priority order.

    Searches ``DATA_DIR`` then a few fallback locations. Canonicalizes column
    names and (when a ``seed`` column is present) averages per-(k, P, z)
    cells using ``group_cols`` (default: any of k, P, z, T that are present).

    Args:
        *candidate_names: CSV basenames to try, in order.
        group_cols: Columns to group on when aggregating across seeds.

    Returns:
        Canonicalized DataFrame.

    Raises:
        FileNotFoundError: When no candidate resolves.
    """
    search_paths = [DATA_DIR, REPO_ROOT / "results" / "data", REPO_ROOT / "data", Path.cwd()]
    for name in candidate_names:
        for base in search_paths:
            path = base / name
            if path.exists():
                df = _canonicalize_schema(pd.read_csv(path))
                if group_cols is None:
                    group_cols = [c for c in ["k", "P", "z", "T"] if c in df.columns]
                return _aggregate_seeds(df, group_cols)
    tried = "\n  ".join(
        str(base / name) for name in candidate_names for base in search_paths
    )
    raise FileNotFoundError(
        f"None of these CSVs exist. Tried:\n  {tried}\n"
        "Run the appropriate scripts/experiments/* script first."
    )


def load_short_window_csv(
    *candidate_names: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load the §8.5 short-window CSV.

    Aggregates over seeds at the (k, T, P) level. Default candidate list is
    ``("short_window.csv",)``.
    """
    if not candidate_names:
        candidate_names = ("short_window.csv",)
    if group_cols is None:
        group_cols = ["k", "T", "P"]
    return load_sweep_csv(*candidate_names, group_cols=group_cols)


# ---------------------------------------------------------------------------
# OLS overlay (used by Figs 1, 15)
# ---------------------------------------------------------------------------

def plot_ols_with_ci(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = "steelblue",
) -> tuple[float | None, float | None, float | None]:
    """Overlay OLS regression line + 95% confidence band; return (slope, intercept, r).

    Returns ``(None, None, None)`` when fewer than 4 points are supplied.
    """
    if len(x) < 4:
        return None, None, None
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    xf = np.linspace(x.min(), x.max(), 100)
    yf = slope * xf + intercept
    ax.plot(xf, yf, color=color, lw=2, alpha=0.85)
    n = len(x)
    residual_std = float(np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2)))
    x_mean = float(x.mean())
    ci = (
        1.96 * residual_std
        * np.sqrt(1 / n + (xf - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
    )
    ax.fill_between(
        xf, yf - ci, yf + ci, alpha=0.18, color=color,
        label="95% CI for OLS fit (virtuous)",
    )
    return float(slope), float(intercept), float(r_value)


# ---------------------------------------------------------------------------
# Short-window evenly-spaced RFF axis (used by Figs 8-11)
# ---------------------------------------------------------------------------

RFF_GRID: list[int] = [0, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096]

SHORT_WINDOW_STYLES: dict[tuple[int, int], dict[str, str]] = {
    (17, 24): {"label": r"$k=17, T=24$", "color": "#1f77b4", "marker": "o", "linestyle": "-"},
    (18, 24): {"label": r"$k=18, T=24$", "color": "#d62728", "marker": "s", "linestyle": "--"},
    (18, 36): {"label": r"$k=18, T=36$", "color": "#9467bd", "marker": "^", "linestyle": "-"},
}

FIGSIZE_2X2: tuple[float, float] = (8.90, 6.612)
FIGSIZE_1X2: tuple[float, float] = (8.90, 3.268)
FIG_DPI: int = 250


def rff_axis_position(values: np.ndarray | list[float]) -> np.ndarray:
    """Map RFF dimensions to evenly-spaced tick positions on RFF_GRID."""
    index = {value: i for i, value in enumerate(RFF_GRID)}
    positions: list[float] = []
    for value in values:
        v = int(value)
        if v in index:
            positions.append(float(index[v]))
            continue
        lowers = [x for x in RFF_GRID if x < v]
        uppers = [x for x in RFF_GRID if x > v]
        if not lowers or not uppers:
            raise ValueError(f"Value {v} is outside the RFF grid {RFF_GRID}")
        lo, hi = max(lowers), min(uppers)
        positions.append(index[lo] + (v - lo) / (hi - lo))
    return np.asarray(positions, dtype=float)


def set_rff_axis(ax: matplotlib.axes.Axes) -> None:
    """Configure ``ax`` with the short-window evenly-spaced RFF tick axis."""
    ax.set_xlim(-0.35, len(RFF_GRID) - 0.65)
    ax.set_xticks(range(len(RFF_GRID)))
    ax.set_xticklabels([str(v) for v in RFF_GRID], rotation=28, ha="right")
    ax.set_xlabel(r"RFF components $\tilde{m}$")
    ax.grid(True, alpha=0.28, color="#cfcfcf", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_short_window_run(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    config: tuple[int, int],
    y_col: str,
    *,
    label: bool = True,
) -> None:
    """Plot one (k, T) run's series on a short-window axis."""
    style = SHORT_WINDOW_STYLES[config]
    frame = df.dropna(subset=["P", y_col]).copy()
    ax.plot(
        rff_axis_position(frame["P"]),
        frame[y_col],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2.2,
        markersize=5.8,
        markeredgewidth=0.0,
        label=style["label"] if label else None,
    )


def add_rff_threshold(
    ax: matplotlib.axes.Axes, m_value: float, *, color: str = "0.25", alpha: float = 0.9
) -> None:
    """Vertical dashed line at one RFF value on the short-window axis."""
    ax.axvline(
        rff_axis_position([m_value])[0],
        color=color, linestyle="--", linewidth=1.45, alpha=alpha, zorder=0,
    )


def add_right_legend(
    fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, *, y: float = 0.52
) -> matplotlib.legend.Legend:
    """Right-anchored figure legend deduplicated by label text."""
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    return fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="center right",
        bbox_to_anchor=(0.985, y),
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.0,
    )


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def savefig(fig: matplotlib.figure.Figure, name: str, *, dpi: int | None = None) -> Path:
    """Save ``fig`` under ``FIGURES_DIR/name`` and return the path.

    PDF outputs use vector defaults; PNGs respect ``dpi`` (default 200).
    """
    out = ensure_figures_dir() / name
    if name.endswith(".png"):
        fig.savefig(out, dpi=dpi if dpi is not None else 200, bbox_inches="tight")
    else:
        fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    return out
