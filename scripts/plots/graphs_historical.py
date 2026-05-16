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
# # Historical Backtest Graphs From Excel
#
# This notebook reads the exported historical/backtest Excel files from the repo root and recreates the paper-style comparison plots. Run `experiment_ppp.ipynb` through its export cell first if these Excel files are missing.

# %%
from __future__ import annotations
import os

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

RESULTS_DIR = Path(os.environ.get("XLSX_DIR", "results/data/xlsx")).resolve()
PLOTS_DIR = Path(os.environ.get("FIGURES_DIR", "results/figures")).resolve()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Historical/backtest files are the unlabeled exports, e.g. features_data_12_24.xlsx.
PLOT_DATA_LABEL = None
RUN_CONFIGS = [
    {"k": 12, "T": 24},
    {"k": 14, "T": 36},
    {"k": 16, "T": 24},
    {"k": 18, "T": 24},
    {"k": 18, "T": 36},
]

print(f"Reading Excel files from: {RESULTS_DIR}")
print(f"Saving plots to: {PLOTS_DIR}")


# %%
def normalize_data_label(label):
    if label is None:
        return None
    safe = "_".join(str(label).strip().split())
    return None if safe.lower() in {"", "all", "all_data", "real", "openap"} else safe


def parse_experiment_file(path_like, base_name):
    path = Path(path_like)
    parts = path.stem.split("_")
    base_parts = base_name.split("_")
    if parts[: len(base_parts)] != base_parts or len(parts) < len(base_parts) + 2:
        raise ValueError(f"Could not parse experiment metadata from {path.name}")
    k = int(parts[-2])
    T = int(parts[-1])
    label = "_".join(parts[len(base_parts): -2]) or None
    return {"path": path, "k": k, "T": T, "data_label": label}


def discover_experiment_files(base_name, data_label=None):
    target_label = normalize_data_label(data_label)
    matches = []
    for path in sorted(RESULTS_DIR.glob(f"{base_name}*.xlsx")):
        try:
            info = parse_experiment_file(path, base_name)
        except Exception:
            continue
        file_label = normalize_data_label(info["data_label"])
        if target_label is None:
            if file_label is not None:
                continue
        elif file_label != target_label:
            continue
        matches.append(info)
    return sorted(matches, key=lambda item: (item["k"], item["T"], item["path"].name))


def ordered_experiment_infos(base_name):
    infos = discover_experiment_files(base_name, data_label=PLOT_DATA_LABEL)
    if not infos:
        raise FileNotFoundError(f"No {base_name} Excel files found for the selected data label.")

    desired_configs = [(int(cfg["k"]), int(cfg["T"])) for cfg in RUN_CONFIGS]
    info_by_config = {(info["k"], info["T"]): info for info in infos}
    ordered = [info_by_config[cfg] for cfg in desired_configs if cfg in info_by_config]
    missing = [cfg for cfg in desired_configs if cfg not in info_by_config]
    extras = sorted(set(info_by_config) - set(desired_configs))

    if missing:
        print(f"Missing {base_name} Excel files for RUN_CONFIGS: {missing}")
    if extras:
        print(f"Ignoring {base_name} Excel files outside RUN_CONFIGS: {extras}")
    if not ordered:
        raise FileNotFoundError(f"No {base_name} Excel files match RUN_CONFIGS.")
    return ordered


def excel_col_to_idx(col_letters):
    n = 0
    for ch in col_letters:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


def read_xlsx_fallback(path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            cell_map = {}
            max_idx = -1
            for c in row.findall("a:c", ns):
                ref = c.attrib.get("r", "")
                m = re.match(r"([A-Z]+)", ref)
                if not m:
                    continue
                idx = excel_col_to_idx(m.group(1))
                max_idx = max(max_idx, idx)
                cell_type = c.attrib.get("t")
                if cell_type == "inlineStr":
                    is_node = c.find("a:is", ns)
                    val = "" if is_node is None else "".join(t.text or "" for t in is_node.findall(".//a:t", ns))
                else:
                    v = c.find("a:v", ns)
                    val = "" if v is None or v.text is None else v.text
                    if cell_type == "s" and val != "":
                        val = shared[int(val)]
                cell_map[idx] = val
            if max_idx >= 0:
                vals = [""] * (max_idx + 1)
                for idx, val in cell_map.items():
                    vals[idx] = val
                rows.append(vals)

    if not rows:
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    header = [str(x).strip() for x in rows[0]]
    if header and header[0] == "":
        header[0] = "rff_n_component"
    return pd.DataFrame(rows[1:], columns=header)


def read_excel_table(path):
    try:
        return pd.read_excel(path)
    except Exception:
        return read_xlsx_fallback(path)


STYLE_BY_CONFIG = {
    (12, 24): dict(color="#1f77b4", marker="o", linestyle="-"),
    (14, 36): dict(color="#2ca02c", marker="s", linestyle="--"),
    (16, 24): dict(color="#9467bd", marker="^", linestyle="-"),
    (18, 24): dict(color="#ff7f0e", marker="v", linestyle=":"),
    (18, 36): dict(color="#d62728", marker="D", linestyle="-"),
}
FALLBACK_STYLES = [
    dict(color="#1f77b4", marker="o", linestyle="-"),
    dict(color="#2ca02c", marker="s", linestyle="--"),
    dict(color="#9467bd", marker="^", linestyle="-"),
    dict(color="#ff7f0e", marker="v", linestyle=":"),
    dict(color="#d62728", marker="D", linestyle="-"),
    dict(color="#8c564b", marker="P", linestyle="--"),
    dict(color="#e377c2", marker="X", linestyle="-."),
]


def style_for(k, T, i=0, **overrides):
    style = dict(STYLE_BY_CONFIG.get((k, T), FALLBACK_STYLES[i % len(FALLBACK_STYLES)]))
    style.update(overrides)
    return style


# %% [markdown]
# ## Figures 19-20: Historical Diagnostics
#
# Reads `features_data_*.xlsx` and saves the paper-style spectral diagnostics (Figure 19) and subspace-stability diagnostics (Figure 20).

# %%
def load_feature_metric(path, metric_names):
    df = read_excel_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    need = {"rff_n_component", "column", "mean", "std"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    out = df[df["column"].astype(str).str.strip().isin(metric_names)].copy()
    if out.empty:
        raise ValueError(f"{path} does not contain any of {metric_names}")

    out["rff_n_component"] = pd.to_numeric(out["rff_n_component"], errors="coerce")
    out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
    out["std"] = pd.to_numeric(out["std"], errors="coerce").fillna(0.0)
    out = (
        out.dropna(subset=["rff_n_component", "mean"])
        .groupby("rff_n_component", as_index=False)[["mean", "std"]]
        .mean()
        .sort_values("rff_n_component")
    )
    return out[out["rff_n_component"] > 0].copy()


def plot_metric_with_band(ax, frame, style, label):
    ax.plot(frame["rff_n_component"], frame["mean"], label=label, **style)
    ax.fill_between(
        frame["rff_n_component"],
        frame["mean"] - frame["std"],
        frame["mean"] + frame["std"],
        color=style["color"],
        alpha=0.10,
    )


feature_infos = ordered_experiment_infos("features_data")
print("Feature files:", [info["path"].name for info in feature_infos])

plt.style.use("seaborn-v0_8-whitegrid")

fig19, axes19 = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
fig20, axes20 = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)

for i, info in enumerate(feature_infos):
    path = info["path"]
    k, T = info["k"], info["T"]
    style = style_for(k, T, i, linewidth=2.6, markersize=7.5)
    label = fr"$k={k},\, T={T}$"

    erank_df = load_feature_metric(path, {"erank", "effective_rank"})
    gap_df = load_feature_metric(path, {"gap_ratio"})
    # Frozen xlsx files use the legacy "grassmann_dist" column name;
    # freshly-regenerated runs use "grassmannian_dist". Accept either.
    grassmannian_df = load_feature_metric(path, {"grassmannian_dist", "grassmann_dist"})
    angle_df = load_feature_metric(path, {"principal_angle_max"})

    plot_metric_with_band(axes19[0], erank_df, style, label)
    plot_metric_with_band(axes19[1], gap_df, style, label)
    plot_metric_with_band(axes20[0], grassmannian_df, style, label)
    plot_metric_with_band(axes20[1], angle_df, style, label)

for ax in axes19:
    ax.set_xscale("log")
    ax.set_xlabel(r"RFF components $\tilde{m}$ (log scale)", fontsize=13)
    ax.grid(True, alpha=0.25)

axes19[0].set_title("Effective rank vs. RFF", fontsize=16)
axes19[0].set_ylabel(r"$\mathrm{erank}(\hat{\Sigma}_t)$", fontsize=13)
axes19[1].set_title("Gap ratio vs. RFF", fontsize=16)
axes19[1].set_ylabel(r"Gap ratio $\lambda_{\min}/\lambda_{\max}$", fontsize=13)
axes19[1].legend(frameon=True, fontsize=11, loc="best")
fig19.suptitle("Figure 19: Spectral Diagnostics In The Extended Backtest", fontsize=18)
figure19_path = PLOTS_DIR / "figure_19_extended_backtest_spectral_diagnostics.png"
fig19.savefig(figure19_path, dpi=150, bbox_inches="tight")
print(f"Saved {figure19_path}")

for ax in axes20:
    ax.set_xlabel(r"RFF components $\tilde{m}$", fontsize=13)
    ax.grid(True, alpha=0.25)

axes20[0].set_title("Mean geodesic Grassmannian distance", fontsize=16)
axes20[0].set_ylabel("Geodesic Grassmannian distance", fontsize=13)
axes20[1].set_title("Max principal angle", fontsize=16)
axes20[1].set_ylabel("Max principal angle (rad)", fontsize=13)
axes20[1].legend(frameon=True, fontsize=11, loc="best")
fig20.suptitle("Figure 20: Subspace Stability Diagnostics In The Extended Backtest", fontsize=18)
figure20_path = PLOTS_DIR / "figure_20_extended_backtest_subspace_stability_diagnostics.png"
fig20.savefig(figure20_path, dpi=150, bbox_inches="tight")
print(f"Saved {figure20_path}")

plt.show()


# %% [markdown]
# ## Portfolio Performance
#
# Reads `portfolio_performance_*.xlsx` and plots Sharpe ratio and max drawdown against RFF components.

# %%
def read_portfolio_excel(path):
    df = read_excel_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "rff_n_component" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "rff_n_component"})
    for col in ["rff_n_component", "sharpe", "max_dd"]:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["rff_n_component", "sharpe", "max_dd"]).sort_values("rff_n_component")


portfolio_infos = ordered_experiment_infos("portfolio_performance")
print("Portfolio files:", [info["path"].name for info in portfolio_infos])

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
best_sharpe = (-np.inf, None, None, None)

for i, info in enumerate(portfolio_infos):
    path = info["path"]
    k, T = info["k"], info["T"]
    df = read_portfolio_excel(path)
    style = style_for(k, T, i, linewidth=2.4, markersize=7)
    label = fr"$k={k},\, T={T}$"

    axes[0].plot(df["rff_n_component"], df["sharpe"], label=label, **style)
    axes[1].plot(df["rff_n_component"], 100 * df["max_dd"].abs(), label=label, **style)

    i_best = df["sharpe"].idxmax()
    sharpe_peak = float(df.loc[i_best, "sharpe"])
    m_peak = int(df.loc[i_best, "rff_n_component"])
    if sharpe_peak > best_sharpe[0]:
        best_sharpe = (sharpe_peak, m_peak, style["color"], label)

axes[0].set_title("Sharpe vs. RFF", fontsize=17)
axes[0].set_xlabel(r"RFF components $\tilde{m}$", fontsize=13)
axes[0].set_ylabel("Sharpe ratio", fontsize=13)
axes[0].grid(True, alpha=0.25)
axes[0].legend(frameon=True, fontsize=11, loc="best")

if best_sharpe[1] is not None:
    y_peak, m_peak, color_peak, _ = best_sharpe
    axes[0].annotate(
        fr"Sharpe peak: $\tilde{{m}}={m_peak}$, $\approx {y_peak:.2f}$",
        xy=(m_peak, y_peak),
        xytext=(10, 10),
        textcoords="offset points",
        color=color_peak,
        fontsize=12,
        arrowprops=dict(arrowstyle="-", color=color_peak, lw=1.5),
    )

axes[1].set_title("Max drawdown vs. RFF", fontsize=17)
axes[1].set_xlabel(r"RFF components $\tilde{m}$", fontsize=13)
axes[1].set_ylabel("Max drawdown (%)", fontsize=13)
axes[1].grid(True, alpha=0.25)
axes[1].invert_yaxis()
axes[1].legend(frameon=True, fontsize=11, loc="best")

out_path = PLOTS_DIR / "portfolio_performance.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()


# %% [markdown]
# ## OOS R2 Vs. RFF
#
# Reads `oos_r2_*.xlsx` and recreates the paper-style OOS R2 comparison plot.

# %%
def read_r2_excel(path):
    df = read_excel_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["rff_n_component", "oos_r2"]:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["rff_n_component", "oos_r2"]).sort_values("rff_n_component").reset_index(drop=True)


r2_infos = ordered_experiment_infos("oos_r2")
print("OOS R2 files:", [info["path"].name for info in r2_infos])

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
})

fig, ax = plt.subplots(figsize=(8.4, 6.8), dpi=150)
fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.20)
best = (-np.inf, None, None, None)

for i, info in enumerate(r2_infos):
    path = info["path"]
    k, T = info["k"], info["T"]
    df = read_r2_excel(path)
    df = df[df["rff_n_component"] > 0].copy()
    if df.empty:
        continue

    y_pct = 100.0 * df["oos_r2"].to_numpy()
    style = style_for(k, T, i, linewidth=3.2, markersize=10.5)
    if (k, T) == (18, 36):
        style["linewidth"] = 3.6
        style["markersize"] = 11.0
    elif (k, T) in {(16, 24), (18, 24)}:
        style["markersize"] = 11.0

    ax.plot(df["rff_n_component"], y_pct, label=fr"$k={k},\, T={T}$", **style)

    local_idx = int(np.argmax(y_pct))
    local_best = y_pct[local_idx]
    local_x = float(df.iloc[local_idx]["rff_n_component"])
    if local_best > best[0]:
        best = (local_best, local_x, style["color"], (k, T))

ax.set_title(r"OOS $R^2$ vs. RFF", pad=12, fontsize=24)
ax.set_xlabel(r"RFF components $\tilde{m}$", labelpad=18, fontsize=22)
ax.set_ylabel(r"OOS $R^2$ (%)", labelpad=8, fontsize=22)
ax.set_xlim(-180, 4300)
ax.set_ylim(0.27, 0.91)
ax.xaxis.set_major_locator(mticker.FixedLocator([0, 1000, 2000, 3000, 4000]))
ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(0.3, 1.0, 0.1)))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.grid(True, alpha=0.45, color="#d9d9d9", linewidth=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.7)
ax.spines["bottom"].set_linewidth(1.7)
ax.tick_params(axis="both", which="major", width=1.5, length=6, labelsize=18)

if best[1] is not None:
    best_y, best_x, best_color, (best_k, best_T) = best
    arrow_x = min(best_x, 2800.0)
    arrow_y = best_y if arrow_x == best_x else best_y - 0.005
    ax.annotate(
        fr"new best: {best_y:.2f}%",
        xy=(arrow_x, arrow_y),
        xytext=(700, 0.86),
        color=best_color,
        fontsize=18,
        ha="left",
        arrowprops=dict(arrowstyle="-", lw=1.8, color=best_color, alpha=0.75),
    )

leg = ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98), frameon=False, handlelength=1.8, borderaxespad=0.2)
for h in leg.get_lines():
    h.set_linewidth(3.5)

out_path = PLOTS_DIR / "figure_18_extended_backtest_oos_r2_vs_rff.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
plt.show()
