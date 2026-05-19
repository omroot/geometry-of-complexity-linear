#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: notebooks///ipynb,scripts///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
# ---

# %%

# %%
from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os

# Dual-mode path resolution (works in both notebook and script contexts):
# walk up from the script/notebook's location to find the dir containing src/.
if "__file__" in globals():
    _start = Path(__file__).resolve().parent
else:
    _start = Path.cwd()
PROJECT_ROOT = next(p for p in [_start, *_start.parents] if (p / "src").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"PROJECT_ROOT = {PROJECT_ROOT}, cwd = {Path.cwd()}")

import src
import src.models.als_ipca as ipca
import src.workflows.grassmannian_ipca as GrassmannianIPCAWorkflowModule
import src.workflows.ipca as IPCAWorkflowModule
import src.portfolio as portfolio_utils

importlib.reload(IPCAWorkflowModule)
importlib.reload(GrassmannianIPCAWorkflowModule)
importlib.reload(portfolio_utils)
importlib.reload(ipca)

# %%


WORKDIR = Path.cwd()
RESULTS_DIR = Path(os.environ.get("XLSX_DIR", WORKDIR))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

USE_SYNTHETIC_DATA = False  # refactored_real: switch to real-data path
USE_CACHED_REAL_CHAR_DATA = True  # refactored_real: load cached parquet at data/char_data.parquet
# Originally hardcoded to a co-author path; refactored to look in ./data/ relative to repo root.
CHAR_DATA_CACHE_PATH = Path(os.environ.get("CHAR_DATA_PARQUET", "data/char_data.parquet"))

REAL_DATA_LABEL = "all_data"
SYNTHETIC_DATA_LABEL = "synthetic"
data_label = SYNTHETIC_DATA_LABEL if USE_SYNTHETIC_DATA else REAL_DATA_LABEL
PLOT_DATA_LABEL = data_label

REAL_CHAR_DOWNLOAD = {
    "start_yyyymm": "199001",
    "end_yyyymm": "202512",
}
REAL_RETURNS_DOWNLOAD = {
    "start_date": "1990-01-01",
    "end_date": "2026-01-31",
}

# Per-(k, T) override via env vars (set by run_asset_pricing_ipca_k{K}_T{T}.py entries).
if "GOCL_K" in os.environ and "GOCL_T" in os.environ:
    RUN_CONFIGS = [{"k": int(os.environ["GOCL_K"]), "T": int(os.environ["GOCL_T"])}]
else:
    RUN_CONFIGS = [
        {"k": 12, "T": 24},
        {"k": 14, "T": 36},
        {"k": 16, "T": 24},
        {"k": 18, "T": 24},
        {"k": 18, "T": 36},
    ]

SYNTHETIC_CONFIG = {
    "T": 120,
    "N": 50,
    "m": 75,
    "seed": 10,
    "start_date": "1990-01-01",
    "include_intercept": False,
    "beta_rho": 0.9,
    "sigma_beta": 0.5,
    "z_rho": 0.4,
    "z_scale": 1.0,
    "heavy_tail_df": 5.0,
    "sigma_eps_base": 0.5,
    "hetero_strength": 0.5,
    "missing_prob": 0.05,
    "missing_mode": "mcAR",
    "impute": "zero",
    "z_drift_scale": 0.02,
}

FULL_SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
LARGE_RFF_SEEDS = [10, 20, 30]
RFF_SMALL_COMPONENTS = [24, 32, 48, 64, 128, 256]
RFF_LARGE_COMPONENTS = [512, 1024, 2048, 4096]
BASELINE_RFF_COMPONENT = 0
RFF_GAMMA = 0.25
ALPHA = 100
SHOW_PROGRESS = True

FORECAST_START = "1993-01-01"
MIN_TRAIN_OBS = 500
MARKET_CAP_TOP_N = 50

# Smoke-mode overrides for the .smoke/ harness — opt in with GOCL_SMOKE=1.
if os.environ.get("GOCL_SMOKE") == "1":
    SYNTHETIC_CONFIG = {**SYNTHETIC_CONFIG, "T": 60, "N": 20, "m": 10}
    FULL_SEEDS = [10, 20]
    LARGE_RFF_SEEDS = [10]
    RFF_SMALL_COMPONENTS = [8, 16]
    RFF_LARGE_COMPONENTS = [32]
    FORECAST_START = "2010-01-01"
    MIN_TRAIN_OBS = 24
    MARKET_CAP_TOP_N = 20

SAVE_RESULTS_TO_PICKLES = True
LOAD_RESULTS_FROM_PICKLES = False

char_data = None
truth = None
returns_data = None
run_results = {}
analysis_outputs = {}
created_files = []
ACTIVE_EXPERIMENT = None

def normalize_data_label(label):
    if label is None:
        return None
    safe = "_".join(str(label).strip().split())
    return None if safe.lower() in {"", "all", "all_data", "real", "openap"} else safe

def build_result_pickle_name(base_name, n_factors, train_window_months, data_label=None):
    safe_base = "_".join(str(base_name).strip().split())
    label = normalize_data_label(data_label)
    parts = [safe_base]
    if label:
        parts.append(label)
    parts.extend([str(n_factors), str(train_window_months)])
    return "_".join(parts) + ".pkl"

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

config_preview = pd.DataFrame(RUN_CONFIGS)
print(f"USE_SYNTHETIC_DATA={USE_SYNTHETIC_DATA}, data_label='{data_label}'")
config_preview


# **Synthetic Data (Run When `USE_SYNTHETIC_DATA=True`)**

# %%


from src.models.grassmannian_ipca import generate_ipca_workflow_panel

if USE_SYNTHETIC_DATA:
    synthetic_kwargs = dict(SYNTHETIC_CONFIG)
    synthetic_kwargs["m"] = max(
        synthetic_kwargs["m"],
        max(cfg["k"] for cfg in RUN_CONFIGS),
    )
    char_data, truth = generate_ipca_workflow_panel(**synthetic_kwargs)
    print(
        f"Synthetic panel ready: {char_data.shape[0]:,} rows, "
        f"{char_data['permno'].nunique()} assets, "
        f"{char_data['yyyymm'].nunique()} months."
    )
    char_data.head()
else:
    print("Skipping synthetic generation because USE_SYNTHETIC_DATA=False.")


# **OpenAP Download And Cleaning (Real-Data Path Only)**

# %%


from src.ingest.pipeline import DataPipeline

if not USE_SYNTHETIC_DATA and not USE_CACHED_REAL_CHAR_DATA:
    new_mod = DataPipeline()
    char_data = new_mod.download_openap_data(**REAL_CHAR_DOWNLOAD)
    print(f"Downloaded OpenAP characteristics: {char_data.shape}")
else:
    print("Skipping OpenAP download.")


# %%


from src.ingest.pipeline import DataPipeline

if not USE_SYNTHETIC_DATA and not USE_CACHED_REAL_CHAR_DATA:
    if char_data is None:
        raise ValueError("char_data is not loaded yet. Run the OpenAP download cell first.")
    new_mod = DataPipeline()
    char_data, dropped_cols = new_mod.remove_mostly_nan_columns(char_data, max_nan_frac=0.5)
    print(f"Dropped mostly-NaN columns: {dropped_cols}")
    char_data, kept_chars, dropped_chars = new_mod.drop_low_std_and_high_corr(
        char_data,
        min_std=1e-5,
        max_corr=0.75,
    )
    char_data = new_mod.fill_remaining_missing(char_data, use_past_only=True)
    print(f"Kept {len(kept_chars)} characteristics after cleaning.")
    char_data.head()
else:
    print("Skipping real-data characteristic cleaning.")


# **Optional Cached Real-Data Load**

# %%


if not USE_SYNTHETIC_DATA and USE_CACHED_REAL_CHAR_DATA:
    if not CHAR_DATA_CACHE_PATH.exists():
        raise FileNotFoundError(f"Could not find {CHAR_DATA_CACHE_PATH}")
    char_data = pd.read_parquet(CHAR_DATA_CACHE_PATH, engine="pyarrow")
    print(f"Loaded cached characteristics from {CHAR_DATA_CACHE_PATH}")
    # refactored_real: LinearCase panel uses 'Size' (log market-cap z-score) instead of 'mcap';
    # alias it so the MARKET_CAP_TOP_N filter in run_rff_sweep triggers correctly.
    if "mcap" not in char_data.columns and "Size" in char_data.columns:
        char_data["mcap"] = char_data["Size"]
        print("  added mcap alias from Size column")
    char_data.head()
else:
    print("Skipping cached characteristic load.")


# **WRDS Returns Merge (Real-Data Path Only)**

# %%


# refactored_real: WRDS download + merge disabled because the LinearCase parquet
# already has `excess_ret` joined in. The `if False:` branches keep the original
# code visible for diff/review but never execute.
if False:  # refactored_real: skip WRDS SP500 returns download
    from src.ingest.pipeline import DataPipeline
    new_mod = DataPipeline()
    returns_data = new_mod.download_sp500_returns_wrds(**REAL_RETURNS_DOWNLOAD)
    print(f"Downloaded returns panel: {returns_data.shape}")
else:
    print("refactored_real: skipping WRDS returns download (parquet already has excess_ret).")


# **Finalize Panel And Inspect**

# %%


if False:  # refactored_real: skip merge (parquet already has excess_ret)
    from src.ingest.pipeline import DataPipeline
    if char_data is None:
        raise ValueError("char_data is not loaded yet.")
    new_mod = DataPipeline()
    returns_data = returns_data.rename(columns={"excess_ret": "ret_adj"})
    char_data = new_mod.merge_openap_with_crsp_returns(char_data, returns_data)
    print(f"Merged real-data panel shape: {char_data.shape}")
else:
    print("Skipping real-data merge.")


# **Working Panel Preview**

# %%


if char_data is None:
    raise ValueError("char_data is not ready yet. Finish either the synthetic or real-data path first.")

print(
    f"Working panel: {char_data.shape[0]:,} rows, "
    f"{char_data['permno'].nunique()} assets, "
    f"{char_data['yyyymm'].nunique()} months."
)
char_data.head()


# **Run Multi-(k, T) Sweeps**

# This cell loops over every `(k, T)` pair in `RUN_CONFIGS` and runs the full RFF sweep on the current `char_data` panel.
# 
# It also saves one pickle per completed experiment, so if the runs take a while you can later set `LOAD_RESULTS_FROM_PICKLES = True` and jump straight to the export and plotting cells.
# 

# %%


from src.workflows.grassmannian_ipca import GrassmannianIPCAWorkflow

def run_rff_sweep(char_data, k, train_window_months):
    wf = GrassmannianIPCAWorkflow()
    result_rff_all = {seed: {} for seed in FULL_SEEDS}
    market_cap_col = "mcap" if "mcap" in char_data.columns else None
    market_cap_top_n = min(MARKET_CAP_TOP_N, char_data["permno"].nunique()) if market_cap_col else None

    common_kwargs = dict(
        char_data=char_data,
        forecast_start=FORECAST_START,
        target_col="y_ipca",
        n_factors=k,
        train_window_months=train_window_months,
        min_train_obs=MIN_TRAIN_OBS,
        normalize=True,
        mean_factor=True,
        silent=True,
        iter_tol=1e-4,
        warm_start=True,
        alpha=ALPHA,
        show_progress=SHOW_PROGRESS,
        market_cap_filter_col=market_cap_col,
        market_cap_filter_top_n=market_cap_top_n,
    )

    baseline_seed = FULL_SEEDS[0]
    result_rff_all[baseline_seed][BASELINE_RFF_COMPONENT] = wf.rolling_ipca_predictions(
        use_rff=False,
        rff_n_components=0,
        rff_gamma=RFF_GAMMA,
        rff_random_state=baseline_seed,
        **common_kwargs,
    )

    for seed in FULL_SEEDS:
        for rff_n_component in RFF_SMALL_COMPONENTS:
            result_rff_all[seed][rff_n_component] = wf.rolling_ipca_predictions(
                use_rff=True,
                rff_n_components=rff_n_component,
                rff_gamma=RFF_GAMMA,
                rff_random_state=seed,
                **common_kwargs,
            )

    for seed in LARGE_RFF_SEEDS:
        for rff_n_component in RFF_LARGE_COMPONENTS:
            result_rff_all[seed][rff_n_component] = wf.rolling_ipca_predictions(
                use_rff=True,
                rff_n_components=rff_n_component,
                rff_gamma=RFF_GAMMA,
                rff_random_state=seed,
                **common_kwargs,
            )

    return result_rff_all

if char_data is None:
    raise ValueError("char_data is not ready yet. Prepare the dataset first.")

run_results = {}
saved_pickle_files = []

for cfg in RUN_CONFIGS:
    k = int(cfg["k"])
    T = int(cfg["T"])
    print(f"=== Running experiment: k={k}, T={T}, data_label='{data_label}' ===")

    result_rff_all = run_rff_sweep(
        char_data=char_data,
        k=k,
        train_window_months=T,
    )

    payload = {
        "k": k,
        "T": T,
        "data_label": data_label,
        "alpha": ALPHA,
        "rff_gamma": RFF_GAMMA,
        "result_rff_all": result_rff_all,
    }
    run_results[(k, T)] = payload

    if SAVE_RESULTS_TO_PICKLES:
        out_path = RESULTS_DIR / build_result_pickle_name(
            "result_rff_all",
            k,
            T,
            data_label=data_label,
        )
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        saved_pickle_files.append(out_path.name)
        print(f"Saved {out_path.name}")

pd.DataFrame(
    [
        {
            "k": payload["k"],
            "T": payload["T"],
            "data_label": payload["data_label"],
            "n_rff_levels": len({comp for seed_metrics in payload["result_rff_all"].values() for comp in seed_metrics}),
        }
        for payload in run_results.values()
    ]
).sort_values(["k", "T"]).reset_index(drop=True)


# %%


if LOAD_RESULTS_FROM_PICKLES:
    run_results = {}
    for cfg in RUN_CONFIGS:
        k = int(cfg["k"])
        T = int(cfg["T"])
        in_path = RESULTS_DIR / build_result_pickle_name(
            "result_rff_all",
            k,
            T,
            data_label=data_label,
        )
        if not in_path.exists():
            raise FileNotFoundError(f"Missing pickle: {in_path}")
        with open(in_path, "rb") as f:
            payload = pickle.load(f)
        if "result_rff_all" not in payload:
            payload = {
                "k": k,
                "T": T,
                "data_label": data_label,
                "alpha": ALPHA,
                "rff_gamma": RFF_GAMMA,
                "result_rff_all": payload,
            }
        run_results[(k, T)] = payload

    pd.DataFrame(
        [
            {"k": payload["k"], "T": payload["T"], "data_label": payload["data_label"]}
            for payload in run_results.values()
        ]
    ).sort_values(["k", "T"]).reset_index(drop=True)
else:
    print("Skipping pickle load because LOAD_RESULTS_FROM_PICKLES=False.")


# **Export OOS R², Diagnostics, And Portfolio Tables**

# %%


from src.io.naming import build_experiment_excel_name
from src.portfolio import (
    build_directional_portfolio,
    compute_portfolio_returns,
    portfolio_performance,
)

def summarize_experiment(result_rff_all):
    rff_components = sorted(
        {
            rff_n_component
            for seed_metrics in result_rff_all.values()
            for rff_n_component in seed_metrics
        }
    )
    oos_r2_results = {rff_n_component: [] for rff_n_component in rff_components}
    diag_summaries = {}
    diag_frames = []

    for seed, metrics in result_rff_all.items():
        for rff_n_component, all_metrics in metrics.items():
            pred_df_in = all_metrics[0].copy()
            diag_df_in = all_metrics[1].copy()

            ss_res = ((pred_df_in["y_true"] - pred_df_in["y_pred"]) ** 2).sum()
            ss_tot = (pred_df_in["y_true"] ** 2).sum()
            r2_oos = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
            oos_r2_results[rff_n_component].append(float(r2_oos))

            diag_df_in["seed"] = seed
            diag_df_in["rff_n_component"] = rff_n_component
            diag_frames.append(diag_df_in)
            diag_summaries[(seed, rff_n_component)] = (
                diag_df_in.describe(include="all")
                .T.reset_index()
                .rename(columns={"index": "column"})
            )

    oos_r2_avg = {
        rff_n_component: float(np.nanmean(values)) if len(values) else np.nan
        for rff_n_component, values in oos_r2_results.items()
    }
    n_runs = {
        rff_n_component: int(len(values))
        for rff_n_component, values in oos_r2_results.items()
    }
    oos_r2_export = (
        pd.DataFrame({"rff_n_component": sorted(oos_r2_avg.keys())})
        .assign(
            oos_r2=lambda d: d["rff_n_component"].map(oos_r2_avg),
            n_runs=lambda d: d["rff_n_component"].map(n_runs),
        )
        .sort_values("rff_n_component")
        .reset_index(drop=True)
    )

    features_data_df = (
        pd.concat(diag_summaries.values(), keys=diag_summaries.keys())
        .reset_index(level=[0, 1])
        .rename(columns={"level_0": "seed", "level_1": "rff_n_component"})
    )
    num_cols = [
        c for c in ["count", "mean", "min", "25%", "50%", "75%", "max", "std"]
        if c in features_data_df.columns
    ]
    features_data_df[num_cols] = features_data_df[num_cols].apply(pd.to_numeric, errors="coerce")
    features_data_df = (
        features_data_df
        .groupby(["rff_n_component", "column"], as_index=False)[num_cols]
        .mean()
    )

    mean_prediction_by_rff = {}
    portfolio_results_mean_seed = {}

    for rff_n_component in rff_components:
        seed_pred_list = []
        for seed, seed_metrics in result_rff_all.items():
            if rff_n_component not in seed_metrics:
                continue
            pred_df = seed_metrics[rff_n_component][0].copy()
            pred_df["yyyymm"] = pd.to_datetime(pred_df["yyyymm"])
            pred_df["seed"] = seed
            seed_pred_list.append(
                pred_df[
                    [
                        "permno",
                        "yyyymm",
                        "y_true",
                        "y_pred",
                        "train_end",
                        "n_train",
                        "n_test",
                        "seed",
                    ]
                ]
            )

        if not seed_pred_list:
            continue

        stacked = pd.concat(seed_pred_list, ignore_index=True)
        y_true_check = stacked.groupby(["permno", "yyyymm"])["y_true"].nunique()
        if (y_true_check > 1).any():
            raise ValueError(
                f"y_true is not identical across seeds for rff_n_component={rff_n_component}"
            )

        mean_pred_df = (
            stacked
            .groupby(["permno", "yyyymm"], as_index=False)
            .agg(
                y_true=("y_true", "first"),
                y_pred=("y_pred", "mean"),
                train_end=("train_end", "first"),
                n_train=("n_train", "first"),
                n_test=("n_test", "first"),
                n_seeds=("seed", "nunique"),
            )
            .sort_values(["yyyymm", "permno"])
            .reset_index(drop=True)
        )

        mean_prediction_by_rff[rff_n_component] = mean_pred_df
        port = build_directional_portfolio(mean_pred_df)
        monthly_ret = compute_portfolio_returns(port)
        ss_res = ((mean_pred_df["y_true"] - mean_pred_df["y_pred"]) ** 2).sum()
        ss_tot = (mean_pred_df["y_true"] ** 2).sum()
        oos_r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
        stats = portfolio_performance(monthly_ret)
        stats["oos_r2"] = float(oos_r2) if pd.notna(oos_r2) else np.nan

        portfolio_results_mean_seed[rff_n_component] = {
            "mean_pred_df": mean_pred_df,
            "portfolio": port,
            "returns": monthly_ret,
            "oos_r2": oos_r2,
            "performance": stats,
        }

    perf_table = pd.DataFrame(
        {k: v["performance"] for k, v in portfolio_results_mean_seed.items()}
    ).T.sort_index()
    perf_table.index.name = "rff_n_component"

    diag_stack = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()

    return {
        "oos_r2_results": oos_r2_results,
        "oos_r2_avg": oos_r2_avg,
        "oos_r2_export": oos_r2_export,
        "features_data_df": features_data_df,
        "mean_prediction_by_rff": mean_prediction_by_rff,
        "portfolio_results_mean_seed": portfolio_results_mean_seed,
        "perf_table": perf_table,
        "diag_stack": diag_stack,
    }

if not run_results:
    raise ValueError("run_results is empty. Run the sweep cell or load the saved pickles first.")

analysis_outputs = {}
created_files = []

for (k, T), payload in sorted(run_results.items()):
    summary = summarize_experiment(payload["result_rff_all"])
    analysis_outputs[(k, T)] = summary

    features_export_name = build_experiment_excel_name(
        "features_data",
        k,
        T,
        data_label=payload["data_label"],
    )
    portfolio_export_name = build_experiment_excel_name(
        "portfolio_performance",
        k,
        T,
        data_label=payload["data_label"],
    )
    oos_r2_export_name = build_experiment_excel_name(
        "oos_r2",
        k,
        T,
        data_label=payload["data_label"],
    )

    summary["features_data_df"].to_excel(RESULTS_DIR / features_export_name, index=False)
    summary["perf_table"].to_excel(RESULTS_DIR / portfolio_export_name, index=True)
    summary["oos_r2_export"].assign(k=k, T=T).to_excel(
        RESULTS_DIR / oos_r2_export_name,
        index=False,
    )

    created_files.extend(
        [
            {"artifact": "features", "file": features_export_name, "k": k, "T": T},
            {"artifact": "portfolio", "file": portfolio_export_name, "k": k, "T": T},
            {"artifact": "oos_r2", "file": oos_r2_export_name, "k": k, "T": T},
        ]
    )

ACTIVE_EXPERIMENT = (RUN_CONFIGS[-1]["k"], RUN_CONFIGS[-1]["T"])
if ACTIVE_EXPERIMENT not in analysis_outputs:
    ACTIVE_EXPERIMENT = sorted(analysis_outputs.keys())[0]

pd.DataFrame(created_files).sort_values(["artifact", "k", "T"]).reset_index(drop=True)


# %%


analysis_outputs[ACTIVE_EXPERIMENT]["oos_r2_export"]


# %%


analysis_outputs[ACTIVE_EXPERIMENT]["features_data_df"].head(20)


# %%


analysis_outputs[ACTIVE_EXPERIMENT]["perf_table"]


# %%


pd.DataFrame(created_files).sort_values(["artifact", "k", "T"]).reset_index(drop=True)


# **Single-Experiment OOS R² Diagnostic Plot**

# %%


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

active_summary = analysis_outputs[ACTIVE_EXPERIMENT]
k_active, T_active = ACTIVE_EXPERIMENT
oos_r2_avg = active_summary["oos_r2_avg"]
diag_stack = active_summary["diag_stack"]
n_train = int(diag_stack["n_train"].median()) if not diag_stack.empty else 1

m_vals = np.array(sorted(oos_r2_avg.keys()))
r2_vals = np.array([100 * oos_r2_avg[m] for m in m_vals])
x_vals = 2 * m_vals * k_active / max(n_train, 1)

mask_zero = m_vals == 0
mask_nonzero = m_vals > 0

x_zero = x_vals[mask_zero]
y_zero = r2_vals[mask_zero]
x_nonzero = x_vals[mask_nonzero]
y_nonzero = r2_vals[mask_nonzero]
m_nonzero = m_vals[mask_nonzero]

positive_idx = np.where(y_nonzero > 0)[0]
crossover_x = x_nonzero[positive_idx[0]] if len(positive_idx) else None
crossover_y = y_nonzero[positive_idx[0]] if len(positive_idx) else None

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax_top, ax_bot) = plt.subplots(
    2,
    1,
    figsize=(11, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 4], "hspace": 0.05},
)

line_color = "#e53935"
threshold_color = "#9575cd"
under_color = "#a5d6a7"
over_color = "#ef9a9a"
xmax = max(1.1, x_vals.max() * 1.1)

for ax in (ax_top, ax_bot):
    ax.axvspan(0, 1, color=under_color, alpha=0.10, lw=0)
    ax.axvspan(1, xmax, color=over_color, alpha=0.10, lw=0)
    ax.axvline(1, color=threshold_color, linestyle="--", linewidth=1.8, alpha=0.95)
    ax.grid(True, alpha=0.25)

if len(x_zero):
    ax_top.plot(x_zero, y_zero, marker="X", markersize=14, color=line_color, linewidth=0)
    ax_top.annotate(
        fr"$\tilde{{m}}=0$: {y_zero[0]:.2f}%",
        xy=(x_zero[0], y_zero[0]),
        xytext=(10, -2),
        textcoords="offset points",
        color=line_color,
        fontsize=13,
        va="center",
    )

ax_bot.plot(
    x_nonzero,
    y_nonzero,
    color=line_color,
    linewidth=3,
    marker="D",
    markersize=10,
    markerfacecolor="white",
    markeredgecolor=line_color,
    markeredgewidth=2.5,
)

for m_label in [24, 64, 256, 1024, 4096]:
    if m_label in oos_r2_avg and m_label in m_nonzero:
        i = np.where(m_nonzero == m_label)[0][0]
        ax_bot.annotate(
            fr"$\tilde{{m}}={m_label}$",
            xy=(x_nonzero[i], y_nonzero[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.7),
        )

if crossover_x is not None:
    ax_bot.annotate(
        fr"crossover at $p/n \approx {crossover_x:.2f}$",
        xy=(crossover_x, crossover_y),
        xytext=(crossover_x + 0.16 * xmax, crossover_y + 0.15),
        color=threshold_color,
        fontsize=14,
        arrowprops=dict(arrowstyle="-", color=threshold_color, lw=1.8, alpha=0.75),
    )

ax_top.set_ylim(-4.0, -2.0)
ax_bot.set_ylim(min(0.0, y_nonzero.min() - 0.1 if len(y_nonzero) else 0.0), max(1.05, y_nonzero.max() + 0.1 if len(y_nonzero) else 1.05))
ax_bot.set_xlim(-0.5, xmax)

ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.tick_params(labeltop=False, bottom=False)
ax_bot.xaxis.tick_bottom()

d = 0.015
kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.2)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax_top.set_ylabel(r"OOS $R^2$ (%)", fontsize=16)
ax_bot.set_ylabel(r"OOS $R^2$ (%)", fontsize=16)
ax_bot.set_xlabel(r"$p/n = 2\tilde{m}k / n_{\mathrm{train}}$", fontsize=16)

fig.suptitle(
    fr"OOS $R^2$ crossover: $k={k_active}, T={T_active}, \alpha={ALPHA}$",
    fontsize=20,
    y=0.98,
)

legend_handles = [
    Patch(facecolor=under_color, alpha=0.10, label="under-parameterized"),
    Patch(facecolor=over_color, alpha=0.10, label="over-parameterized"),
    Line2D([0], [0], color=threshold_color, lw=1.8, linestyle="--", label=r"interpolation threshold $p/n=1$"),
]
ax_bot.legend(handles=legend_handles, loc="lower right", fontsize=12, frameon=True)

plt.tight_layout()
_fig_out = Path(os.environ.get("FIGURES_DIR", "../figures"))
_fig_out.mkdir(parents=True, exist_ok=True)
fig.savefig(_fig_out / "ext_plotA.pdf", bbox_inches="tight")
print(f"Saved {_fig_out / 'ext_plotA.pdf'}")
plt.show()

