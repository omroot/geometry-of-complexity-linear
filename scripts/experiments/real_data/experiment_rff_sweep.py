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
# # Running IPCA + RFF geometric analysis of complexity for 500 stocks and 30 years data

# %% [markdown]
# ## Loading data and imports

# %%
pip install autograd

# %%
pip install wrds

# %%
pip install pymanopt

# %%
#imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import missingno as msno
import itertools

import autograd.numpy as anp
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.function import autograd as pymanopt_autograd
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions

import requests
import zipfile
import io
import os
import sys
from pathlib import Path

# Dual-mode path resolution (works in both notebook and script contexts):
# walk up from the script/notebook's location to find the dir containing src/.
if "__file__" in globals():
    _start = Path(__file__).resolve().parent
else:
    _start = Path.cwd()
PROJECT_ROOT = next(p for p in [_start, *_start.parents] if (p / "src").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rff import RandomFourierFeatures

# %%
# load data from ipca_char_data_filtered.parquet

DATA_DIR = ROOT_DIR / "data"
df_char_filtered = pd.read_parquet(DATA_DIR / "ipca_char_data_filtered.parquet")

# %%
df_char_filtered.head()

# %%
# check if any column has nan/missin values
df_char_filtered.isna().sum()
# only print the columns with missing values
print(df_char_filtered.isna().sum()[df_char_filtered.isna().sum() > 0])
#check if these columns with missing values are important for our analysis, if not we can drop them
missing_columns = df_char_filtered.isna().sum()[df_char_filtered.isna().sum() > 0].index
#print(df_char_filtered[missing_columns].head())
#print the dates in whch these columns have missing values
#print(df_char_filtered[df_char_filtered[missing_columns].isna().any(axis=1)].yyyymm.unique())
# find the longest consecutive sequence of missing values in these columns(streak in terms of date)
for col in missing_columns:
    is_missing = df_char_filtered[col].isna()
    max_streak = 0
    current_streak = 0
    for missing in is_missing:
        if missing:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    print(f"Column: {col}, Max Consecutive Missing Values: {max_streak}")
#show the number of missing values per permno
missing_by_permno = df_char_filtered.groupby("permno")[missing_columns].apply(lambda x: x.isna().sum())
#show in the order of decreasing number of missing values
missing_by_permno = missing_by_permno.sort_values(by=missing_columns.tolist(), ascending=False)
print(missing_by_permno.head(20))  

# %%
#Lets drop the top4 and fill the others with previous values
permnos_to_drop = missing_by_permno.head(4).index
df_char_filtered = df_char_filtered[~df_char_filtered.permno.isin(permnos_to_drop)]
df_char_filtered[missing_columns] = df_char_filtered[missing_columns].ffill()


# %%
import sys
import importlib
from pathlib import Path
import types

from src.sweeps.grassmannian_worker import (
    build_rff_inputs, build_jobs, run_sweep, aggregate, make_results_df,
    Z_VALUES, _z_label,
    GAMMA, WINDOW_LEN, N_FEATURES_RFF, NUM_FACTORS_LIST, NUM_ITER_RFF,
)
# Ensure src/ is on sys.path so top-level modules in src can be imported
src_dir = ROOT_DIR / "src" #this seems to be the only way to reliably import from src/ in both .ipynb and .py contexts without causing import errors in one or the other
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Try to import the real module; if missing, install a lightweight shim to avoid ImportError
try:
    backtest_GRASS_IPCA = importlib.import_module("backtest_GRASS_IPCA")
except Exception:
    shim = types.ModuleType("backtest_GRASS_IPCA")
    def run_backtest(*args, **kwargs):
        raise RuntimeError("backtest_GRASS_IPCA.run_backtest is not available in this environment.")
    shim.run_backtest = run_backtest
    # add any other expected symbols here as simple stubs if you discover they're required
    sys.modules["backtest_GRASS_IPCA"] = shim
    backtest_GRASS_IPCA = shim
    print("Warning: using shim for backtest_GRASS_IPCA (real module not found).")

# %%
# Convert yyyymm → datetime, set MultiIndex (date, permno)
df = df_char_filtered.copy()
df["date"] = pd.to_datetime(df["yyyymm"])
df = df.drop(columns=["yyyymm"]).set_index(["date", "permno"]).sort_index()

# Create y_ipca (next-period return target) if not already in file
if "y_ipca" not in df.columns:
    df["y_ipca"] = df.groupby(level="permno")["excess_ret"].shift(-1)

# Characteristic columns (all except the return target and raw return)
char_cols = [c for c in df.columns if c not in ("y_ipca", "excess_ret")]
print(f"Characteristic columns ({len(char_cols)}): {char_cols[:6]} ...")

# # Lag characteristics 1 period within each permno (Z_{t-1} predicts r_t)
# df[char_cols] = df[char_cols].groupby(level="permno").shift(1)

# Rename return column to the convention expected by run_ipca_grassmannian_v2
df = df.rename(columns={"y_ipca": "ret"})
df = df.dropna(subset=["ret"])

# build_rff_inputs expects df_base to have a "Price" column alongside "ret";
# it is not used by the backtest — add a dummy so the API stays unchanged.
df["Price"] = 1.0

print(f"Panel shape   : {df.shape}")
print(f"Date range    : {df.index.get_level_values('date').min().date()} → "
      f"{df.index.get_level_values('date').max().date()}")
print(f"Unique permnos: {df.index.get_level_values('permno').nunique()}")

# %% [markdown]
# ### RFF expansion of characteristics

# %%
# ===========================================================================
# 1. (OPTIONAL) OVERRIDE GRID DEFAULTS
#    Uncomment / edit any line to override the defaults from _grass_worker.py
# ===========================================================================
N_FEATURES_RFF   = [256, 1024, 4096]   # low/mid/high — captures P-scaling cleanly
NUM_FACTORS_LIST = [4, 8, 16, 32]      # brackets the expected k₀ for 500 stocks
NUM_ITER_RFF     = 3                   # keep at 3, enough to average out RFF noise
WINDOW_LEN       = 12                  # keep as is
GAMMA            = 0.25                # keep as is
Z_VALUES         = [0, 10, 100]        # zero (illusory baseline), sweet spot, over-regularised       # 0.0 = OLS baseline; add e.g. 1e-2, 1.0 for shrinkage

# ===========================================================================
# 2. PREPARE INPUT DATA
#    Dense char matrix for RFF — impute cross-sectionally, then col-wise fallback
# ===========================================================================
date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)        # ensure float64 — required for np.sin/cos in RFF
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")


# %%
X_chars

# %% [markdown]
# ### Rolling IPCA + RFF parallel sweep
#
# Mirrors the cell-40 pattern from `VOC_everywhere.ipynb`:
# 1. `build_rff_inputs` — pre-compute all (n_feat × seed) RFF DataFrames
# 2. `build_jobs` — build full (k, P, z, seed) Cartesian job list, sorted longest-first (LPT)
# 3. `run_sweep` — dispatch in parallel via joblib/loky
# 4. `aggregate` + `make_results_df` — average over seeds → MultiIndex summary

# %%
# # ===========================================================================
# # 3. RUN PIPELINE
# # ===========================================================================
# print("Pre-computing RFF features ...")
# rff_inputs = build_rff_inputs(
#     X_chars               = X_chars,
#     df_base               = df,          # must contain ["Price", "ret"]
#     n_features_rff        = N_FEATURES_RFF,
#     num_iter_rff          = NUM_ITER_RFF,
#     gamma                 = GAMMA,
#     RandomFourierFeatures = RandomFourierFeatures,   # already imported in cell 2
# )
# print(f"  {len(rff_inputs)} RFF datasets ready.")

# jobs = build_jobs(
#     rff_inputs       = rff_inputs,
#     num_factors_list = NUM_FACTORS_LIST,
#     n_features_rff   = N_FEATURES_RFF,
#     z_values         = Z_VALUES,
#     num_iter_rff     = NUM_ITER_RFF,
#     window_len       = WINDOW_LEN,
# )

# raw_results           = run_sweep(jobs)
# results_by_factor_rff = aggregate(raw_results)
# results_rff_df        = make_results_df(results_by_factor_rff)
# display(results_rff_df)

# from src.io.results_logger import append_results
# append_results(
#     results_rff_df,
#     num_stocks        = len(df.index.get_level_values('permno').unique()),
#     window_len        = WINDOW_LEN,
#     gamma             = GAMMA,
#     num_iter_rff      = NUM_ITER_RFF,
#     run_label         = 'Run1',
# )


# %% [markdown]
# ### Results — plots

# %%
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle(
#     f"IPCA + RFF on 500 stocks (1994-2025)\n"
#     f"window_len={WINDOW_LEN} months, gamma={GAMMA}, z={Z_VALUES}",
#     fontsize=13,
# )

# metrics = [
#     ("avg_r2_oos",               "OOS R²"),
#     ("avg_sharpe",               "Sharpe (annualised)"),
#     ("avg_subspace_stability",   "Mean d_proj  (↓ better)"),
#     ("avg_erank",                "Mean erank(Σ̂_f)  (↑ better)"),
#     ("avg_spectral_gap",         "Spectral gap ratio  (↑ better)"),
#     ("avg_max_principal_angle",  "Max principal angle rad  (↓ better)"),
# ]

# for ax, (col, title) in zip(axes.flat, metrics):
#     for nf in NUM_FACTORS_LIST:
#         subset = results_rff_df.xs(nf, level="k")[col]
#         ax.plot(subset.index.get_level_values("P"), subset.values,
#                 marker="o", label=f"k={nf}")
#     ax.set_title(title)
#     ax.set_xlabel("N_FEATURES_RFF  (P)")
#     ax.set_xscale("log")
#     ax.legend(fontsize=8)
#     ax.grid(True, alpha=0.4)

# plt.tight_layout()
# plt.show()


# %% [markdown]
# ### Run with top50 stocks

# %% [markdown]
# ### Loading top50(mkt cap) stocks

# %%
import wrds
import pandas as pd

db = wrds.Connection()

# Get the 496 permnos already in your panel
panel_permnos = df_char_filtered["permno"].unique().tolist()
permnos_str = ",".join(map(str, panel_permnos))

sql = f"""
    SELECT
        msf.permno,
        msf.date,
        ABS(msf.prc) * msf.shrout AS me
    FROM crsp.msf
    JOIN crsp.msenames AS names
      ON msf.permno = names.permno
     AND msf.date BETWEEN names.namedt AND names.nameendt
    WHERE msf.date    >= '1994-01-01'
      AND msf.date    <= '2025-12-31'
      AND msf.permno  IN ({permnos_str})
"""
df_me = db.raw_sql(sql, date_cols=["date"])
db.close()

top50_permnos = (
    df_me.groupby("permno")["me"]
    .mean()
    .nlargest(50)
    .index.tolist()
)

df_top50 = df_char_filtered[df_char_filtered["permno"].isin(top50_permnos)]
print(f"Stocks: {len(top50_permnos)}, Panel shape: {df_top50.shape}")



# %%
''' Uncomment to check the name sof top5 0 stocks'''
# import wrds

# db = wrds.Connection()

# permnos_str = ",".join(map(str, top50_permnos))

# name_map = db.raw_sql(f"""
#     SELECT DISTINCT ON (permno) permno, ticker, comnam
#     FROM crsp.msenames
#     WHERE permno IN ({permnos_str})
#     ORDER BY permno, namedt DESC   -- most recent name entry
# """)

# print(name_map.sort_values("comnam").to_string(index=False))


# %%
# name_map.head()


# %% [markdown]
#

# %%
# #lets see if there are any missing nans in df_top50
# print(df_top50.isna().sum()[df_top50.isna().sum() > 0])

# %%
# prepare panel like above for the top50 stocks and run the backtest on this smaller universe to see if we can get better convergence with more RFF features and more iterations
df = df_top50.copy()
df["date"] = pd.to_datetime(df["yyyymm"])
df = df.drop(columns=["yyyymm"]).set_index(["date", "permno"]).sort_index()

# Characteristic columns (all except the return target)
char_cols = [c for c in df.columns if c not in ("y_ipca", "excess_ret")]
print(f"Characteristic columns ({len(char_cols)}): {char_cols[:6]} ...")

# # Lag characteristics 1 period within each permno (Z_{t-1} predicts r_t)
# df[char_cols] = df[char_cols].groupby(level="permno").shift(1)

# Rename return column to the convention expected by run_ipca_grassmannian_v2
df = df.rename(columns={"y_ipca": "ret"})
df = df.dropna(subset=["ret"])

# build_rff_inputs expects df_base to have a "Price" column alongside "ret";
# it is not used by the backtest — add a dummy so the API stays unchanged.
df["Price"] = 1.0

print(f"Panel shape   : {df.shape}")
print(f"Date range    : {df.index.get_level_values('date').min().date()} → "
      f"{df.index.get_level_values('date').max().date()}")
print(f"Unique permnos: {df.index.get_level_values('permno').nunique()}")

# %% [markdown]
# ## Different simulations with top50 stocks

# %% [markdown]
# ### Run1 : k = {1, 4, 8, 12, 16, 24}, z: {0,10}

# %%
# these will have one nan per columns, as we shifted to make Z_t-1) predict r_t, so this is expected


# %%
# ===========================================================================
# 1. (OPTIONAL) OVERRIDE GRID DEFAULTS
#    Uncomment / edit any line to override the defaults from _grass_worker.py
# ===========================================================================
NUM_FACTORS_LIST = [4, 8, 12, 16, 24]     # below k₀, at k₀, above k₀
Z_VALUES         = [0, 10]         # illusory baseline vs virtuous
N_FEATURES_RFF   = [1024]          # P-independence already established
NUM_ITER_RFF     = 3
WINDOW_LEN       = 12
GAMMA            = 0.25
# ===========================================================================
# 2. PREPARE INPUT DATA
#    Dense char matrix for RFF — impute cross-sectionally, then col-wise fallback
# ===========================================================================
date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)        # ensure float64 — required for np.sin/cos in RFF
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
# ===========================================================================
# 3. RUN PIPELINE
# ===========================================================================
print("Pre-computing RFF features ...")
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,          # must contain ["Price", "ret"]
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,   # already imported in cell 2
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs       = rff_inputs,
    num_factors_list = NUM_FACTORS_LIST,
    n_features_rff   = N_FEATURES_RFF,
    z_values         = Z_VALUES,
    num_iter_rff     = NUM_ITER_RFF,
    window_len       = WINDOW_LEN,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    run_label         = 'Run2',
)


# %% [markdown]
# ### Run2  z = {1, 5, 25, 100} atk=12, T = 24

# %%
# ===========================================================================
# 1. (OPTIONAL) OVERRIDE GRID DEFAULTS
#    Uncomment / edit any line to override the defaults from _grass_worker.py
# ===========================================================================
NUM_FACTORS_LIST = [12]     # below k₀, at k₀, above k₀
Z_VALUES         = [1, 5, 25, 100]         # illusory baseline vs virtuous
N_FEATURES_RFF   = [1024]          # P-independence already established
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25
# ===========================================================================
# 2. PREPARE INPUT DATA
#    Dense char matrix for RFF — impute cross-sectionally, then col-wise fallback
# ===========================================================================
date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)        # ensure float64 — required for np.sin/cos in RFF
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
# ===========================================================================
# 3. RUN PIPELINE
# ===========================================================================
print("Pre-computing RFF features ...")
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,          # must contain ["Price", "ret"]
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,   # already imported in cell 2
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs       = rff_inputs,
    num_factors_list = NUM_FACTORS_LIST,
    n_features_rff   = N_FEATURES_RFF,
    z_values         = Z_VALUES,
    num_iter_rff     = NUM_ITER_RFF,
    window_len       = WINDOW_LEN,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    run_label         = 'Run3',
)


# %%
results_rff_df

# %%

# %% [markdown]
# ### Run3 : NUM_FACTORS_LIST = [4, 8, 12, 16, 24], Z_VALUES         = [20]

# %%
# ===========================================================================
# 1. (OPTIONAL) OVERRIDE GRID DEFAULTS
#    Uncomment / edit any line to override the defaults from _grass_worker.py
# ===========================================================================
NUM_FACTORS_LIST = [4, 8, 12, 16, 24]     # below k₀, at k₀, above k₀
Z_VALUES         = [20]         # illusory baseline vs virtuous
N_FEATURES_RFF   = [1024]          # P-independence already established
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25
# ===========================================================================
# 2. PREPARE INPUT DATA
#    Dense char matrix for RFF — impute cross-sectionally, then col-wise fallback
# ===========================================================================
date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)        # ensure float64 — required for np.sin/cos in RFF
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
# ===========================================================================
# 3. RUN PIPELINE
# ===========================================================================
print("Pre-computing RFF features ...")
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,          # must contain ["Price", "ret"]
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,   # already imported in cell 2
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs       = rff_inputs,
    num_factors_list = NUM_FACTORS_LIST,
    n_features_rff   = N_FEATURES_RFF,
    z_values         = Z_VALUES,
    num_iter_rff     = NUM_ITER_RFF,
    window_len       = WINDOW_LEN,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    run_label         = 'Run4',
)


# %%

# %% [markdown]
# ### Run4 : n_rff : N_FEATURES_RFF   = [16, 64,256,1024, 4096]  

# %% [markdown]
#

# %%
# ===========================================================================
# 1. (OPTIONAL) OVERRIDE GRID DEFAULTS
#    Uncomment / edit any line to override the defaults from _grass_worker.py
# ===========================================================================
NUM_FACTORS_LIST = [24]     # below k₀, at k₀, above k₀
Z_VALUES         = [20]         # illusory baseline vs virtuous
N_FEATURES_RFF   = [64,256,1024, 4096]         
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25
# ===========================================================================
# 2. PREPARE INPUT DATA
#    Dense char matrix for RFF — impute cross-sectionally, then col-wise fallback
# ===========================================================================
date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)        # ensure float64 — required for np.sin/cos in RFF
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
# ===========================================================================
# 3. RUN PIPELINE
# ===========================================================================
print("Pre-computing RFF features ...")
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,          # must contain ["Price", "ret"]
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,   # already imported in cell 2
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs       = rff_inputs,
    num_factors_list = NUM_FACTORS_LIST,
    n_features_rff   = N_FEATURES_RFF,
    z_values         = Z_VALUES,
    num_iter_rff     = NUM_ITER_RFF,
    window_len       = WINDOW_LEN,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    run_label         = 'Run5',
)


# %%

# %% [markdown]
# ### Run 5 k sweep = [28, 32, 36], z_values = [0, 10]

# %%
# ===========================================================================
# 1. (OPTIONAL) OVERRIDE GRID DEFAULTS
#    Uncomment / edit any line to override the defaults from _grass_worker.py
# ===========================================================================
NUM_FACTORS_LIST = [28, 32]     # below k₀, at k₀, above k₀
Z_VALUES         = [0, 10]         # illusory baseline vs virtuous
N_FEATURES_RFF   = [1024]          # P-independence already established
NUM_ITER_RFF     = 3
WINDOW_LEN       = 12
GAMMA            = 0.25
# ===========================================================================
# 2. PREPARE INPUT DATA
#    Dense char matrix for RFF — impute cross-sectionally, then col-wise fallback
# ===========================================================================
date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)        # ensure float64 — required for np.sin/cos in RFF
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
# ===========================================================================
# 3. RUN PIPELINE
# ===========================================================================
print("Pre-computing RFF features ...")
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,          # must contain ["Price", "ret"]
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,   # already imported in cell 2
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs       = rff_inputs,
    num_factors_list = NUM_FACTORS_LIST,
    n_features_rff   = N_FEATURES_RFF,
    z_values         = Z_VALUES,
    num_iter_rff     = NUM_ITER_RFF,
    window_len       = WINDOW_LEN,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    run_label         = 'Run6',
)


# %% [markdown]
# #### important  observation -
#  While erank is a good measure for a given k, It seems to be increasig with a lot, just from the extra spread we get in higher dimensions even though higher k isnt helping the oos performance of the model perse
#  Conclusion? While this maybe a greate measure to compare stability acroros models for a given/fixed k, If we want to compare strategies across k, spetral gap may be a better measure, rate of decrease in spectral gap to be precise?

# %% [markdown]
# 2. Across k, spectral gap seems to be a reasonable measure of stability, within k erank is good

# %% [markdown]
# ### run6: k = {12, 24}, z = {50, 100}
#

# %% [markdown]
#

# %%

# ===========================================================================
# 1. (OPTIONAL) OVERRIDE GRID DEFAULTS
#    Uncomment / edit any line to override the defaults from _grass_worker.py
# ===========================================================================
NUM_FACTORS_LIST = [20, 24]     # below k₀, at k₀, above k₀
Z_VALUES         = [50, 100]         # illusory baseline vs virtuous
N_FEATURES_RFF   = [4096]          # P-independence already established
NUM_ITER_RFF     = 3
WINDOW_LEN       = 12
GAMMA            = 0.25
# ===========================================================================
# 2. PREPARE INPUT DATA
#    Dense char matrix for RFF — impute cross-sectionally, then col-wise fallback
# ===========================================================================
date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)        # ensure float64 — required for np.sin/cos in RFF
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
# ===========================================================================
# 3. RUN PIPELINE
# ===========================================================================
print("Pre-computing RFF features ...")
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,          # must contain ["Price", "ret"]
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,   # already imported in cell 2
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    min_gradient_norm = 1e-4,
    run_label         = 'Run7',
)


# %% [markdown]
# ### Combined Results — All Longer Backtest Runs (Top-50 Stocks, 1994–2025)
#
# Consolidates Runs 1–6 into a single sheet with `T` (window length) and `run` labels added.  
# Derived column `erank_over_k` = erank / k (normalised effective rank, scale-invariant across k).

# %%
import pandas as pd, numpy as np

# ── All 30 data points from Runs 1-6, hand-verified from output tables ──
_cols = ["run", "k", "P", "z", "T",
         "avg_r2_oos", "avg_sharpe", "avg_subspace_stability",
         "avg_max_principal_angle", "avg_mean_principal_angle",
         "avg_geodesic_accel", "avg_erank", "avg_erank_collapse",
         "avg_spectral_gap"]

_rows = [
    # ── Run1: k sweep × z={0,10}, P=1024, T=12 ──────────────────────────
    ("Run1",  4, 1024,   0, 12, -0.068738, 0.033662, 0.588709, 0.396841, 0.162658, 0.141023,  3.382539, 0.002778, 2.209187e-01),
    ("Run1",  4, 1024,  10, 12, -0.039922, 0.124149, 1.237812, 0.802710, 0.401132, 0.114294,  3.896735, 0.000000, 5.742925e-01),
    ("Run1",  8, 1024,   0, 12, -0.057367, 0.204040, 0.454414, 0.312912, 0.064608, 0.118081,  5.580585, 0.018519, 4.193812e-02),
    ("Run1",  8, 1024,  10, 12, -0.032434, 0.124780, 1.399760, 0.931821, 0.282069, 0.114831,  7.413556, 0.000926, 3.069561e-01),
    ("Run1", 12, 1024,   0, 12, -0.071727, 0.003268, 0.344688, 0.238305, 0.031907, 0.091721,  6.591813, 0.201852, 1.341996e-03),
    ("Run1", 12, 1024,  10, 12, -0.029523, 0.230358, 1.389662, 1.044526, 0.191994, 0.131073,  9.536090, 0.017593, 4.654315e-02),
    ("Run1", 16, 1024,   0, 12, -0.087863,-0.053282, 0.265426, 0.183657, 0.017896, 0.067055,  7.245099, 0.842593, 9.301977e-20),
    ("Run1", 16, 1024,  10, 12, -0.029346, 0.222280, 1.348005, 1.101556, 0.127026, 0.115175,  9.799155, 0.039815, 2.700167e-19),
    ("Run1", 24, 1024,   0, 12, -0.131690, 0.212184, 0.166530, 0.115938, 0.006965, 0.046309,  7.838482, 1.000000, 0.000000e+00),
    ("Run1", 24, 1024,  10, 12, -0.029227, 0.235614, 1.322483, 1.119238, 0.077522, 0.097656,  9.829161, 1.000000, 0.000000e+00),
    # ── Run2: z sweep at k=12, P=1024, T=24 ─────────────────────────────
    ("Run2", 12, 1024,   1, 24, -0.025695, 0.128234, 0.702229, 0.476902, 0.087819, 0.087755, 10.190136, 0.000000, 0.125995),
    ("Run2", 12, 1024,   5, 24, -0.025263, 0.136353, 1.091539, 0.697246, 0.162069, 0.119769, 11.261268, 0.000000, 0.307757),
    ("Run2", 12, 1024,  20, 24, -0.024232, 0.210213, 1.523093, 0.926063, 0.239712, 0.129528, 11.476672, 0.000000, 0.413193),
    ("Run2", 12, 1024, 100, 24, -0.023507, 0.228131, 1.680029, 1.077929, 0.271522, 0.182214, 11.453914, 0.000000, 0.416803),
    # ── Run3: k sweep at z=20, P=1024, T=24 ─────────────────────────────
    ("Run3",  4, 1024,  20, 24, -0.028626, 0.205029, 1.207916, 0.665261, 0.420080, 0.195350,  3.949825, 0.000000, 0.682406),
    ("Run3",  8, 1024,  20, 24, -0.025686, 0.139891, 1.365448, 0.775875, 0.295348, 0.132879,  7.804608, 0.000000, 0.530543),
    ("Run3", 12, 1024,  20, 24, -0.024179, 0.199393, 1.489824, 0.900135, 0.232489, 0.125496, 11.474806, 0.000000, 0.410738),
    ("Run3", 16, 1024,  20, 24, -0.023297, 0.269948, 1.595065, 1.032811, 0.194694, 0.124586, 14.761079, 0.004789, 0.289968),
    ("Run3", 24, 1024,  20, 24, -0.022573, 0.283874, 1.535360, 1.201785, 0.116767, 0.140719, 18.656600, 0.035441, 0.038902),
    # ── Run4: P sweep at k=24, z=20, T=24 ───────────────────────────────
    ("Run4", 24,   64,  20, 24, -0.034993, 0.029264, 1.419981, 1.438387, 0.083558, 0.033678, 16.232313, 0.068966, 0.027958),
    ("Run4", 24,  256,  20, 24, -0.025566, 0.322742, 1.431380, 1.489980, 0.079834, 0.027295, 18.172800, 0.045019, 0.066218),
    ("Run4", 24, 1024,  20, 24, -0.022553, 0.292026, 1.533880, 1.201230, 0.116747, 0.146800, 18.654884, 0.036398, 0.038813),
    ("Run4", 24, 4096,  20, 24, -0.021038, 0.425746, 1.168918, 0.865519, 0.087508, 0.131056, 17.809959, 0.028736, 0.003130),
    # ── Run5: high-k stress test, P=1024, T=12 ──────────────────────────
    ("Run5", 28, 1024,   0, 12, -0.176347,-0.032572, 0.124225, 0.086697, 0.004288, 0.036144,  7.991680, 1.000000, 0.000000),
    ("Run5", 28, 1024,  10, 12, -0.029199, 0.236002, 1.314867, 1.120466, 0.064850, 0.091851,  9.832119, 1.000000, 0.000000),
    ("Run5", 32, 1024,   0, 12, -0.245029, 0.060255, 0.092439, 0.064689, 0.002693, 0.026888,  8.060311, 1.000000, 0.000000),
    ("Run5", 32, 1024,  10, 12, -0.029209, 0.234576, 1.308373, 1.117502, 0.055888, 0.090320,  9.833184, 1.000000, 0.000000),
    # ── Run6: high-z + high-P, T=12 ─────────────────────────────────────
    ("Run6", 20, 4096,  50, 12, -0.025457, 0.428213, 1.390612, 1.267997, 0.092826, 0.073419,  9.874499, 0.482407, 0.000000),
    ("Run6", 20, 4096, 100, 12, -0.025342, 0.432275, 1.412940, 1.458871, 0.083848, 0.020042,  9.856880, 0.487037, 0.000000),
    ("Run6", 24, 4096,  50, 12, -0.025427, 0.428969, 1.384300, 1.272153, 0.075312, 0.068956,  9.873561, 1.000000, 0.000000),
    ("Run6", 24, 4096, 100, 12, -0.025335, 0.433658, 1.410176, 1.453405, 0.068983, 0.017149,  9.855667, 1.000000, 0.000000),
]

combined = pd.DataFrame(_rows, columns=_cols)

# ── Derived columns ──────────────────────────────────────────────────────
combined["erank_over_k"]   = combined["avg_erank"] / combined["k"]
combined["spectral_gap_pct"] = combined["avg_spectral_gap"] * 100  # easier to read

# ── Sort for readability ─────────────────────────────────────────────────
combined = combined.sort_values(["T", "k", "z", "P"]).reset_index(drop=True)

# ── Save to CSV ──────────────────────────────────────────────────────────
out_path = ROOT_DIR / "notebooks" / "combined_longer_backtest_results.csv"
combined.to_csv(out_path, index=False, float_format="%.6f")
print(f"Saved {len(combined)} rows → {out_path.name}")

# ── Display ──────────────────────────────────────────────────────────────
display_cols = ["run", "k", "P", "z", "T",
                "avg_r2_oos", "avg_sharpe",
                "avg_subspace_stability", "avg_spectral_gap",
                "avg_erank", "erank_over_k", "avg_erank_collapse"]

with pd.option_context("display.max_rows", 40, "display.float_format", "{:.4f}".format):
    display(combined[display_cols])

# %%
# ── Full combined table (all columns) ────────────────────────────────────
with pd.option_context("display.max_rows", 40, "display.max_columns", 20,
                        "display.float_format", "{:.4f}".format):
    display(combined)

# %%
# ── Coverage heatmap: which (k, z) cells exist at each T? ────────────────
print("=" * 60)
print("  COVERAGE MAP — (k, z) cells by window length T")
print("=" * 60)
for t_val in sorted(combined["T"].unique()):
    sub = combined[combined["T"] == t_val]
    pivot = sub.pivot_table(index="k", columns="z", values="avg_sharpe",
                            aggfunc="count", fill_value=0)
    print(f"\n  T = {t_val}  ({len(sub)} cells)")
    print(pivot.to_string())

print("\n" + "=" * 60)
print("  COVERAGE MAP — (k, P) cells by T")
print("=" * 60)
for t_val in sorted(combined["T"].unique()):
    sub = combined[combined["T"] == t_val]
    pivot = sub.pivot_table(index="k", columns="P", values="avg_sharpe",
                            aggfunc="count", fill_value=0)
    print(f"\n  T = {t_val}  ({len(sub)} cells)")
    print(pivot.to_string())

# %% [markdown]
# ### Gap Analysis & Suggested Next Runs
#
# **Current coverage** — 30 cells across 6 runs:
#
# | T | Coverage | What's there |
# |---|---------|-------------|
# | 12 | 18 cells | k={4–32} × z={0,10} at P=1024; k={20,24} × z={50,100} at P=4096 |
# | 24 | 12 cells | k=12 × z={1,5,20,100}; k={4–24} × z=20; k=24 × P={64–4096} |
#
# **Critical gaps:**
#
# 1. **z=10 at T=24 is completely missing.** z=10 was the clear sweet spot in the shorter backtest (Sharpe=0.591 at k=8). We have z=10 only at T=12 (Run1). Without T=24 at z=10 we cannot tell whether the z* threshold shifts with window length.
#
# 2. **z=0 (illusory baseline) at T=24 is missing.** We cannot compare virtuous vs illusory regimes at T=24 without it.
#
# 3. **P variation at T=24** exists only for k=24. No P sweep at lower k values.
#
# ---
#
# #### Suggested Run 7 — *highest priority, fills the biggest gap*
#
# ```
# k = {4, 12, 24},  z = {0, 10},  P = 1024,  T = 24
# →  6 jobs
# ```
#
# **Why:** Directly mirrors Run1 (T=12) at T=24 for three representative k values (low / mid / high). Gives the z=0 vs z=10 comparison at T=24, and enables a clean T=12 vs T=24 panel with all else held fixed. This is the single most informative run.
#
# #### Suggested Run 8 — *second priority, tests P-scaling at the current best configs*
#
# ```
# k = {12, 16},  z = 20,  P = 4096,  T = 24
# →  2 jobs
# ```
#
# **Why:** Run4 showed P=4096 gives the best Sharpe (0.43) at k=24. But we don't know if that holds at lower k where the spectral gap is healthy. k=12 and k=16 are in the "sweet zone" where sg > 0.25. This tests whether the P-scaling virtue persists when factor structure is genuine.
#
# #### Suggested Run 9 — *optional, completes the z grid at T=24*
#
# ```
# k = {4, 12, 24},  z = {10, 50},  P = 4096,  T = 24
# →  6 jobs
# ```
#
# **Why:** Tests the "productive illusion" finding (Run6: sg=0 but Sharpe=0.43) at T=24. If the high-z + high-P performance persists at longer windows, it's a real phenomenon, not a short-window artefact.

# %% [markdown]
# ### Run7 — z={0,10} × k={4,12,24}, P=1024, T=24  *(ready to run)*

# %%
# ===========================================================================
# Run7: fills the biggest gap — z=10 at T=24 (+ z=0 baseline)
# ===========================================================================
NUM_FACTORS_LIST = [4, 12, 24]
Z_VALUES         = [0, 10]
N_FEATURES_RFF   = [1024]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
# ===========================================================================
# Run7: EXECUTE
# ===========================================================================
print("Pre-computing RFF features ...")
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    min_gradient_norm = 1e-4,
    run_label         = 'Run8',
)


# %% [markdown]
# ### Run8 — k={12,16}, z=20, P=4096, T=24  *(P-scaling at healthy sg)*

# %%
# ===========================================================================
# Run8: P-scaling test where spectral gap is still meaningful
# ===========================================================================
NUM_FACTORS_LIST = [12, 16]
Z_VALUES         = [20]
N_FEATURES_RFF   = [4096]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
# ===========================================================================
# Run8: EXECUTE
# ===========================================================================
print("Pre-computing RFF features ...")
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    min_gradient_norm = 1e-4,
    run_label         = 'Run9',
)


# %% [markdown]
# ### Run 9 k = {4, 12, 24},  z = {10, 50},  P = 4096,  T = 24

# %%
NUM_FACTORS_LIST = [4, 12, 24]
Z_VALUES         = [10,50]
N_FEATURES_RFF   = [4096]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    min_gradient_norm = 1e-4,
    run_label         = 'Run10',
)


# %% [markdown]
# ### Run 10 k = {4, 12, 24},  z = {10, 50},  P = 6000,  T = 24

# %%
NUM_FACTORS_LIST = [24, 12, 4]
Z_VALUES         = [10,50]
N_FEATURES_RFF   = [6000]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    min_gradient_norm = 1e-4,
    run_label         = 'Run11',
)


# %%
#lets see the data from results_rff_log.parquet
log_path = ROOT_DIR / "cache" / "results_rff_log.parquet"
log_df = pd.read_parquet(log_path)
#display(log_df)

# %% [markdown]
# ### Run 11 - k = {8, 16}, z = {0, 10,20, 50}, P = [1024], window_len = 24

# %%
NUM_FACTORS_LIST = [8, 16]
Z_VALUES         = [0, 10, 20, 50]
N_FEATURES_RFF   = [1024]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

from src.io.results_logger import append_results
append_results(
    results_rff_df,
    num_stocks        = len(df.index.get_level_values('permno').unique()),
    window_len        = WINDOW_LEN,
    gamma             = GAMMA,
    num_iter_rff      = NUM_ITER_RFF,
    min_gradient_norm = 1e-4,
    run_label         = 'Run11',
)

# %%
# lets load the results from results_rff_log.parquet again and see if the new results are there
log_path = ROOT_DIR / "cache" / "results_rff_log.parquet"
log_df = pd.read_parquet(log_path)
display(log_df)
#lets copy the dataframe to clipboard
log_df.to_clipboard(index=False)

# %% [markdown]
#

# %% [markdown]
# ### Run 11 - k = {20, 28, 32}, z = {0, 10, 20}, P = [1024], window_len = 24

# %%
NUM_FACTORS_LIST = [20, 28, 32]
Z_VALUES         = [0, 10, 20]
N_FEATURES_RFF   = [6000]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ### Run 12 - k = {12, 16, 24}, z = {0, 10,}, P = [8000], window_len = 24

# %%
NUM_FACTORS_LIST = [12, 16, 24]
Z_VALUES         = [0, 10]
N_FEATURES_RFF   = [8000]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ### Run 13 - k = {20, 28}, z = {0, 10,}, P = [8000], window_len = 24

# %%
NUM_FACTORS_LIST = [20, 28]
Z_VALUES         = [0, 10]
N_FEATURES_RFF   = [8000]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%

# %% [markdown]
# ### Run 14 : 

# %%
NUM_FACTORS_LIST = [12, 16, 24]
Z_VALUES         = [50, 100]
N_FEATURES_RFF   = [8000]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%

# %% [markdown]
# ### Run 15 : P = 8k, z = {50, 100}

# %%
NUM_FACTORS_LIST = [20, 28]
Z_VALUES         = [50, 100]
N_FEATURES_RFF   = [10000]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%
import pandas as pd
import numpy as np
from pathlib import Path

# ── 1. Load the logged results (Runs 1–10, 49 rows) ──────────────────────────
ROOT_DIR = Path("..")
log_df = pd.read_parquet(ROOT_DIR / "cache" / "results_rff_log.parquet")

# ── 2. New per-run results not yet logged to parquet ──────────────────────────
cols = [
    "k", "P", "z", "run_label",
    "avg_r2_oos", "avg_sharpe", "avg_subspace_stability",
    "avg_max_principal_angle", "avg_mean_principal_angle",
    "avg_geodesic_accel", "avg_erank", "avg_erank_collapse", "avg_spectral_gap",
]

new_data = [
    # Run11a — k={8,16}, P=1024, z={0,10,20,50}
    ( 8, 1024,  0.0, "Run11a", -0.043253,  0.173987, 0.432091, 0.290397, 0.068076, 0.094777,  6.483162, 0.004789, 0.123801),
    ( 8, 1024, 10.0, "Run11a", -0.025977,  0.225183, 0.892737, 0.565022, 0.171611, 0.195868,  7.624488, 0.000000, 0.386694),
    ( 8, 1024, 20.0, "Run11a", -0.026424,  0.076923, 1.106930, 0.651555, 0.225930, 0.157386,  7.761711, 0.000000, 0.484364),
    ( 8, 1024, 50.0, "Run11a", -0.025459,  0.107487, 1.305106, 0.756775, 0.275683, 0.187245,  7.794596, 0.000000, 0.521718),
    (16, 1024,  0.0, "Run11a", -0.054048,  0.145245, 0.265427, 0.182086, 0.020536, 0.064722, 10.040398, 0.056513, 0.013109),
    (16, 1024, 10.0, "Run11a", -0.024450,  0.180271, 0.831223, 0.576320, 0.083812, 0.224845, 12.807839, 0.000000, 0.058621),
    (16, 1024, 20.0, "Run11a", -0.023861,  0.224426, 1.097350, 0.748039, 0.124745, 0.301335, 13.867329, 0.002874, 0.120313),
    (16, 1024, 50.0, "Run11a", -0.023250,  0.251982, 1.499874, 0.957369, 0.184581, 0.233587, 14.646912, 0.007663, 0.254532),
    # Run11b — k={20,28,32}, P=6000, z={0,10,20}
    (20, 6000,  0.0, "Run11b", -0.062825,  0.084933, 0.089372, 0.062058, 0.004535, 0.022135, 11.256407, 0.083333, 3.227031e-03),
    (20, 6000, 10.0, "Run11b", -0.022391,  0.296530, 0.411029, 0.286152, 0.025358, 0.113614, 12.683253, 0.000958, 5.477980e-03),
    (20, 6000, 20.0, "Run11b", -0.022222,  0.266937, 0.550187, 0.387583, 0.034801, 0.149173, 13.130887, 0.001916, 7.162304e-03),
    (28, 6000,  0.0, "Run11b", -0.103037,  0.122190, 0.051709, 0.036082, 0.001777, 0.012962, 12.652044, 0.883142, 1.004277e-19),
    (28, 6000, 10.0, "Run11b", -0.022888,  0.233650, 0.398710, 0.277577, 0.017927, 0.107615, 14.390327, 0.243295, 1.210849e-19),
    (28, 6000, 20.0, "Run11b", -0.022153,  0.287607, 0.545588, 0.385334, 0.024880, 0.158572, 14.753550, 0.149425, 1.142121e-19),
    (32, 6000,  0.0, "Run11b", -0.131510,  0.008640, 0.036884, 0.025825, 0.001068, 0.011559, 13.097060, 0.996169, 0.000000e+00),
    (32, 6000, 10.0, "Run11b", -0.022943,  0.164853, 0.395726, 0.275572, 0.015578, 0.115179, 14.977114, 0.936782, 0.000000e+00),
    (32, 6000, 20.0, "Run11b", -0.022130,  0.230603, 0.543139, 0.384319, 0.021683, 0.160145, 15.288557, 0.814176, 0.000000e+00),
    # Run12 — k={12,16,24}, P=8000, z={0,10}
    (12, 8000,  0.0, "Run12", -0.040362,  0.212520, 0.126478, 0.087179, 0.010944, 0.032692,  8.560354, 0.053640, 0.042940),
    (12, 8000, 10.0, "Run12", -0.023201,  0.200562, 0.376820, 0.260259, 0.036777, 0.091061,  9.450363, 0.000000, 0.068078),
    (16, 8000,  0.0, "Run12", -0.049475,  0.172739, 0.097288, 0.067281, 0.006244, 0.022361, 10.129745, 0.068008, 0.013498),
    (16, 8000, 10.0, "Run12", -0.022934,  0.273638, 0.369896, 0.256053, 0.027666, 0.095828, 11.277965, 0.000000, 0.023356),
    (24, 8000,  0.0, "Run12", -0.073184,  0.255498, 0.059026, 0.041117, 0.002402, 0.015892, 11.969841, 0.380268, 0.000161),
    (24, 8000, 10.0, "Run12", -0.022089,  0.328677, 0.360324, 0.249800, 0.018497, 0.101114, 13.630097, 0.026820, 0.000298),
    # Run13 — k={20,28}, P=8000, z={0,10}
    (20, 8000,  0.0, "Run13", -0.062689,  0.044308, 0.077712, 0.053978, 0.003890, 0.021497, 11.296383, 0.103448, 3.429828e-03),
    (20, 8000, 10.0, "Run13", -0.023129,  0.222971, 0.365007, 0.253381, 0.022174, 0.095427, 12.690983, 0.000958, 5.371326e-03),
    (28, 8000,  0.0, "Run13", -0.097559, -0.005863, 0.045647, 0.031880, 0.001546, 0.011913, 12.628636, 0.850575, 8.742610e-20),
    (28, 8000, 10.0, "Run13", -0.022291,  0.309833, 0.358857, 0.249520, 0.015868, 0.097324, 14.288102, 0.279693, 8.735904e-20),
    # Run14 — k={12,16,24}, P=8000, z={50,100}
    (12, 8000, 100.0, "Run14", -0.022389,  0.284311, 0.955788, 0.663906, 0.120878, 0.268483, 10.919813, 0.000000, 0.215149),
    (12, 8000,  50.0, "Run14", -0.022510,  0.276459, 0.724941, 0.512187, 0.078804, 0.184396, 10.217573, 0.000000, 0.125581),
    (16, 8000, 100.0, "Run14", -0.021737,  0.303832, 0.953069, 0.685967, 0.090897, 0.271378, 13.301521, 0.000000, 0.080438),
    (16, 8000,  50.0, "Run14", -0.022728,  0.195564, 0.721776, 0.516585, 0.059128, 0.187300, 12.266635, 0.000000, 0.041110),
    (24, 8000, 100.0, "Run14", -0.021540,  0.308345, 0.945758, 0.700001, 0.059796, 0.273874, 15.940382, 0.027778, 0.000967),
    (24, 8000,  50.0, "Run14", -0.022132,  0.243305, 0.713623, 0.514599, 0.039375, 0.194111, 14.651303, 0.023946, 0.000439),
]

new_df = pd.DataFrame(new_data, columns=cols)

# ── 3. Align columns and combine ──────────────────────────────────────────────
shared_cols = [c for c in log_df.columns if c in new_df.columns]
combined = pd.concat([log_df[shared_cols], new_df[shared_cols]], ignore_index=True)

# ── 4. Display ────────────────────────────────────────────────────────────────
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 160)
combined


# %%
combined.to_clipboard(index=False)

# %%

# %% [markdown]
# ### Run 16a — Plain IPCA (no RFF), top-50 universe
# k = {4, 8, 12, 16, 20, 24, 28} × z = {0, 10, 50, 100}, window = 24.  
# No P or seed dimension — one deterministic run per (k, z).  
# Baseline to compare geometry/performance against all RFF runs.

# %%
import time
from src.backtest.grassmannian_ipca import run_ipca_grassmannian_v2

# z=0 excluded: raw chars are correlated → (Z@W)ᵀ(Z@W) singular without ridge.
# max_zero=1.0: the 0.3 zero-fraction filter was designed for RFF features
# (continuous random values, so zeros are suspicious).  Raw characteristics
# legitimately have many imputed zeros, so we disable that filter here.
#
# "Price" is in char_cols (defined before df["Price"]=1.0 overwrites it) AND
# in df[["Price","ret"]], so df_plain would have a duplicate "Price" column.
# That makes unstack() return a DataFrame instead of a Series inside the
# backtest, silently poisoning all predictions with NaN.  Exclude it: as a
# constant (std=0 cross-sectionally) it contributes nothing after standardisation.
NUM_FACTORS_LIST_PLAIN = [4, 8, 12, 16, 20, 24, 28]
Z_VALUES_PLAIN         = [10.0, 50.0, 100.0]
WINDOW_LEN_PLAIN       = 24

char_feats_plain = [c for c in char_cols if c != "Price"]
print(f"char_feats_plain ({len(char_feats_plain)}): {char_feats_plain[:5]} ...")

# -- Impute characteristics (same recipe as all RFF runs) -------------------
X_chars_plain = df[char_cols].copy()
X_chars_plain = (
    X_chars_plain
    .groupby(level="date").transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars_plain.median())
    .fillna(0.0)
    .astype(np.float64)
)
df_plain = (
    pd.concat([df[["Price", "ret"]], X_chars_plain], axis=1)
    .dropna(subset=["ret"])
)
print(f"df_plain shape : {df_plain.shape}  ({len(char_feats_plain)} instruments)")
print(f"Universe       : {df_plain.index.get_level_values('permno').nunique()} stocks")

# -- Sequential sweep over (k, z) -------------------------------------------
plain_raw = {}
t0 = time.time()
for num_fact in NUM_FACTORS_LIST_PLAIN:
    for z in Z_VALUES_PLAIN:
        t1 = time.time()
        try:
            res = run_ipca_grassmannian_v2(
                df_trunc          = df_plain,
                char_feats        = char_feats_plain,
                num_fact          = num_fact,
                verbose           = False,
                window_len        = WINDOW_LEN_PLAIN,
                max_nan           = 0.3,
                max_zero          = 1.0,
                min_non_nan_frac  = 0.7,
                shrinkage         = z,
                min_gradient_norm = 1e-6,
            )
            elapsed = time.time() - t1
            print(f"  k={num_fact:2d}  z={z:6.0f}  r2_oos={res['r2_oos']:+.4f}  "
                  f"sharpe={res['sharpe']:+.3f}  ({elapsed:.1f}s)")
            plain_raw[(num_fact, _z_label(z))] = {
                "avg_r2_oos"              : res["r2_oos"],
                "avg_sharpe"              : res["sharpe"],
                "avg_subspace_stability"  : res["mean_subspace_stability"],
                "avg_max_principal_angle" : res["mean_max_principal_angle"],
                "avg_mean_principal_angle": res["mean_mean_principal_angle"],
                "avg_geodesic_accel"      : res["mean_geodesic_accel"],
                "avg_erank"               : res["mean_erank"],
                "avg_erank_collapse"      : res["erank_collapse_frac"],
                "avg_spectral_gap"        : res["mean_spectral_gap"],
            }
        except np.linalg.LinAlgError as e:
            elapsed = time.time() - t1
            print(f"  k={num_fact:2d}  z={z:6.0f}  SKIPPED ({e})  ({elapsed:.1f}s)")

total_time = time.time() - t0
print(f"\nRun 16a total: {total_time/60:.1f} min  ({total_time:.0f}s)")
if plain_raw:
    print(f"  → per (k,z) avg: {total_time/len(plain_raw):.1f}s")

# -- Summary DataFrame -------------------------------------------------------
ipca_plain_df = pd.DataFrame(plain_raw).T
ipca_plain_df.index = pd.MultiIndex.from_tuples(ipca_plain_df.index, names=["k", "z"])
ipca_plain_df = ipca_plain_df.sort_index()
display(ipca_plain_df)

# %% [markdown]
# ### Few other missings
# Priority 1 — Direct baseline match (5 jobs)
# k = {4, 8, 12, 16, 24},   z = 10,   P = 1024,   T = 24
#
# Priority 2 — Illusory regime at T=24 (3 jobs)
# k = {4, 12, 24},   z = 0,   P = 1024,   T = 24
#
# Priority 3 — VoC P-sweep at k below k₀ (3 jobs)
# k = 12,   z = 20,   P = {64, 256, 4096},   T = 24
#
# Priority 4 — Baseline z-match at k=24 high-P (4 jobs)
# k = 24,   z = {10, 50},   P = {1024, 4096},   T = 24
#

# %%
NUM_FACTORS_LIST = [4, 8, 12, 16, 24]
Z_VALUES         = [10]
N_FEATURES_RFF   = [1024]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%
NUM_FACTORS_LIST = [4,12, 24]
Z_VALUES         = [0]
N_FEATURES_RFF   = [1024]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ### ----------- ###

# %%
NUM_FACTORS_LIST = [12]
Z_VALUES         = [20]
N_FEATURES_RFF   = [64, 256, 4096]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%

# %%
NUM_FACTORS_LIST = [ 24]
Z_VALUES         = [10, 50]
N_FEATURES_RFF   = [1024, 4096]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%

# %% [markdown]
# ### V3 Report — New Run Combinations (SR_PA Metric Enabled)
#
# These cells fill the **13 missing (k, P, z) configurations** needed for the v3 comprehensive mapping report.
# All runs use `T = 24`, `γ = 0.25`, `NUM_ITER_RFF = 3`.
#
# The updated `backtest_GRASS_IPCA.py` now computes **SR_PA** (Sharpe ratio of principal angles) automatically.
#
# | Priority | k | P | z values | Rationale |
# |----------|---|---|----------|-----------|
# | 1 (CRITICAL) | 20 | 1024 | 0, 10, 20, 50 | Completely missing — needed to confirm k=20 optimal |
# | 2 | 20 | 4096 | 0, 10, 20 | Fill VoC P-curve at high P for k=20 |
# | 3 | 20 | 256 | 10, 20 | Start of P-curve for k=20 |
# | 4 | 20 | 64 | 10, 20 | Baseline for P-curve for k=20 |
# | 5 | 12 | 10000 | 10, 50 | Confirm saturation at very high P |

# %% [markdown]
# ### Extra simulations - Priority 1 (CRITICAL) — k=20, P=1024, z ∈ {0, 10, 20, 50}
# k=20 at P=1024 is **completely missing** from existing data. These 4 runs are the most important
# to confirm that the optimal operating point is k ∈ [20, 24] rather than k ∈ [12, 16].

# %%
NUM_FACTORS_LIST = [20]
Z_VALUES         = [0, 10, 20, 50]
N_FEATURES_RFF   = [1024]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ### Priority 2 — k=20, P=4096, z ∈ {0, 10, 20}
# Fill the VoC P-curve at high P for k=20. Existing data only has k=20 at P=4096 with z=100.
# Need z ∈ {0, 10, 20} to trace the ridge shrinkage → productive illusion transition.

# %%
NUM_FACTORS_LIST = [20]
Z_VALUES         = [0, 10, 20]
N_FEATURES_RFF   = [4096]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ### Priority 3 — k=20, P=256, z ∈ {10, 20}
# Lower end of the P-curve for k=20. Establishes whether VoC starts kicking in at moderate P
# when the latent dimension is 20 (above the spectral gap cliff at k₀ ≈ 12–16).

# %%
NUM_FACTORS_LIST = [20]
Z_VALUES         = [10, 20]
N_FEATURES_RFF   = [256]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ### Priority 4 — k=20, P=64, z ∈ {10, 20}
# Baseline for the k=20 P-curve. At P=64, the RFF dimension barely exceeds k=20, so we expect
# poor performance (under-parameterised regime). This anchors the left end of the VoC curve.

# %%
NUM_FACTORS_LIST = [20]
Z_VALUES         = [10, 20]
N_FEATURES_RFF   = [64]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ### Priority 5 — k=12, P=10000, z ∈ {10, 50}
# Confirm VoC saturation at very high P for k=12 (below spectral gap cliff).
# If the Sharpe plateaus or declines at P=10000 relative to P=8000, it validates
# the implicit-shrinkage saturation predicted by KMZ Theorem 2.

# %%
NUM_FACTORS_LIST = [12]
Z_VALUES         = [10, 50]
N_FEATURES_RFF   = [10000]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)
prio
raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ## Post-Run — Append New Results to all_rff_results.csv
# After running all cells above, execute the cell below to append the new results into the master CSV.
# This also adds the new SR_PA columns (sr_mean_principal_angle, sr_max_principal_angle).

# %%
import pandas as pd
import os

# ---------- load existing CSV ----------
csv_path = os.path.join("..", "data", "all_rff_results.csv")
existing_df = pd.read_csv(csv_path)
print(f"Existing rows: {len(existing_df)}")

# ---------- collect new run results ----------
# results_rff_df from the LAST run_sweep is still in memory,
# but we want ALL new runs.  Re-scan the results_by_factor_rff dicts
# that were created in the cells above.
# A safer approach: read the cache parquet files written by the backtest.

cache_dir = os.path.join("..", "cache")
new_rows = []
for fname in sorted(os.listdir(cache_dir)):
    if not fname.endswith(".parquet"):
        continue
    fpath = os.path.join(cache_dir, fname)
    try:
        tmp = pd.read_parquet(fpath)
        # Extract (k, P, z) from the filename or the data
        for col in ["k", "num_factors", "P", "n_features_rff", "z", "z_value"]:
            if col in tmp.columns:
                pass  # available
        new_rows.append(tmp)
    except Exception:
        pass

# Alternative: just display what we have and manually append after review
print("\\n--- After running the new cells, re-run the master aggregation ---")
print("--- from the 'Aggregate All Results' section above to refresh   ---")
print(f"--- the CSV.  Current CSV has {len(existing_df)} rows.           ---")
print(f"--- Expected after new runs: {len(existing_df) + 13} rows.      ---")
display(existing_df.tail(5))

# %%
# !python src/generate_extended_mapping_report.py --help 2>/dev/null || cat src/generate_extended_mapping_report.py

# %%
#read all_rff_results.csv
import os
import pandas as pd
csv_path = os.path.join("..", "data", "all_rff_results.csv")
df_results = pd.read_csv(csv_path)
print(f"Loaded {len(df_results)} rows from {csv_path}")
display(df_results.tail(5))

# %%
df_results[df_results["k"] == 24]

# %%
# missings - k=12,z = 10, P = 64, 256
# k=24, z=10, P=64, 256is missings


NUM_FACTORS_LIST = [12, 24]
Z_VALUES         = [10]
N_FEATURES_RFF   = [64, 256]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%
import os
import pandas as pd

csv_path = os.path.join("..", "data", "all_rff_results.csv")

# Hardcoded results from cell 135 output (k=12,24  P=64,256  z=10)
new_rows = [
    {"k": 12, "P": 64,  "z": 10.0, "run_label": "Run_fill", "avg_r2_oos": -0.021889, "avg_sharpe": -0.131188, "avg_subspace_stability": 1.339935, "avg_max_principal_angle": 1.070668, "avg_mean_principal_angle": 0.173496, "avg_geodesic_accel": 0.264186, "avg_erank": 10.716564, "avg_erank_collapse": 0.0, "avg_spectral_gap": 0.233379},
    {"k": 12, "P": 256, "z": 10.0, "run_label": "Run_fill", "avg_r2_oos": -0.042810, "avg_sharpe":  0.028664, "avg_subspace_stability": 1.632055, "avg_max_principal_angle": 1.169825, "avg_mean_principal_angle": 0.279285, "avg_geodesic_accel": 0.361126, "avg_erank": 10.627672, "avg_erank_collapse": 0.0, "avg_spectral_gap": 0.225665},
    {"k": 24, "P": 64,  "z": 10.0, "run_label": "Run_fill", "avg_r2_oos": -0.022888, "avg_sharpe": -0.088186, "avg_subspace_stability": 1.397245, "avg_max_principal_angle": 1.446847, "avg_mean_principal_angle": 0.061644, "avg_geodesic_accel": 0.022397, "avg_erank": 17.709649, "avg_erank_collapse": 0.0, "avg_spectral_gap": 0.041903},
    {"k": 24, "P": 256, "z": 10.0, "run_label": "Run_fill", "avg_r2_oos": -0.057331, "avg_sharpe":  0.187607, "avg_subspace_stability": 1.411048, "avg_max_principal_angle": 1.517072, "avg_mean_principal_angle": 0.063955, "avg_geodesic_accel": 0.004029, "avg_erank": 20.301237, "avg_erank_collapse": 0.0, "avg_spectral_gap": 0.124279},
]

new_df = pd.DataFrame(new_rows)
print(f"Appending {len(new_df)} new rows to {csv_path}")
new_df.to_csv(csv_path, mode='a', header=False, index=False)
print("Done.")
display(new_df)

# %%
# for k = 12, 24, lets plot Sharpe vs P, for z = 10 and P in {64, 256, 1024, 4096}
import matplotlib.pyplot as plt
import seaborn as sns

full_df = pd.read_csv(csv_path)
plot_df = full_df[
    (full_df["k"].isin([12, 24])) &
    (full_df["z"] == 10) &
    (full_df["P"].isin([64, 256, 1024, 4096]))
]
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df, x="P", y="avg_sharpe", hue="k", marker="o")
plt.xscale("log")
plt.title("Sharpe Ratio vs P for k=12 and k=24 (z=10)")
plt.xlabel("P (log scale)")
plt.ylabel("Average Sharpe Ratio")
plt.legend(title="k")
plt.grid(True)
plt.show()

# %% [markdown]
# ### IMPORTANT NOTE : Z = 10, could be the z where we get to the permanent ascent in sharpe stage

# %% [markdown]
# ## making sharpe plots smoother

# %%
# lets make the z = 10 run much smoother with more data points ie, P = 32, 128, 512, 2048, 6048, 8192


NUM_FACTORS_LIST = [12, 24]
Z_VALUES         = [10]
N_FEATURES_RFF   = [32, 128, 512,]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%
results_rff_df

# %%
## Add these to the existing csv as well, and 

# for k = 12, 24, lets plot Sharpe vs P, for z = 10 and P in {64, 256, 1024, 4096}
import matplotlib.pyplot as plt
import seaborn as sns
csv_path = os.path.join("..", "data", "all_rff_results.csv")
full_df = pd.read_csv(csv_path)
plot_df = full_df[
    (full_df["k"].isin([12, 24])) &
    (full_df["z"] == 10) &
    (full_df["P"].isin([64, 256, 1024, 4096]))
]

# add the results fromresults_rff_df to plot_df
# Diagnose
print("plot_df cols:", list(plot_df.columns))
print("results_rff_df cols:", list(results_rff_df.columns))
print("results_rff_df index names:", results_rff_df.index.names)

# Fix: convert MultiIndex -> columns before concat
new_points = results_rff_df.reset_index().copy()

# z is like 'z=1e+01' in results_rff_df; convert to numeric to match plot_df
new_points["z"] = new_points["z"].str.replace("z=", "", regex=False).astype(float)

# add missing metadata column expected by plot_df
new_points["run_label"] = "Run_smooth"

# align column order and concatenate
plot_df_updated = pd.concat([plot_df, new_points[plot_df.columns]], ignore_index=True)

display(plot_df_updated)



# %%
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df_updated, x="P", y="avg_sharpe", hue="k", marker="o")
plt.xscale("log")
plt.title("Sharpe Ratio vs P for k=12 and k=24 (z=10)")
plt.xlabel("P (log scale)")
plt.ylabel("Average Sharpe Ratio")
plt.legend(title="k")
plt.grid(True)
plt.show()

# %% [markdown]
# ### Diagonstics : Drops at 64 for some random reason, worth looking into this

# %%

NUM_FACTORS_LIST = [12, 24]
Z_VALUES         = [10]
N_FEATURES_RFF   = [2048, 6048, 8192]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float64)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)

raw_results           = run_sweep(jobs)
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%
# append thes to plot_df_updated and plot everything again
new_points = results_rff_df.reset_index().copy()
new_points["z"] = new_points["z"].str.replace("z=", "", regex=False).astype(float)
new_points["run_label"] = "Run_smooth"
plot_df_updated = pd.concat([plot_df_updated, new_points[plot_df_updated.columns]], ignore_index=True)
display(plot_df_updated)
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df_updated, x="P", y="avg_sharpe", hue="k", marker="o")
plt.xscale("log")
plt.title("Sharpe Ratio vs P for k=12 and k=24 (z=10)")
plt.xlabel("P (log scale)")
plt.ylabel("Average Sharpe Ratio")
plt.legend(title="k")
plt.grid(True)
plt.show()

# %%
# Add results from plot_df_updated to all_rff_resulst.csv, ensure not to add duplicates
csv_path = os.path.join("..", "data", "all_rff_results.csv")
existing_df = pd.read_csv(csv_path)
# Create a unique identifier for each row to avoid duplicates (e.g., based on k, P, z)
existing_df["id"] = existing_df.apply(lambda row: f"{row['k']}_{row['P']}_{row['z']}", axis=1)
plot_df_updated["id"] = plot_df_updated.apply(lambda row: f"{row['k']}_{row['P']}_{row['z']}", axis=1)
# Filter out rows that are already in
new_rows_df = plot_df_updated[~plot_df_updated["id"].isin(existing_df["id"])]
print(f"Appending {len(new_rows_df)} new rows to {csv_path}")
new_rows_df.drop(columns=["id"], inplace=True)  # Remove the helper column before saving
new_rows_df.to_csv(csv_path, mode='a', header=False, index=False)
print("Done.")


# %%
# Lets make the plots saturating, 


NUM_FACTORS_LIST = [12, 24]
Z_VALUES         = [10]
N_FEATURES_RFF   = [7000]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float32)        # float32 to halve memory (RFF sin/cos values don't need float64 precision)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
import gc

rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)
del rff_inputs; gc.collect()  # free DataFrames — payloads are already serialised in jobs

raw_results           = run_sweep(jobs)
del jobs; gc.collect()
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %% [markdown]
# ## P = 48 case for smoother plots

# %%
# Lets make the plots saturating, 


NUM_FACTORS_LIST = [12, 24]
Z_VALUES         = [10]
N_FEATURES_RFF   = [48]
NUM_ITER_RFF     = 3
WINDOW_LEN       = 24
GAMMA            = 0.25

date_level = "date"
X_chars = df[char_cols].copy()
X_chars = (
    X_chars
    .groupby(level=date_level).transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars.median())
    .fillna(0.0)
    .astype(np.float32)        # float32 to halve memory (RFF sin/cos values don't need float64 precision)
)
print(f"X_chars shape : {X_chars.shape}  (obs × chars)")
print(f"X_chars dtype : {X_chars.dtypes.unique()}")

# %%
import gc

rff_inputs = build_rff_inputs(
    X_chars               = X_chars,
    df_base               = df,
    n_features_rff        = N_FEATURES_RFF,
    num_iter_rff          = NUM_ITER_RFF,
    gamma                 = GAMMA,
    RandomFourierFeatures = RandomFourierFeatures,
)
print(f"  {len(rff_inputs)} RFF datasets ready.")

jobs = build_jobs(
    rff_inputs        = rff_inputs,
    num_factors_list  = NUM_FACTORS_LIST,
    n_features_rff    = N_FEATURES_RFF,
    z_values          = Z_VALUES,
    num_iter_rff      = NUM_ITER_RFF,
    window_len        = WINDOW_LEN,
    min_gradient_norm = 1e-4,
)
del rff_inputs; gc.collect()  # free DataFrames — payloads are already serialised in jobs

raw_results           = run_sweep(jobs)
del jobs; gc.collect()
results_by_factor_rff = aggregate(raw_results)
results_rff_df        = make_results_df(results_by_factor_rff)
display(results_rff_df)

# %%

# %%
# R² plot using the comprehensive dataframe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataframe from the provided data
data_dict = {
    'k': [12, 12, 24, 24, 12, 12, 24, 24, 12, 12, 12, 24, 24, 24, 12, 12, 12, 24, 24, 24],
    'P': [1024, 4096, 1024, 4096, 64, 256, 64, 256, 32, 128, 512, 32, 128, 512, 2048, 6048, 8192, 2048, 6048, 8192],
    'z': [10.0]*20,
    'avg_r2_oos': [-0.025357, -0.022933, -0.023974, -0.022335, -0.021889, -0.042810, -0.022888, -0.057331, 
                    -0.037517, -0.038371, -0.026474, -0.039079, -0.033627, -0.024649, -0.024129, -0.022795, -0.023028,
                    -0.023203, -0.022297, -0.022467],
    'avg_sharpe': [0.129996, 0.294711, 0.194859, 0.314774, -0.131188, 0.028664, -0.088186, 0.187607,
                   0.065761, 0.058853, 0.174036, 0.032611, 0.111300, 0.214099, 0.173702, 0.275588, 0.342160,
                   0.209750, 0.269939, 0.263948]
}

df_plot = pd.DataFrame(data_dict)

# Create R² plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_plot, x="P", y="avg_r2_oos", hue="k", marker="o")
plt.xscale("log")
plt.title("OOS R² vs P for k=12 and k=24 (z=10)")
plt.xlabel("P (log scale)")
plt.ylabel("Average OOS R²")
plt.legend(title="k")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ── Appendix: Full list of 73 characteristics used in Plain IPCA experiments ──
# Derived from char_cols (defined in cell 11) minus "Price"

from pathlib import Path

chars = [c for c in char_cols if c != "Price"]   # 73 names

# ── Print as plain list ──────────────────────────────────────────────────────
print(f"Total characteristics: {len(chars)}")
for i, c in enumerate(chars, 1):
    print(f"  {i:3d}. {c}")

# ── Generate LaTeX table (three-column) ──────────────────────────────────────
n = len(chars)
n_rows = (n + 2) // 3
cols_split = [chars[i::3] for i in range(3)]

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"  \centering")
lines.append(r"  \small")
lines.append(f"  \\caption{{All {n} raw characteristics used as IPCA instruments (JKP/GHZ factor zoo, Open Asset Pricing).}}")
lines.append(r"  \label{tab:characteristics}")
lines.append(r"  \begin{tabular}{@{} r l | r l | r l @{}}")
lines.append(r"    \toprule")
lines.append(r"    \# & Characteristic & \# & Characteristic & \# & Characteristic \\")
lines.append(r"    \midrule")

for row_i in range(n_rows):
    parts = []
    for col_j in range(3):
        idx = row_i + col_j * n_rows
        if idx < n:
            parts.append(f"    {idx+1} & \\texttt{{{chars[idx]}}}")
        else:
            parts.append("      &          ")
    lines.append(" & ".join(parts) + r" \\")

lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}")
lines.append(r"\end{table}")

latex_table = "\n".join(lines)
print()
print("=" * 70)
print("LaTeX table:")
print("=" * 70)
print(latex_table)

# ── Save to docs/ ────────────────────────────────────────────────────────────
out_path = Path("..") / "docs" / "characteristics_table.tex"
out_path.write_text(latex_table + "\n")
print(f"\nSaved → {out_path.resolve()}")


# %%
