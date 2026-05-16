"""
Smoke test for the real-data Tier B/C pipeline.

Loads the LinearCase parquet (via the symlink at data/char_data.parquet), applies
the Size->mcap alias, and runs ONE minimal (k=12, T=24) backtest at baseline
(no RFF). Verifies:
  - parquet load works
  - alias logic triggers
  - MARKET_CAP_TOP_N=50 filter reduces panel to 50 stocks/month
  - GrassmannianIPCAWorkflow accepts the real panel
  - 1 baseline rolling-IPCA backtest completes end-to-end

Times the single baseline run so we can extrapolate the full real-data run cost.

Usage (from refactored/tier_bc_experiment/):
    python tests/smoke_test_tier_bc_real.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/X.py -> project root
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.workflows.grassmannian_ipca import GrassmannianIPCAWorkflow

SMOKE_K = 12
SMOKE_T = 24
SMOKE_SEED = 10
ALPHA = 100
RFF_GAMMA = 0.25
FORECAST_START = "1993-01-01"
MIN_TRAIN_OBS = 500
MARKET_CAP_TOP_N = 50
CHAR_DATA_PATH = PROJECT_ROOT / "data" / "char_data.parquet"


def banner(s: str) -> None:
    print(f"\n{'=' * (len(s) + 4)}\n  {s}\n{'=' * (len(s) + 4)}")


def main() -> None:
    banner("Real-data Tier B/C smoke test: 1 baseline IPCA on full LinearCase panel")

    # 1. Load parquet
    t0 = time.time()
    if not CHAR_DATA_PATH.exists():
        raise FileNotFoundError(f"Symlink missing or broken: {CHAR_DATA_PATH}")
    char_data = pd.read_parquet(CHAR_DATA_PATH)
    print(f"[1/4] loaded parquet         {time.time() - t0:.2f}s  shape={char_data.shape}")
    print(f"       columns (last 5): {list(char_data.columns)[-5:]}")
    print(f"       time range: {char_data['yyyymm'].min()} .. {char_data['yyyymm'].max()}")
    print(f"       permnos: {char_data['permno'].nunique():,}")

    # 2. Apply Size->mcap alias + cleaning (drop >70% NaN cols, impute the rest)
    if "mcap" not in char_data.columns and "Size" in char_data.columns:
        char_data["mcap"] = char_data["Size"]
        print(f"[2/4] aliased Size -> mcap     non-null mcap rows: {char_data['mcap'].notna().sum():,}")
    from src.ingest.pipeline import DataPipeline
    pipe = DataPipeline()
    pre_cols = char_data.shape[1]
    # Full cleaning recipe (matches the fresh-download path in experiment_ppp.py)
    char_data, dropped_nan = pipe.remove_mostly_nan_columns(char_data, max_nan_frac=0.6)
    print(f"       dropped {len(dropped_nan)} chars (>60% NaN): {dropped_nan}")
    char_data, low_std, high_corr = pipe.drop_low_std_and_high_corr(
        char_data, min_std=1e-6, max_corr=0.98,
    )
    print(f"       dropped {len(low_std)} low-std + {len(high_corr)} high-corr chars")
    char_data = pipe.fill_remaining_missing(char_data, use_past_only=True)
    # Drop any remaining rows still NaN (e.g., earliest months pre-history)
    char_cols = [c for c in char_data.columns
                 if c not in ('permno', 'yyyymm', 'mcap', 'Size', 'excess_ret', 'date')]
    pre_rows = len(char_data)
    char_data = char_data.dropna(subset=char_cols + ['excess_ret'])
    print(f"       dropped {pre_rows - len(char_data):,} rows with residual NaNs")
    print(f"       final shape: {char_data.shape}  (was cols={pre_cols})")
    nan_any = char_data[char_cols].isna().any(axis=1).sum()
    print(f"       rows with ANY null char now: {nan_any:,} / {len(char_data):,}")

    # 3. Confirm market-cap filter logic activates
    market_cap_col = "mcap" if "mcap" in char_data.columns else None
    market_cap_top_n = (min(MARKET_CAP_TOP_N, char_data["permno"].nunique())
                       if market_cap_col else None)
    print(f"[3/4] filter wiring          market_cap_col={market_cap_col}, top_n={market_cap_top_n}")
    if market_cap_col is None:
        raise RuntimeError("Filter would NOT trigger; aborting before slow run.")

    # 4. Run ONE baseline rolling-IPCA backtest (no RFF)
    banner("[4/4] baseline rolling-IPCA on real panel (1 backtest, ~10–60 min)")
    wf = GrassmannianIPCAWorkflow()
    t0 = time.time()
    result = wf.rolling_ipca_predictions(
        char_data=char_data, forecast_start=FORECAST_START, target_col="excess_ret",
        n_factors=SMOKE_K, train_window_months=SMOKE_T, min_train_obs=MIN_TRAIN_OBS,
        normalize=True, mean_factor=True, silent=True, iter_tol=1e-4,
        warm_start=True, alpha=ALPHA, show_progress=True,
        market_cap_filter_col=market_cap_col, market_cap_filter_top_n=market_cap_top_n,
        use_rff=False, rff_n_components=0, rff_gamma=RFF_GAMMA, rff_random_state=SMOKE_SEED,
    )
    dt = time.time() - t0

    pred_df, diag_df = result[0], result[1]
    ss_res = ((pred_df["y_true"] - pred_df["y_pred"]) ** 2).sum()
    ss_tot = (pred_df["y_true"] ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    banner(f"Real-data smoke test PASSED  ({dt:.1f}s for 1 baseline backtest)")
    print(f"  predictions rows: {len(pred_df):,}")
    print(f"  diagnostic rows : {len(diag_df):,}")
    print(f"  OOS R² (baseline, no RFF): {r2:+.5f}")
    if not diag_df.empty and "n_train" in diag_df.columns and diag_df["n_train"].notna().any():
        print(f"  n_train median: {int(diag_df['n_train'].median())}")
        print(f"  rolling windows: {diag_df['train_end'].nunique()}")
    else:
        print(f"  (diag empty — all windows skipped; likely MIN_TRAIN_OBS not met)")
    print()
    print(f"=== Extrapolation ===")
    print(f"  This was 1 of 85 jobs in 1 of 5 configs.")
    print(f"  Per-config (avg, naively): {dt * 85 / 60:.1f} min,  but P=4096 dominates 70% of cost.")
    print(f"  Full real-data run (rough): ~{dt * 425 * 8 / 3600:.1f} h (with P=4096 ~10x baseline weighting)")


if __name__ == "__main__":
    main()
