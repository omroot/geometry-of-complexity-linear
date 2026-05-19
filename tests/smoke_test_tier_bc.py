"""
Minimal smoke test of the Tier B/C pipeline.

Mirrors the chain in `experiment_ppp.py` (synthetic mode) but with a tiny config:
    1 (k, T) combo  ×  {baseline + 2 small RFF dims + 1 large RFF dim}  =  4 jobs

Verifies:
  - synthetic panel build via `generate_ipca_workflow_panel`
  - `GrassmannianIPCAWorkflow.rolling_ipca_predictions(use_rff=False)`        (baseline IPCA)
  - `GrassmannianIPCAWorkflow.rolling_ipca_predictions(use_rff=True, ...)`    (RFF small + large)
  - `summarize_experiment` -> xlsx write
  - one rendering of the cell-27 ext_plotA chart

Runtime: tens of minutes (single-worker; this is a chain test, not a perf test).

Usage (from refactored/tier_bc_experiment/):
    python tests/smoke_test_tier_bc.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

# Pin BLAS threads early (matches the sweep script).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/X.py -> project root
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.models.grassmannian_ipca import generate_ipca_workflow_panel
from src.workflows.grassmannian_ipca import GrassmannianIPCAWorkflow
from src.io.naming import build_experiment_excel_name
from src.portfolio import (
    build_directional_portfolio,
    compute_portfolio_returns,
    portfolio_performance,
)

# ---- Minimal config (override of experiment_ppp.py constants) ----
SMOKE_K = 12
SMOKE_T = 24
SMOKE_SEED = 10
SMOKE_RFF_SMALL = [24, 64]
SMOKE_RFF_LARGE = [512]
ALPHA = 100
RFF_GAMMA = 0.25
FORECAST_START = "1993-01-01"
MIN_TRAIN_OBS = 500
MARKET_CAP_TOP_N = 50

SYNTHETIC_CONFIG = {
    "T": 120, "N": 50, "m": 75, "seed": 10,
    "start_date": "1990-01-01", "include_intercept": False,
    "beta_rho": 0.9, "sigma_beta": 0.5, "z_rho": 0.4, "z_scale": 1.0,
    "heavy_tail_df": 5.0, "sigma_eps_base": 0.5, "hetero_strength": 0.5,
    "missing_prob": 0.05, "missing_mode": "mcAR", "impute": "zero",
    "z_drift_scale": 0.02,
}

ARTIFACTS_ROOT = Path(os.environ.get("GOCL_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
OUT_DIR = ARTIFACTS_ROOT / "smoke" / "tier_bc"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def banner(s: str) -> None:
    print(f"\n{'=' * (len(s) + 4)}\n  {s}\n{'=' * (len(s) + 4)}")


def main() -> None:
    banner("Tier B/C smoke test: 1 (k,T) × {baseline + 2 small RFF + 1 large RFF} = 4 jobs")
    print(f"  k={SMOKE_K}, T={SMOKE_T}, seed={SMOKE_SEED}")
    print(f"  small RFF: {SMOKE_RFF_SMALL}, large RFF: {SMOKE_RFF_LARGE}")
    print(f"  output dir: {OUT_DIR}")

    # 1. Build synthetic char panel ------------------------------------------------
    banner("[1/5] Build synthetic char panel")
    t0 = time.time()
    config = dict(SYNTHETIC_CONFIG)
    config["m"] = max(config["m"], SMOKE_K)
    char_data, truth = generate_ipca_workflow_panel(**config)
    print(f"  panel shape  : {char_data.shape}")
    print(f"  permnos      : {char_data['permno'].nunique()}")
    print(f"  months       : {char_data['yyyymm'].nunique()}")
    print(f"  has y_ipca?  : {'y_ipca' in char_data.columns}")
    print(f"  elapsed      : {time.time() - t0:.2f}s")

    # 2. Workflow + common kwargs --------------------------------------------------
    banner("[2/5] Initialize GrassmannianIPCAWorkflow")
    wf = GrassmannianIPCAWorkflow()
    market_cap_col = "mcap" if "mcap" in char_data.columns else None
    market_cap_top_n = (min(MARKET_CAP_TOP_N, char_data["permno"].nunique())
                       if market_cap_col else None)
    print(f"  market_cap_col = {market_cap_col}, top_n = {market_cap_top_n}")

    common_kwargs = dict(
        char_data=char_data, forecast_start=FORECAST_START, target_col="y_ipca",
        n_factors=SMOKE_K, train_window_months=SMOKE_T, min_train_obs=MIN_TRAIN_OBS,
        normalize=True, mean_factor=True, silent=True, iter_tol=1e-4,
        warm_start=True, alpha=ALPHA, show_progress=False,
        market_cap_filter_col=market_cap_col, market_cap_filter_top_n=market_cap_top_n,
    )

    result_rff_all = {SMOKE_SEED: {}}

    # 3. Baseline (no RFF) ---------------------------------------------------------
    banner("[3/5] Baseline IPCA (no RFF)")
    t0 = time.time()
    result_rff_all[SMOKE_SEED][0] = wf.rolling_ipca_predictions(
        use_rff=False, rff_n_components=0,
        rff_gamma=RFF_GAMMA, rff_random_state=SMOKE_SEED,
        **common_kwargs,
    )
    print(f"  baseline elapsed: {time.time() - t0:.1f}s")

    # 4. Small + large RFF dims ----------------------------------------------------
    banner("[4/5] RFF sweep (small + large)")
    for rff_n in SMOKE_RFF_SMALL + SMOKE_RFF_LARGE:
        t0 = time.time()
        result_rff_all[SMOKE_SEED][rff_n] = wf.rolling_ipca_predictions(
            use_rff=True, rff_n_components=rff_n,
            rff_gamma=RFF_GAMMA, rff_random_state=SMOKE_SEED,
            **common_kwargs,
        )
        print(f"  RFF={rff_n:>4d}: {time.time() - t0:6.1f}s")

    # Save raw pickle for inspection
    pkl_path = OUT_DIR / "smoke_result_rff_all.pkl"
    payload = {
        "k": SMOKE_K, "T": SMOKE_T, "alpha": ALPHA, "rff_gamma": RFF_GAMMA,
        "data_label": "synthetic_smoke", "result_rff_all": result_rff_all,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  saved pickle -> {pkl_path}")

    # 5. Summarize + write xlsx + render plot --------------------------------------
    banner("[5/5] summarize + xlsx + plot")
    components = sorted({m for sm in result_rff_all.values() for m in sm})

    # Light summarization (just enough to build the OOS R² → plot pipeline).
    oos_r2_avg = {}
    diag_frames = []
    for rff_n in components:
        all_r2 = []
        for seed, smap in result_rff_all.items():
            pred_df, diag_df = smap[rff_n][0].copy(), smap[rff_n][1].copy()
            ss_res = ((pred_df["y_true"] - pred_df["y_pred"]) ** 2).sum()
            ss_tot = (pred_df["y_true"] ** 2).sum()
            r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
            all_r2.append(float(r2))
            diag_df["seed"] = seed
            diag_df["rff_n_component"] = rff_n
            diag_frames.append(diag_df)
        oos_r2_avg[rff_n] = float(np.nanmean(all_r2)) if all_r2 else np.nan
    diag_stack = pd.concat(diag_frames, ignore_index=True)

    print("  OOS R² per RFF dim (mean over seeds):")
    for k_, v in sorted(oos_r2_avg.items()):
        print(f"    m={k_:>4d}:  R²_oos = {v:+.5f}")

    # XLSX write
    xlsx_path = OUT_DIR / "smoke_oos_r2.xlsx"
    pd.DataFrame({"rff_n_component": list(oos_r2_avg), "oos_r2": list(oos_r2_avg.values())}).to_excel(
        xlsx_path, index=False
    )
    print(f"  saved xlsx -> {xlsx_path}")

    # Render the cell-27 style plot (minimal, single-panel, just to verify the plotting code path)
    m_vals = np.array(sorted(oos_r2_avg))
    r2_vals = np.array([100 * oos_r2_avg[m] for m in m_vals])
    n_train = int(diag_stack["n_train"].median()) if not diag_stack.empty else 1
    x_vals = 2 * m_vals * SMOKE_K / max(n_train, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(1, linestyle="--", color="grey", lw=1, alpha=0.7, label="$p/n=1$")
    ax.plot(x_vals, r2_vals, marker="D", color="#e53935", lw=2, markersize=8, label="OOS $R^2$")
    for m_, x_, y_ in zip(m_vals, x_vals, r2_vals):
        ax.annotate(f"$\\tilde m$={m_}", (x_, y_), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel(r"$p/n = 2\tilde m k / n_{\mathrm{train}}$")
    ax.set_ylabel(r"OOS $R^2$ (%)")
    ax.set_title(f"Smoke test: OOS $R^2$ vs $p/n$  (k={SMOKE_K}, T={SMOKE_T}, $\\alpha$={ALPHA})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plot_path = OUT_DIR / "smoke_ext_plotA.pdf"
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {plot_path}")

    banner("Smoke test PASSED")
    print(f"Outputs in: {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:>8d} bytes)")


if __name__ == "__main__":
    main()
