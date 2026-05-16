"""
One-job smoke test of the synthetic sweep pipeline.

Runs a single (k=12, P=1024, z=10) job through the same code path as
`rerun_synthetic_v3_checkpointed.py` to verify the full chain:
    generate_synthetic_panel -> RFF features -> IPCA-Grassmannian backtest -> aggregation.

Does NOT touch all_synthetic_datarun_results.csv. Writes a one-row summary to
$GOCL_ARTIFACTS_DIR/smoke/tier_a/smoke_test_output.csv (default:
<project_root>/artifacts/smoke/tier_a/smoke_test_output.csv) for inspection.

Usage (from tier_a_analytical/):
    python tests/smoke_test_one_job.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Force single-threaded BLAS for repeatable timing (mirrors the sweep script).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/X.py -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate.synthetic_panel import generate_synthetic_panel
from src.models.rff import RandomFourierFeatures
from src.sweeps.grassmannian_worker import (
    aggregate,
    build_jobs,
    build_rff_inputs,
    make_results_df,
    run_sweep_checkpointed,
)

# Verbatim from rerun_synthetic_v3_checkpointed.py:
TRUE_K0 = 12
GAMMA = 0.25
WINDOW_LEN = 24
NUM_ITER_RFF = 1


def main() -> None:
    print("=== Smoke test: 1 job through the full synthetic sweep chain ===\n")

    # Step 1: generate the synthetic panel.
    t0 = time.time()
    df_synth, truth, char_cols = generate_synthetic_panel(
        T=360, N=50, m=30, k0=TRUE_K0, seed=42,
        snr_base=5.0, eigenvalue_decay=0.82, sigma_eps=1.0,
        factor_ar1=0.5, factor_vol=0.3, w_drift_scale=None,
        target_dproj=0.6, z_corr=0.3, hetero_strength=0.3,
    )
    x_chars = df_synth[char_cols].apply(lambda s: s.fillna(s.mean()), axis=0).astype(np.float64)
    print(f"[1/4] synthetic panel        {time.time()-t0:6.2f}s  shape={df_synth.shape}, k0={truth['true_k0']}")

    # Step 2: RFF feature lift.
    t0 = time.time()
    P_list = [1024]
    rff_inputs = build_rff_inputs(
        x_chars, df_synth, P_list, NUM_ITER_RFF, GAMMA, RandomFourierFeatures,
    )
    print(f"[2/4] RFF feature lift       {time.time()-t0:6.2f}s  P={P_list}")

    # Step 3: build the ONE job.
    K_list = [12]
    Z_list = [10]
    jobs = build_jobs(rff_inputs, K_list, P_list, Z_list, NUM_ITER_RFF, WINDOW_LEN)
    print(f"[3/4] built {len(jobs)} job(s)        k={K_list}, P={P_list}, z={Z_list}")

    # Step 4: run the job (checkpointed worker), aggregate, summarize.
    t0 = time.time()
    cache_root = Path(os.environ.get("GOCL_CACHE_DIR", PROJECT_ROOT / "cache"))
    checkpoint_dir = cache_root / "smoke_test_checkpoints"
    raw = run_sweep_checkpointed(jobs, checkpoint_dir, n_workers=1, verbose=True, force=True)
    print(f"[4/4] sweep + checkpoint     {time.time()-t0:6.2f}s")

    agg = aggregate(raw)
    df = make_results_df(agg)
    print("\n=== Result ===")
    print(df.to_string())

    artifacts_root = Path(os.environ.get("GOCL_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
    out_path = artifacts_root / "smoke" / "tier_a" / "smoke_test_output.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(out_path, index=False)
    print(f"\nSaved one-row summary -> {out_path}")
    print("Smoke test PASSED — full Tier A sweep chain works end-to-end.")


if __name__ == "__main__":
    main()
