#!/usr/bin/env python3
"""Paper §9.1-§9.9 baseline: 88-config grid on the top-50 fixed universe.

Drives Figures 12, 14, 15, 16 and Tables 3, 4. Figure 13 (ridge sweep) has
its own finer-z slice in 03_ridge_sweep.py.

Universe: top-50 stocks by **mean** market cap (using the LinearCase
``Size`` proxy) over the full sample -- a fixed-universe choice that
introduces a mild look-ahead but keeps the Grassmannian optimization
problem tractable, matching the paper's baseline.

Outputs:
    results/data/real_baseline_grid.csv -- per (k, P, z) row.

Usage:
    python scripts/experiments/real_data/02_baseline_grid.py
    CHAR_DATA_PARQUET=/path/to/char_data.parquet python scripts/experiments/real_data/02_baseline_grid.py

Expects ``data/preprocessed/char_data.parquet`` from
``scripts/00_build_panel.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sweep import (
    FORECAST_START,
    RFF_GAMMA,
    TRAIN_WINDOW_MONTHS,
    load_real_panel,
    run_one_fit,
)


# Paper §9 baseline grid.
SWEEP_TAG: str = "real_baseline"
K_VALUES: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)
P_VALUES: tuple[int, ...] = (64, 128, 256, 512, 1024, 4096, 10000)
Z_VALUES: tuple[float, ...] = (0.0, 10.0, 20.0, 50.0, 100.0)
RFF_SEED: int = 42  # baseline grid uses a single fixed seed per cell
TOP_N_UNIVERSE: int = 50

OUT_PATH = REPO_ROOT / "results" / "data" / "real_baseline_grid.csv"


def main() -> None:
    print("Loading top-50 fixed-universe panel...")
    panel = load_real_panel(top_n=TOP_N_UNIVERSE)
    print(
        f"  panel: {panel.shape}, "
        f"{panel['permno'].nunique()} permnos, "
        f"{panel['yyyymm'].nunique()} months"
    )

    cells = [(k, p, z) for k in K_VALUES for p in P_VALUES for z in Z_VALUES]
    print(f"Baseline grid: {len(cells)} cells")

    rows: list[dict] = []
    for i, (k, p, z) in enumerate(cells, start=1):
        print(f"  [{i:>3}/{len(cells)}] k={k:>2}, P={p:>5}, z={z:>5}", flush=True)
        metrics = run_one_fit(
            panel,
            k=k,
            train_window_months=TRAIN_WINDOW_MONTHS,
            forecast_start=FORECAST_START,
            rff_n_components=p,
            rff_gamma=RFF_GAMMA,
            rff_seed=RFF_SEED,
            alpha=z,
            cross_sectional_normalize=True,
            min_train_obs=TOP_N_UNIVERSE * 6,  # need at least 6 months of cross-sections
        )
        if metrics is not None:
            rows.append({
                "sweep": SWEEP_TAG,
                "k": k, "P": p, "z": z, "T": TRAIN_WINDOW_MONTHS,
                **metrics,
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
