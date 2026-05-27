#!/usr/bin/env python3
"""Paper §9.5 ridge slice: top-50 fixed, k=12, m=1024, finer z grid.

Drives Figure 13: geometric metrics and Sharpe co-move with ridge
regularisation on real data, mirroring the synthetic Sweep C story.

Three seeds per ridge value smooth the RFF sampling noise.

Outputs:
    results/data/real_ridge_sweep.csv -- per (z, seed) row.

Usage:
    python scripts/experiments/real_data/03_ridge_sweep.py
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


# Paper §9.5 grid.
SWEEP_TAG: str = "real_ridge_sweep"
K_FIXED: int = 12
RFF_M: int = 1024
Z_VALUES: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
SEEDS: tuple[int, ...] = (10, 20, 30)
TOP_N_UNIVERSE: int = 50

OUT_PATH = REPO_ROOT / "results" / "data" / "real_ridge_sweep.csv"


def main() -> None:
    print("Loading top-50 fixed-universe panel...")
    panel = load_real_panel(top_n=TOP_N_UNIVERSE)
    print(f"  panel: {panel.shape}")

    cells = [(z, seed) for z in Z_VALUES for seed in SEEDS]
    print(f"Ridge sweep: {len(cells)} cells")

    rows: list[dict] = []
    for i, (z, seed) in enumerate(cells, start=1):
        print(
            f"  [{i:>2}/{len(cells)}] k={K_FIXED}, P={RFF_M}, "
            f"z={z:>5}, seed={seed}",
            flush=True,
        )
        metrics = run_one_fit(
            panel,
            k=K_FIXED,
            train_window_months=TRAIN_WINDOW_MONTHS,
            forecast_start=FORECAST_START,
            rff_n_components=RFF_M,
            rff_gamma=RFF_GAMMA,
            rff_seed=seed,
            alpha=z,
            cross_sectional_normalize=True,
            min_train_obs=TOP_N_UNIVERSE * 6,
        )
        if metrics is not None:
            rows.append({
                "sweep": SWEEP_TAG,
                "k": K_FIXED, "P": RFF_M, "z": z, "T": TRAIN_WINDOW_MONTHS,
                "seed": seed,
                **metrics,
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
