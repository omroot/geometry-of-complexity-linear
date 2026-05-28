#!/usr/bin/env python3
"""Sweep C from paper §8.2: vary ridge z at fixed k=12, m=1024.

Drives Figure 3 (§8.3.3): geometric metrics and Sharpe co-move with ridge
regularisation at the true rank.

Cells: 7 ridge values x N_SEEDS = up to 21 rolling fits.

Outputs:
    results/data/sweep_ridge.csv  -- per (k, P, z, seed) row.

Usage:
    python scripts/experiments/synthetic/03_sweep_ridge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate import generate_workflow_panel
from src.sweep import (
    FORECAST_START,
    PAPER_DGP_DEFAULTS,
    RFF_GAMMA,
    TRAIN_WINDOW_MONTHS,
    SweepCell,
    run_sweep,
)


# Paper §8.2 Sweep C grid.
SWEEP_TAG: str = "sweep_C_fine_z"
K_FIXED: int = 12
RFF_M: int = 1024
Z_VALUES: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
SEEDS: tuple[int, ...] = (10, 20, 30)

OUT_PATH = REPO_ROOT / "results" / "data" / "sweep_ridge.csv"


def main() -> None:
    print(f"Generating §8.1 panel...")
    panel, _ = generate_workflow_panel(**PAPER_DGP_DEFAULTS)
    cells = [
        SweepCell(
            identity={"sweep": SWEEP_TAG, "k": K_FIXED, "P": RFF_M, "z": z, "seed": seed},
            fit_kwargs=dict(
                k=K_FIXED,
                train_window_months=TRAIN_WINDOW_MONTHS,
                forecast_start=FORECAST_START,
                rff_n_components=RFF_M,
                rff_gamma=RFF_GAMMA,
                rff_seed=seed,
                alpha=z,
            ),
        )
        for z in Z_VALUES for seed in SEEDS
    ]
    print(f"Sweep C: {len(cells)} cells")

    rows = run_sweep(panel, cells)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
