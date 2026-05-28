#!/usr/bin/env python3
"""Sweep B from paper §8.2: vary (m, k, z) on the (256, 1024, 4096) RFF grid.

Together with sweeps A/C/D, provides the full broad-grid table consumed by
Figure 1 (§8.3.1) drift-anisotropy scatter and Table 1 (§8.4)
metric-Sharpe correlations.

Cells: 3 RFF dims x 2 factor counts x 2 ridges x N_SEEDS = up to 36 rolling fits.

Outputs:
    results/data/sweep_pgrid.csv  -- per (k, P, z, seed) row.

Usage:
    python scripts/experiments/synthetic/05_sweep_pgrid.py
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


# Paper §8.2 Sweep B grid.
SWEEP_TAG: str = "sweep_B_P_grid"
RFF_M_VALUES: tuple[int, ...] = (256, 1024, 4096)
K_VALUES: tuple[int, ...] = (12, 24)
Z_VALUES: tuple[float, ...] = (10.0, 50.0)
SEEDS: tuple[int, ...] = (10, 20, 30)

OUT_PATH = REPO_ROOT / "results" / "data" / "sweep_pgrid.csv"


def main() -> None:
    print("Generating §8.1 panel...")
    panel, _ = generate_workflow_panel(**PAPER_DGP_DEFAULTS)
    cells = [
        SweepCell(
            identity={"sweep": SWEEP_TAG, "k": k, "P": m, "z": z, "seed": seed},
            fit_kwargs=dict(
                k=k,
                train_window_months=TRAIN_WINDOW_MONTHS,
                forecast_start=FORECAST_START,
                rff_n_components=m,
                rff_gamma=RFF_GAMMA,
                rff_seed=seed,
                alpha=z,
            ),
        )
        for k in K_VALUES for m in RFF_M_VALUES for z in Z_VALUES for seed in SEEDS
    ]
    print(f"Sweep B: {len(cells)} cells")

    rows = run_sweep(panel, cells)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
