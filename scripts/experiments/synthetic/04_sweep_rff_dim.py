#!/usr/bin/env python3
"""Sweep D from paper §8.2: vary RFF dim m at k in {12, 24}, z=10.

Drives Figure 7 (§8.3.7): Sharpe and OOS R² rise with feature complexity
in the regularised regime.

Cells: 6 RFF dims x 2 factor counts x N_SEEDS = up to 36 rolling fits.

Outputs:
    results/data/sweep_rff_dim.csv  -- per (k, P, z, seed) row.

Usage:
    python scripts/experiments/synthetic/04_sweep_rff_dim.py
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
    run_one_fit,
)


# Paper §8.2 Sweep D grid.
SWEEP_TAG: str = "sweep_D_fine_P"
K_VALUES: tuple[int, ...] = (12, 24)
RFF_M_VALUES: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)
Z_FIXED: float = 10.0
SEEDS: tuple[int, ...] = (10, 20, 30)

OUT_PATH = REPO_ROOT / "results" / "data" / "sweep_rff_dim.csv"


def main() -> None:
    print("Generating §8.1 panel...")
    panel, _ = generate_workflow_panel(**PAPER_DGP_DEFAULTS)
    cells = [(k, m, seed) for k in K_VALUES for m in RFF_M_VALUES for seed in SEEDS]
    print(f"Sweep D: {len(cells)} cells")

    rows: list[dict] = []
    for i, (k, m, seed) in enumerate(cells, start=1):
        print(f"  [{i:>3}/{len(cells)}] k={k}, P={m:>4}, z={Z_FIXED}, seed={seed}", flush=True)
        metrics = run_one_fit(
            panel,
            k=k,
            train_window_months=TRAIN_WINDOW_MONTHS,
            forecast_start=FORECAST_START,
            rff_n_components=m,
            rff_gamma=RFF_GAMMA,
            rff_seed=seed,
            alpha=Z_FIXED,
        )
        if metrics is not None:
            rows.append({"sweep": SWEEP_TAG, "k": k, "P": m, "z": Z_FIXED, "seed": seed, **metrics})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
