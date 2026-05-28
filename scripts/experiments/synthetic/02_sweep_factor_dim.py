#!/usr/bin/env python3
"""Sweep A from paper §8.2: vary (k, z) at fixed RFF dim m=1024.

Drives Figures 2 (§8.3.2), 4 (§8.3.4), 5 (§8.3.5), 6 (§8.3.6) and
contributes to Figure 1 (§8.3.1) + Table 1 (§8.4) when combined with the
other sweep scripts.

Cells: 8 factor dims x 3 ridges x N_SEEDS = up to 72 rows, each one
rolling Grassmannian-IPCA fit on the §8.1 synthetic panel.

Outputs:
    results/data/sweep_factor_dim.csv  -- per (k, P, z, seed) row.

Usage:
    python scripts/experiments/synthetic/02_sweep_factor_dim.py
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


# Paper §8.2 Sweep A grid.
SWEEP_TAG: str = "sweep_A_k_z_P1024"
K_VALUES: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)
Z_VALUES: tuple[float, ...] = (0.0, 10.0, 50.0)
RFF_M: int = 1024
SEEDS: tuple[int, ...] = (10, 20, 30)

OUT_PATH = REPO_ROOT / "results" / "data" / "sweep_factor_dim.csv"


def main() -> None:
    print(f"Generating §8.1 panel with k0={PAPER_DGP_DEFAULTS['k0']}...")
    panel, _ = generate_workflow_panel(**PAPER_DGP_DEFAULTS)
    cells = [
        SweepCell(
            identity={"sweep": SWEEP_TAG, "k": k, "P": RFF_M, "z": z, "seed": seed},
            fit_kwargs=dict(
                k=k,
                train_window_months=TRAIN_WINDOW_MONTHS,
                forecast_start=FORECAST_START,
                rff_n_components=RFF_M,
                rff_gamma=RFF_GAMMA,
                rff_seed=seed,
                alpha=z,
            ),
        )
        for k in K_VALUES for z in Z_VALUES for seed in SEEDS
    ]
    print(f"Sweep A: {len(cells)} cells")

    rows = run_sweep(panel, cells)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
