#!/usr/bin/env python3
"""Paper §8.5 short-window rank-aligned extension.

Three (k, T) cells with the underlying synthetic rank fixed at k0=18 (per
§8.5 text). Each cell sweeps the RFF dimension across the paper-spec
linear-baseline + 6 small + 4 large grid, with multiple seeds per dim.

Drives Figures 8-11 (§8.5) and Table 2.

Outputs:
    results/data/short_window.csv -- per (k, T, P, seed) row.

Usage:
    python scripts/experiments/synthetic/06_short_window.py
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
    BASELINE_RFF,
    FORECAST_START,
    LARGE_RFF,
    PAPER_DGP_SHORT_WINDOW,
    RFF_GAMMA,
    SEEDS_LARGE,
    SEEDS_SMALL,
    SMALL_RFF,
    SweepCell,
    run_sweep,
)


# Paper §8.5 (k, T) configurations.
KT_CONFIGS: tuple[tuple[int, int], ...] = ((17, 24), (18, 24), (18, 36))
ALPHA: float = 100.0


def cells_for_kt(k: int, T: int) -> list[SweepCell]:
    """Build the (k, T, P, seed) SweepCells for one (k, T) configuration."""
    def make_cell(k: int, T: int, p: int, seed: int) -> SweepCell:
        return SweepCell(
            identity={"k": k, "T": T, "P": p, "seed": seed},
            fit_kwargs=dict(
                k=k,
                train_window_months=T,
                forecast_start=FORECAST_START,
                rff_n_components=p,
                rff_gamma=RFF_GAMMA,
                rff_seed=seed,
                alpha=ALPHA,
            ),
        )

    cells: list[SweepCell] = [make_cell(k, T, BASELINE_RFF, SEEDS_SMALL[0])]
    for p in SMALL_RFF:
        for seed in SEEDS_SMALL:
            cells.append(make_cell(k, T, p, seed))
    for p in LARGE_RFF:
        for seed in SEEDS_LARGE:
            cells.append(make_cell(k, T, p, seed))
    return cells


OUT_PATH = REPO_ROOT / "results" / "data" / "short_window.csv"


def main() -> None:
    print(f"Generating §8.5 panel with k0={PAPER_DGP_SHORT_WINDOW['k0']}...")
    panel, _ = generate_workflow_panel(**PAPER_DGP_SHORT_WINDOW)
    all_cells = [c for k, T in KT_CONFIGS for c in cells_for_kt(k, T)]
    print(f"Short window: {len(all_cells)} cells across {len(KT_CONFIGS)} (k, T) configs")

    rows = run_sweep(panel, all_cells)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
