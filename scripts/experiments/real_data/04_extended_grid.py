#!/usr/bin/env python3
"""Paper §9.10 extended backtest: 3 (k, T) cells x full RFF grid x multiple seeds.

Drives Figures 17 (OOS R^2 crossover), 18 (OOS R^2 across grid),
19 (effective rank + gap ratio), 20 (subspace stability).

Differs from the §9.1-§9.9 baseline (script 02) in two ways:

1. Universe: **rolling** top-50 by previous-month market cap, not
   fixed -- avoids the look-ahead caveat that the paper flags
   in §9.10. Driven by ``market_cap_col`` + ``market_cap_top_n``
   passed into :func:`src.models.rolling_fit.rolling_fit`.

2. Grid: focused on (k, T) cells the paper highlights for the
   crossover and high-complexity-tail story, with multiple seeds.

Outputs:
    results/data/real_extended.csv -- per (k, T, P, seed) row.

Usage:
    python scripts/experiments/real_data/04_extended_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sweep import (
    BASELINE_RFF,
    FORECAST_START,
    LARGE_RFF,
    RFF_GAMMA,
    SEEDS_LARGE,
    SEEDS_SMALL,
    SMALL_RFF,
    load_real_panel,
    run_one_fit,
)


# Paper §9.10 (k, T) cells.
SWEEP_TAG: str = "real_extended"
KT_CONFIGS: tuple[tuple[int, int], ...] = (
    (12, 24), (14, 36), (16, 24), (18, 24), (18, 36),
)
ALPHA: float = 100.0
TOP_N_UNIVERSE: int = 50

OUT_PATH = REPO_ROOT / "results" / "data" / "real_extended.csv"


def cells_for_kt(k: int, T: int) -> list[tuple[int, int, int]]:
    """Build the (T, P, seed) cells for one (k, T) config."""
    cells: list[tuple[int, int, int]] = [(T, BASELINE_RFF, SEEDS_SMALL[0])]
    for p in SMALL_RFF:
        for seed in SEEDS_SMALL:
            cells.append((T, p, seed))
    for p in LARGE_RFF:
        for seed in SEEDS_LARGE:
            cells.append((T, p, seed))
    return cells


def main() -> None:
    print("Loading full panel (rolling top-50 filter happens per OOS month)...")
    panel = load_real_panel()
    print(f"  panel: {panel.shape}, {panel['permno'].nunique()} unique permnos")

    all_cells = [
        (k, T, p, seed)
        for k, T in KT_CONFIGS for (T_, p, seed) in cells_for_kt(k, T)
    ]
    print(f"Extended grid: {len(all_cells)} cells across {len(KT_CONFIGS)} (k, T) configs")

    rows: list[dict] = []
    for i, (k, T, p, seed) in enumerate(all_cells, start=1):
        print(f"  [{i:>4}/{len(all_cells)}] k={k}, T={T}, P={p:>4}, seed={seed}", flush=True)
        metrics = run_one_fit(
            panel,
            k=k,
            train_window_months=T,
            forecast_start=FORECAST_START,
            rff_n_components=p,
            rff_gamma=RFF_GAMMA,
            rff_seed=seed,
            alpha=ALPHA,
            cross_sectional_normalize=True,
            market_cap_col="mcap",
            market_cap_top_n=TOP_N_UNIVERSE,
            min_train_obs=TOP_N_UNIVERSE * 6,
        )
        if metrics is not None:
            rows.append({
                "sweep": SWEEP_TAG,
                "k": k, "P": p, "z": ALPHA, "T": T, "seed": seed,
                **metrics,
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
