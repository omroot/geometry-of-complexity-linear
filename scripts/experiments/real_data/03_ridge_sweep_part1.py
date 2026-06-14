#!/usr/bin/env python3
"""Paper §9.5 ridge slice part 1: z in {0, 1, 5}.

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
    load_voc_top50_permnos,
    run_one_fit,
)


# Paper §9.5 grid.
SWEEP_TAG: str = "real_ridge_sweep"
K_FIXED: int = 12
RFF_M: int = 1024
Z_VALUES: tuple[float, ...] = (0.0, 1.0, 5.0)
SEEDS: tuple[int, ...] = (10, 20, 30)
TOP_N_UNIVERSE: int = 50

OUT_PATH = REPO_ROOT / "results" / "data" / "real_ridge_sweep.csv"


def main() -> None:
    print("Loading top-50 fixed-universe panel...")
    voc_permnos = load_voc_top50_permnos()
    panel = load_real_panel(
        permno_list=voc_permnos,
        top_n=None if voc_permnos is not None else TOP_N_UNIVERSE,
    )
    rectangular_ids = voc_permnos or sorted(panel["permno"].unique().tolist())
    print(f"  panel: {panel.shape}, {panel['permno'].nunique()} permnos")

    cells = [(z, seed) for z in Z_VALUES for seed in SEEDS]
    print(f"Ridge sweep: {len(cells)} cells")

    rows: list[dict] = []
    done: set[tuple[float, int]] = set()
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        rows = existing.to_dict("records")
        done = {(float(r["z"]), int(r["seed"])) for r in rows if "seed" in r}
        n_skip = sum(1 for cell in cells if cell in done)
        print(f"  Resuming: {n_skip} of {len(cells)} cells already done")

    for i, (z, seed) in enumerate(cells, start=1):
        if (z, seed) in done:
            print(
                f"  [{i:>2}/{len(cells)}] k={K_FIXED}, P={RFF_M}, "
                f"z={z:>5}, seed={seed} [skip]",
                flush=True,
            )
            continue
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
            rff_n_components=RFF_M // 2,
            rff_gamma=RFF_GAMMA,
            rff_seed=seed,
            alpha=z,
            cross_sectional_normalize=True,
            voc_rectangular=True,
            rectangular_ids=rectangular_ids,
            cross_sectional_ddof=1,
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
