#!/usr/bin/env python3
"""Baseline grid — Part 3: k=8 P=512,1024.

Cells: 10  |  Estimated runtime: ~40h  (overnight/weekend run)
  k=8, P= 512  →  ~2.65h/cell × 5 z-values = 13h
  k=8, P=1024  →  ~5.29h/cell × 5 z-values = 26h

Recommended: start before a long weekend and let it run. The resume logic
checkpoints after every cell so it can be safely interrupted and restarted.
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
    run_seed_averaged_fit,
)

SWEEP_TAG: str = "real_baseline"
Z_VALUES:  tuple[float, ...] = (0.0, 10.0, 20.0, 50.0, 100.0)
RFF_SEEDS: tuple[int, ...] = (0, 1, 2)
TOP_N_UNIVERSE: int = 50

OUT_PATH = REPO_ROOT / "results" / "data" / "real_baseline_grid.csv"

CELLS: list[tuple[int, int, float]] = [
    (8, p, z) for p in (512, 1024) for z in Z_VALUES
]


def main() -> None:
    print("Loading top-50 fixed-universe panel...")
    voc_permnos = load_voc_top50_permnos()
    panel = load_real_panel(
        permno_list=voc_permnos,
        top_n=None if voc_permnos is not None else TOP_N_UNIVERSE,
    )
    rectangular_ids = voc_permnos or sorted(panel["permno"].unique().tolist())
    print(
        f"  panel: {panel.shape}, "
        f"{panel['permno'].nunique()} permnos, "
        f"{panel['yyyymm'].nunique()} months"
    )

    n_cells = len(CELLS)
    print(f"Part 3: {n_cells} cells  (~40h estimated — long weekend run)")

    done: set = set()
    rows: list[dict] = []
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        rows = existing.to_dict("records")
        done = {(r["k"], r["P"], r["z"]) for r in rows}
        n_skip = sum(1 for k, p, z in CELLS if (k, p, z) in done)
        print(f"  Resuming: {n_skip} of {n_cells} cells already done")

    for i, (k, p, z) in enumerate(CELLS, start=1):
        if (k, p, z) in done:
            print(f"  [{i:>2}/{n_cells}] k={k:>2}, P={p:>5}, z={z:>5} [skip]", flush=True)
            continue
        print(f"  [{i:>2}/{n_cells}] k={k:>2}, P={p:>5}, z={z:>5}", flush=True)
        metrics = run_seed_averaged_fit(
            panel,
            rff_seeds=RFF_SEEDS,
            k=k,
            train_window_months=TRAIN_WINDOW_MONTHS,
            forecast_start=FORECAST_START,
            rff_n_components=p // 2,
            rff_gamma=RFF_GAMMA,
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
                "k": k, "P": p, "z": z, "T": TRAIN_WINDOW_MONTHS,
                **metrics,
            })
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(rows)} total rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
