#!/usr/bin/env python3
"""Baseline grid — Part 4: k=12..32, P=64..1024.

Cells: 150  |  Estimated runtime: ~850h total  (server / multi-week run)
  This part covers the full k-variation story for P=64..1024.
  Because individual cells range from 0.5h (k=12,P=64) to 106h (k=32,P=1024),
  interrupt and resume freely — the CSV checkpoint after each cell is safe.

Per-(k,P) group estimates (5 z-values each):
  k=12 P=  64 →  2.5h    k=16 P=  64 →  3.3h    k=20 P=  64 →  4.1h
  k=12 P= 128 →  5.0h    k=16 P= 128 →  6.6h    k=20 P= 128 →  8.3h
  k=12 P= 256 →  9.9h    k=16 P= 256 → 13.2h    k=20 P= 256 → 16.6h
  k=12 P= 512 → 19.9h    k=16 P= 512 → 26.5h    k=20 P= 512 → 33.1h
  k=12 P=1024 → 39.7h    k=16 P=1024 → 53.0h    k=20 P=1024 → 66.2h
  k=24 P=  64 →  5.0h    k=28 P=  64 →  5.8h    k=32 P=  64 →  6.6h
  k=24 P= 128 →  9.9h    k=28 P= 128 → 11.6h    k=32 P= 128 → 13.2h
  k=24 P= 256 → 19.9h    k=28 P= 256 → 23.2h    k=32 P= 256 → 26.5h
  k=24 P= 512 → 39.7h    k=28 P= 512 → 46.4h    k=32 P= 512 → 53.0h
  k=24 P=1024 → 79.5h    k=28 P=1024 → 92.7h    k=32 P=1024 →105.9h

Practical tip: run k=12..16 first (start with lower k), interrupt when needed,
then resume. The CSV checkpoint makes partial runs safe.
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
    (k, p, z)
    for k in (12, 16, 20, 24, 28, 32)
    for p in (64, 128, 256, 512, 1024)
    for z in Z_VALUES
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
    print(f"Part 4: {n_cells} cells  (~850h total — run incrementally, use resume)")

    done: set = set()
    rows: list[dict] = []
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        rows = existing.to_dict("records")
        done = {(r["k"], r["P"], r["z"]) for r in rows}
        n_skip = sum(1 for k, p, z in CELLS if (k, p, z) in done)
        print(f"  Resuming: {n_skip} of {n_cells} cells already done, {n_cells-n_skip} remaining")

    for i, (k, p, z) in enumerate(CELLS, start=1):
        if (k, p, z) in done:
            print(f"  [{i:>3}/{n_cells}] k={k:>2}, P={p:>5}, z={z:>5} [skip]", flush=True)
            continue
        print(f"  [{i:>3}/{n_cells}] k={k:>2}, P={p:>5}, z={z:>5}", flush=True)
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
