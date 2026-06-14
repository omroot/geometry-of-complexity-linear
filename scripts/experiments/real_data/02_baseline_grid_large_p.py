#!/usr/bin/env python3
"""Baseline grid — Large-P appendix: P=4096, P=10000 for all k.

Cells: 80  |  Estimated runtime: ~6500h total  (server/cluster only)
  These cells require a GPU server or compute cluster.  NOT recommended for a laptop.

Per-(k,P) group estimates (5 z-values each):
  k= 4 P= 4096 →   53h    k= 4 P=10000 →  129h
  k= 8 P= 4096 →  106h    k= 8 P=10000 →  259h
  k=12 P= 4096 →  159h    k=12 P=10000 →  388h
  k=16 P= 4096 →  212h    k=16 P=10000 →  517h
  k=20 P= 4096 →  265h    k=20 P=10000 →  647h
  k=24 P= 4096 →  318h    k=24 P=10000 →  776h
  k=28 P= 4096 →  371h    k=28 P=10000 →  905h
  k=32 P= 4096 →  424h    k=32 P=10000 → 1035h

NOTE: These timing estimates are based on the CPU-only k=4 empirical baseline.
      A GPU or parallelised implementation could reduce these by 10-100×.
      The reference VoC results (4096_20_24_k.csv) were likely run on a server.

Writes to the same real_baseline_grid.csv as all other parts.
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
K_VALUES:  tuple[int, ...]   = (4, 8, 12, 16, 20, 24, 28, 32)
Z_VALUES:  tuple[float, ...] = (0.0, 10.0, 20.0, 50.0, 100.0)
RFF_SEEDS: tuple[int, ...] = (0, 1, 2)
TOP_N_UNIVERSE: int = 50

OUT_PATH = REPO_ROOT / "results" / "data" / "real_baseline_grid.csv"

CELLS: list[tuple[int, int, float]] = [
    (k, p, z)
    for k in K_VALUES
    for p in (4096, 10000)
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
    print(f"Large-P: {n_cells} cells  (~6500h total — SERVER ONLY)")

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
            print(f"  [{i:>2}/{n_cells}] k={k:>2}, P={p:>6}, z={z:>5} [skip]", flush=True)
            continue
        print(f"  [{i:>2}/{n_cells}] k={k:>2}, P={p:>6}, z={z:>5}", flush=True)
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
