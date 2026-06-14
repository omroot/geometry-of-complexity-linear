#!/usr/bin/env python3
"""
Colab runner: replicate all_rff_results.csv cells for P=1024.

Run this on Google Colab Pro after:
  1. Mounting Google Drive and uploading char_data.parquet there
  2. pip install -r requirements.txt
  3. Setting CHAR_DATA_PARQUET env var (see setup cell below)

Colab setup cell (paste into Colab):
─────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess
subprocess.run(['git','clone','https://github.com/YOUR_ORG/geometry-of-complexity-linear',
                '/content/repo'], check=True)
os.chdir('/content/repo')
subprocess.run(['pip','install','-q','-r','requirements.txt'], check=True)

# Point to the parquet you uploaded to Drive:
os.environ['CHAR_DATA_PARQUET'] = '/content/drive/MyDrive/char_data.parquet'

# Then run:
# !python -u scripts/experiments/real_data/05_colab_runner.py
─────────────────────────────────────

Cells: all P=1024 rows from data/reference/all_rff_results.csv (29 cells × 3 seeds).
These complete the laptop-infeasible portion of the comparison grid.

Output: results/data/real_baseline_grid.csv  (resume-safe, same file as laptop runs).

Estimated runtime on Colab Pro high-RAM CPU (~3-5x laptop speed):
  k=4,  P=1024 (3 cells) :  ~3h
  k=8,  P=1024 (4 cells) :  ~6h
  k=12, P=1024 (6 cells) :  ~9h
  k=16, P=1024 (4 cells) : ~12h
  k=20, P=1024 (4 cells) : ~16h
  k=24, P=1024 (4 cells) : ~19h
  k=28, P=1024 (2 cells) : ~22h
  k=32, P=1024 (2 cells) : ~24h   ← full P=1024 suite ~24h total
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

# ── Config ────────────────────────────────────────────────────────────────
SWEEP_TAG      : str             = "real_baseline"
RFF_SEEDS      : tuple[int, ...] = (0, 1, 2)        # 3 seeds, matches VoC
TOP_N_UNIVERSE : int             = 50

OUT_PATH   = REPO_ROOT / "results" / "data" / "real_baseline_grid.csv"
REF_CSV    = REPO_ROOT / "data" / "reference" / "all_rff_results.csv"

# Cells to run: all P=1024 rows from the VoC reference
P_TARGET = 1024


def main() -> None:
    if not REF_CSV.exists():
        raise FileNotFoundError(f"Reference CSV not found: {REF_CSV}")

    ref = pd.read_csv(REF_CSV)
    cells_df = (
        ref[ref["P"] == P_TARGET][["k", "P", "z"]]
        .drop_duplicates()
        .assign(est_h=lambda df: df.apply(
            lambda r: 0.0776 * r.P * (r.k / 4) * len(RFF_SEEDS) / 60, axis=1))
        .sort_values("est_h")
        .reset_index(drop=True)
    )
    cells = [(int(r.k), int(r.P), float(r.z)) for _, r in cells_df.iterrows()]

    # ── Load panel ────────────────────────────────────────────────────────
    voc_permnos = load_voc_top50_permnos()
    if voc_permnos is None:
        raise RuntimeError(
            "top50_permnos.csv not found. Expected at data/reference/top50_permnos.csv."
        )
    print(f"Universe: {len(voc_permnos)} VoC permnos from data/reference/top50_permnos.csv")
    panel = load_real_panel(permno_list=voc_permnos)
    print(
        f"Panel: {panel.shape}, {panel['permno'].nunique()} permnos, "
        f"{panel['yyyymm'].nunique()} months"
    )

    # ── Resume ────────────────────────────────────────────────────────────
    n_cells = len(cells)
    rows: list[dict] = []
    done: set = set()
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        rows = existing.to_dict("records")
        done = {(r["k"], r["P"], r["z"]) for r in rows}
        n_skip = sum(1 for k, p, z in cells if (k, p, z) in done)
        n_remain = n_cells - n_skip
        total_remain_h = cells_df[~cells_df.apply(
            lambda r: (int(r.k), int(r.P), float(r.z)) in done, axis=1
        )]["est_h"].sum()
        print(f"\nP={P_TARGET} grid: {n_cells} cells  ({n_skip} done, {n_remain} remaining)")
        print(f"Estimated remaining: ~{total_remain_h:.0f}h")
    else:
        print(f"\nP={P_TARGET} grid: {n_cells} cells  (~{cells_df.est_h.sum():.0f}h estimated)")

    # ── Main loop ─────────────────────────────────────────────────────────
    for i, (k, p, z) in enumerate(cells, start=1):
        rff_n = p // 2   # output dim = p, matches VoC convention
        est_h = 0.0776 * p * (k / 4) * len(RFF_SEEDS) / 60

        if (k, p, z) in done:
            print(f"  [{i:>2}/{n_cells}] k={k:>2} P={p} z={z:>5.1f} [skip]", flush=True)
            continue

        print(f"  [{i:>2}/{n_cells}] k={k:>2} P={p} z={z:>5.1f} rff_n={rff_n}"
              f"  (~{est_h:.1f}h)", flush=True)

        metrics = run_seed_averaged_fit(
            panel,
            rff_seeds=RFF_SEEDS,
            k=k,
            train_window_months=TRAIN_WINDOW_MONTHS,
            forecast_start=FORECAST_START,
            rff_n_components=rff_n,
            rff_gamma=RFF_GAMMA,
            alpha=z,
            cross_sectional_normalize=False,
            voc_rectangular=True,
            rectangular_ids=voc_permnos,
            cross_sectional_ddof=1,
            voc_max_nan_frac=0.3,
            voc_max_zero_frac=1.0,
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
            print(f"      → written to {OUT_PATH.name}", flush=True)
        else:
            print(f"      → WARNING: no predictions returned", flush=True)

    print(f"\nDone. {len(rows)} total rows in {OUT_PATH}")


if __name__ == "__main__":
    main()
