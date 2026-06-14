#!/usr/bin/env python3
"""
Colab P=1024 runner — parallel seeds, per-seed checkpointing, Drive backup.

USAGE (in Colab):
    !python -u scripts/experiments/real_data/06_colab_p1024_runner.py --part 01

COLAB SETUP (paste these cells before running):
────────────────────────────────────────────────────────────────────
# Cell 1: mount Drive, clone repo, install deps
from google.colab import drive
drive.mount('/content/drive')
import os, subprocess
subprocess.run(['git','clone','https://github.com/omroot/geometry-of-complexity-linear',
                '/content/repo'], check=True)
os.chdir('/content/repo')
subprocess.run(['pip','install','-q','-r','requirements.txt'], check=True)

# Cell 2: set paths
import os
os.environ['CHAR_DATA_PARQUET'] = '/content/drive/MyDrive/char_data.parquet'
os.environ['DRIVE_BACKUP_PATH'] = '/content/drive/MyDrive/real_baseline_grid.csv'

# Cell 3: restore any previous results from Drive
import shutil, os
backup = os.environ.get('DRIVE_BACKUP_PATH','')
if backup and os.path.exists(backup):
    os.makedirs('/content/repo/results/data', exist_ok=True)
    shutil.copy(backup, '/content/repo/results/data/real_baseline_grid.csv')
    print(f"Restored results from Drive ({os.path.getsize(backup)//1024}KB)")

# Cell 4: run (change --part each session)
!python -u scripts/experiments/real_data/06_colab_p1024_runner.py --part 01
────────────────────────────────────────────────────────────────────

PARTS (run in order, one per Colab session):
  Part 01: k= 4 z=0,10,20          ~4h   ← RUN TONIGHT
  Part 02: k= 8 z=0,10,20,50       ~11h
  Part 03: k=12 z=0,1,5            ~12h
  Part 04: k=12 z=10,20,100        ~12h
  Part 05: k=16 z=0,10             ~11h
  Part 06: k=16 z=20,50            ~11h
  Part 07: k=20 z=0                ~7h
  Part 08: k=20 z=10               ~7h
  Part 09: k=20 z=20               ~7h
  Part 10: k=20 z=50               ~7h
  Part 11: k=24 z=0                ~8h
  Part 12: k=24 z=10               ~8h
  Part 13: k=24 z=20               ~8h
  Part 14: k=24 z=50               ~8h
  Part 15: k=28 z=0                ~9h
  Part 16: k=28 z=10               ~9h
  Part 17: k=32 z=0                ~11h
  Part 18: k=32 z=10               ~11h

All 29 P=1024 cells across 18 parts.  Times assume parallel seed execution.
Features:
  - 3 seeds run IN PARALLEL  (3x speedup vs sequential)
  - Per-seed result saved immediately to results/data/seeds/
  - Main CSV updated after each cell completes
  - Drive backup synced after each cell (set DRIVE_BACKUP_PATH)
  - Full resume: if session dies mid-cell, only unfinished seeds rerun
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# ── Part definitions — each ≤12h with parallel seeds ──────────────────────
P = 1024
PARTS: dict[str, list[tuple[int, float]]] = {
    "01": [(4,  0.0), (4, 10.0), (4, 20.0)],
    "02": [(8,  0.0), (8, 10.0), (8, 20.0), (8, 50.0)],
    "03": [(12, 0.0), (12, 1.0), (12, 5.0)],
    "04": [(12,10.0), (12,20.0), (12,100.0)],
    "05": [(16, 0.0), (16,10.0)],
    "06": [(16,20.0), (16,50.0)],
    "07": [(20, 0.0)],
    "08": [(20,10.0)],
    "09": [(20,20.0)],
    "10": [(20,50.0)],
    "11": [(24, 0.0)],
    "12": [(24,10.0)],
    "13": [(24,20.0)],
    "14": [(24,50.0)],
    "15": [(28, 0.0)],
    "16": [(28,10.0)],
    "17": [(32, 0.0)],
    "18": [(32,10.0)],
}

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_CSV    = REPO_ROOT / "results" / "data" / "real_baseline_grid.csv"
SEEDS_DIR  = REPO_ROOT / "results" / "data" / "seeds"
PERMNOS_CSV = REPO_ROOT / "data" / "reference" / "top50_permnos.csv"

SWEEP_TAG = "real_baseline"
RFF_SEEDS = (0, 1, 2)
RFF_N     = P // 2   # 512 — matches VoC convention (output dim = P)
RFF_GAMMA = 0.25
TRAIN_WINDOW = 24
FORECAST_START = "1996-01-01"


# ── Worker function (module-level so ProcessPoolExecutor can pickle it) ────
def _seed_worker(args: tuple) -> tuple[int, dict | None]:
    """Runs in a subprocess — loads panel then fits one seed.

    Returns (seed_int, metrics_dict_or_None).
    All imports are local so the worker process starts cleanly.
    """
    panel_path, permnos_path, seed, k, z, repo_root_str = args

    import os as _os, sys as _sys, warnings as _w
    _w.filterwarnings("ignore")
    if panel_path:
        _os.environ["CHAR_DATA_PARQUET"] = str(panel_path)
    _sys.path.insert(0, repo_root_str)

    import pandas as _pd
    from src.sweep.runner import load_real_panel, run_one_fit

    permnos = _pd.read_csv(permnos_path)["permno"].astype(int).tolist()
    panel   = load_real_panel(permno_list=permnos)

    metrics = run_one_fit(
        panel,
        k=k,
        train_window_months=TRAIN_WINDOW,
        forecast_start=FORECAST_START,
        rff_n_components=RFF_N,
        rff_gamma=RFF_GAMMA,
        rff_seed=seed,
        alpha=z,
        cross_sectional_normalize=False,
        voc_rectangular=True,
        rectangular_ids=permnos,
        cross_sectional_ddof=1,
        voc_max_nan_frac=0.3,
        voc_max_zero_frac=1.0,
        min_train_obs=50 * 6,
    )
    return seed, metrics


# ── Per-seed save / load ───────────────────────────────────────────────────
def _seed_path(k: int, z: float, seed: int) -> Path:
    return SEEDS_DIR / f"k{k}_P{P}_z{z:.1f}_seed{seed}.json"


def _save_seed(k: int, z: float, seed: int, metrics: dict | None) -> None:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_seed_path(k, z, seed), "w") as f:
        json.dump(metrics, f)


def _load_seed(k: int, z: float, seed: int) -> dict | None:
    p = _seed_path(k, z, seed)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _clear_seeds(k: int, z: float) -> None:
    for seed in RFF_SEEDS:
        _seed_path(k, z, seed).unlink(missing_ok=True)


# ── Drive sync ─────────────────────────────────────────────────────────────
def _sync_to_drive() -> None:
    backup = os.environ.get("DRIVE_BACKUP_PATH", "")
    if backup and OUT_CSV.exists():
        try:
            shutil.copy(OUT_CSV, backup)
            print(f"      📁 synced → {backup}", flush=True)
        except Exception as e:
            print(f"      ⚠️  Drive sync failed: {e}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=list(PARTS.keys()),
                        help="Which part to run (01–18)")
    args = parser.parse_args()

    cells_kz = PARTS[args.part]
    cells    = [(k, P, z) for k, z in cells_kz]
    est_h    = sum(0.0776 * P * (k / 4) / 60 for k, _ in cells_kz)
    print(f"\n=== Part {args.part} — {len(cells)} cells, ~{est_h:.0f}h estimated ===")
    print(f"    3 seeds per cell run IN PARALLEL  (rff_n={RFF_N})")
    print(f"    Results: {OUT_CSV.relative_to(REPO_ROOT)}\n")

    # Paths for workers
    panel_path   = os.environ.get("CHAR_DATA_PARQUET", "")
    permnos_path = str(PERMNOS_CSV)

    # Load existing results for resume
    rows: list[dict] = []
    main_done: set[tuple] = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        main_done = {(r["k"], r["P"], r["z"]) for r in rows}

    for cell_num, (k, p_val, z) in enumerate(cells, start=1):
        cell_key = (k, p_val, z)
        est_cell = 0.0776 * P * (k / 4) / 60

        if cell_key in main_done:
            print(f"  [{cell_num}/{len(cells)}] k={k:>2} P={P} z={z:>6.1f}  [skip — already in CSV]",
                  flush=True)
            continue

        print(f"  [{cell_num}/{len(cells)}] k={k:>2} P={P} z={z:>6.1f}  (~{est_cell:.1f}h parallel)",
              flush=True)

        # Check which seeds already have saved results
        pending_seeds = [s for s in RFF_SEEDS if _load_seed(k, z, s) is None]
        completed_seeds = [s for s in RFF_SEEDS if s not in pending_seeds]
        if completed_seeds:
            print(f"      Resuming — seeds {completed_seeds} already done, "
                  f"running {pending_seeds}", flush=True)

        # Run pending seeds in parallel
        if pending_seeds:
            worker_args = [(panel_path, permnos_path, s, k, z, str(REPO_ROOT))
                           for s in pending_seeds]
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=len(pending_seeds)) as pool:
                futures = {pool.submit(_seed_worker, arg): arg[2]
                           for arg in worker_args}
                for future in as_completed(futures):
                    seed_val = futures[future]
                    try:
                        seed_out, metrics = future.result()
                        _save_seed(k, z, seed_out, metrics)
                        if metrics is not None:
                            print(f"      seed={seed_out}: sharpe={metrics['sharpe']:+.4f}"
                                  f"  r2={metrics['oos_r2']:.4f}"
                                  f"  erank={metrics['mean_erank']:.4f}  ✓ saved",
                                  flush=True)
                        else:
                            print(f"      seed={seed_out}: returned None", flush=True)
                    except Exception as e:
                        print(f"      seed={seed_val}: ERROR — {e}", flush=True)
            print(f"      parallel wall time: {(time.time()-t0)/3600:.2f}h", flush=True)

        # Average all 3 seeds
        seed_results = [_load_seed(k, z, s) for s in RFF_SEEDS]
        valid = [m for m in seed_results if m is not None]
        if not valid:
            print(f"      WARNING: no valid seeds — skipping cell", flush=True)
            continue

        avg_metrics = {
            key: float(np.mean([m[key] for m in valid]))
            for key in valid[0]
        }
        avg_metrics["n_seeds"] = len(valid)

        rows.append({
            "sweep": SWEEP_TAG,
            "k": k, "P": P, "z": z, "T": TRAIN_WINDOW,
            **avg_metrics,
        })
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"      avg → sharpe={avg_metrics['sharpe']:+.4f}"
              f"  r2={avg_metrics['oos_r2']:.4f}"
              f"  erank={avg_metrics['mean_erank']:.4f}"
              f"  ✓ written to CSV", flush=True)

        # Clean up seed files for this cell
        _clear_seeds(k, z)

        # Sync to Drive
        _sync_to_drive()

    print(f"\nPart {args.part} complete. {len(rows)} total rows in CSV.")
    _sync_to_drive()


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on some platforms (macOS)
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
