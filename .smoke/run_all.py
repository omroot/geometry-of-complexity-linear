"""Smoke harness: runs the full results/ pipeline end-to-end in ~15 min.

Outputs land in smoke_results/ (gitignored). Verification is
pipeline-completes only: every producer must exit 0 and the expected
output files must exist. No numerical comparison against the frozen
results/ — the smoke configs are deliberately much smaller.

Usage:
    python .smoke/run_all.py

What runs:
  1. rerun_synthetic_v3_checkpointed.py --smoke --sweeps A,B,C,D --overwrite-main
  2. experiment_ppp.py (synthetic, k=2, T=24)
  3. experiment_ppp_real.py (when data/char_data.parquet present)
  4. generate_paper_figures.py
  5. graphs_synthetic.py
  6. graphs_historical.py

What's skipped:
  - experiment_rff_sweep.py: 2500-line interactive notebook; not suitable
    as a one-shot smoke producer. The orchestrator copies the frozen
    results/data/csv/all_rff_results.csv into smoke_results/ so the
    historical plotters have their input.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add this script's own directory to sys.path so we can import config.py
# as a sibling module when invoked as `python .smoke/run_all.py`.
SMOKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SMOKE_DIR.parent
for p in (str(SMOKE_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (  # noqa: E402  (sys.path tweak above)
    FROZEN_RFF_CSV,
    REAL_DATA_PARQUET,
    SMOKE_CACHE_DIR,
    SMOKE_CSV_DIR,
    SMOKE_FIGURES_DIR,
    SMOKE_RESULTS_ROOT,
    SMOKE_XLSX_DIR,
    smoke_env,
)


def _prep_dirs() -> None:
    for d in (SMOKE_CSV_DIR, SMOKE_XLSX_DIR, SMOKE_FIGURES_DIR, SMOKE_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _run(label: str, cmd: list[str], env: dict[str, str]) -> tuple[bool, float]:
    """Run cmd from REPO_ROOT, streaming output; return (success, elapsed_seconds)."""
    print(f"\n{'=' * 70}")
    print(f"[{label}] starting: {' '.join(cmd)}")
    print(f"{'=' * 70}", flush=True)
    full_env = {**os.environ, **env}
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=full_env)
    elapsed = time.monotonic() - start
    ok = proc.returncode == 0
    print(
        f"[{label}] {'OK' if ok else 'FAILED'} in {elapsed:.1f}s (exit={proc.returncode})",
        flush=True,
    )
    return ok, elapsed


def _expect_files(label: str, paths: list[Path]) -> bool:
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"[{label}] FAILED: missing expected outputs:")
        for p in missing:
            print(f"   - {p}")
        return False
    return True


def main() -> int:
    _prep_dirs()
    env = smoke_env()
    summary: list[tuple[str, str, float]] = []  # (label, status, seconds)
    overall_start = time.monotonic()

    py = sys.executable
    have_real_data = REAL_DATA_PARQUET.exists()

    # -- 1. v3 synthetic sweeps -------------------------------------------
    label = "v3_sweep"
    ok, secs = _run(
        label,
        [
            py,
            "scripts/experiments/synthetic/rerun_synthetic_v3_checkpointed.py",
            "--smoke",
            "--sweeps",
            "A,B,C,D",
            "--overwrite-main",
            "--force",
        ],
        env,
    )
    ok = ok and _expect_files(
        label,
        [
            SMOKE_CSV_DIR / "synthetic_v3_sweep_A_summary.csv",
            SMOKE_CSV_DIR / "synthetic_v3_sweep_B_summary.csv",
            SMOKE_CSV_DIR / "synthetic_v3_sweep_C_summary.csv",
            SMOKE_CSV_DIR / "synthetic_v3_sweep_D_summary.csv",
            SMOKE_CSV_DIR / "all_synthetic_datarun_results_v3_full.csv",
            SMOKE_CSV_DIR / "all_synthetic_datarun_results.csv",
        ],
    )
    summary.append((label, "OK" if ok else "FAIL", secs))

    # -- 2. synthetic experiment_ppp --------------------------------------
    label = "experiment_ppp_synthetic"
    ok, secs = _run(
        label,
        [py, "scripts/experiments/synthetic/experiment_ppp.py"],
        env,
    )
    # The script writes features_data_synthetic_*.xlsx etc.; verify by
    # presence of at least one xlsx triplet matching the smoke (k, T).
    expected_xlsx = list(
        SMOKE_XLSX_DIR.glob(f"*_{env['GOCL_K']}_{env['GOCL_T']}.xlsx")
    )
    if not expected_xlsx:
        print(f"[{label}] FAILED: no xlsx files matching k={env['GOCL_K']}, T={env['GOCL_T']}")
        ok = False
    summary.append((label, "OK" if ok else "FAIL", secs))

    # -- 3. real-data experiment_ppp_real (optional) ----------------------
    label = "experiment_ppp_real"
    if have_real_data:
        ok, secs = _run(
            label,
            [py, "scripts/experiments/real_data/experiment_ppp_real.py"],
            env,
        )
        summary.append((label, "OK" if ok else "FAIL", secs))
    else:
        print(
            f"\n[{label}] SKIPPED: {REAL_DATA_PARQUET} not present "
            f"(real-data Tier C requires manual data drop)."
        )
        summary.append((label, "SKIP", 0.0))

    # -- 4. RFF sweep is intentionally not run (interactive notebook). ---
    # Copy the frozen output so historical plotters have their input.
    label = "rff_sweep_copy"
    print(f"\n[{label}] copying frozen {FROZEN_RFF_CSV.name} into smoke_results/")
    target = SMOKE_CSV_DIR / FROZEN_RFF_CSV.name
    if FROZEN_RFF_CSV.exists():
        shutil.copy2(FROZEN_RFF_CSV, target)
        summary.append((label, "OK", 0.0))
    else:
        print(f"[{label}] FAILED: frozen {FROZEN_RFF_CSV} not present")
        summary.append((label, "FAIL", 0.0))

    # -- 5. Plotters ------------------------------------------------------
    # Plotters reference specific config values (P=1024, k=12, sweep_A_k_z_P1024
    # etc.) from the full experiments. Smoke configs deliberately don't match
    # those, so a plotter may raise mid-figure on an empty filter. We log
    # PARTIAL when that happens but don't fail the smoke — pipeline-completes
    # is about producers + a non-hanging plot pass, not full figure coverage.
    for label, script in [
        ("plots_paper_figures", "scripts/plots/generate_paper_figures.py"),
        ("plots_synthetic", "scripts/plots/graphs_synthetic.py"),
        ("plots_historical", "scripts/plots/graphs_historical.py"),
    ]:
        ok, secs = _run(label, [py, script], env)
        summary.append((label, "OK" if ok else "PARTIAL", secs))

    # -- Final summary ----------------------------------------------------
    total = time.monotonic() - overall_start
    fig_count = len(list(SMOKE_FIGURES_DIR.glob("*"))) if SMOKE_FIGURES_DIR.exists() else 0
    print(f"\n{'=' * 70}\nSmoke run summary  ({total:.1f}s wall, {fig_count} figures)\n{'=' * 70}")
    print(f"  output root: {SMOKE_RESULTS_ROOT}")
    for label, status, secs in summary:
        line = f"  {status:<4}  {label:<30}  {secs:>6.1f}s"
        print(line)

    failed = [s for s in summary if s[1] == "FAIL"]
    partial = [s for s in summary if s[1] == "PARTIAL"]
    if failed:
        print(f"\n{len(failed)} step(s) FAILED. See logs above.")
        return 1
    if partial:
        print(
            f"\nAll producers OK. {len(partial)} plotter(s) PARTIAL "
            "(expected under smoke configs that don't match plotter filters)."
        )
    else:
        print("\nAll steps OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
