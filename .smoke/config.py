"""Single source of truth for the .smoke/ harness.

Defines output paths and the env vars passed to each producer/plotter.
Importable from :mod:`.smoke.run_all`.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Output roots (mirror the structure of results/).
SMOKE_RESULTS_ROOT = REPO_ROOT / "smoke_results"
SMOKE_CSV_DIR = SMOKE_RESULTS_ROOT / "data" / "csv"
SMOKE_XLSX_DIR = SMOKE_RESULTS_ROOT / "data" / "xlsx"
SMOKE_FIGURES_DIR = SMOKE_RESULTS_ROOT / "figures"
SMOKE_CACHE_DIR = SMOKE_RESULTS_ROOT / "cache"

# Frozen artifact in the repo that we copy into smoke_results/ when
# experiment_rff_sweep.py is skipped (it's an interactive notebook, not
# suitable as a one-shot smoke producer).
FROZEN_RFF_CSV = REPO_ROOT / "results" / "data" / "csv" / "all_rff_results.csv"

# Real-data Tier C input. If absent, the orchestrator skips the two
# real-data jobs and copies frozen xlsx counterparts (if available) so
# downstream plotters still have inputs.
REAL_DATA_PARQUET = REPO_ROOT / "data" / "char_data.parquet"

# (k, T) cell used by both synthetic and real experiment_ppp scripts.
SMOKE_K = 2
SMOKE_T = 24


def smoke_env() -> dict[str, str]:
    """Env vars passed to every subprocess spawned by the orchestrator."""
    return {
        "GOCL_SMOKE": "1",
        "GOCL_K": str(SMOKE_K),
        "GOCL_T": str(SMOKE_T),
        "XLSX_DIR": str(SMOKE_XLSX_DIR),
        "FIGURES_DIR": str(SMOKE_FIGURES_DIR),
        "GOCL_DATA_DIR": str(SMOKE_CSV_DIR),
        "GOCL_CACHE_DIR": str(SMOKE_CACHE_DIR),
        # Keep BLAS single-threaded so joblib workers don't oversubscribe cores.
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # Non-interactive matplotlib backend: prevents the plotters
        # (jupytext-paired notebooks) from blocking on plt.show().
        "MPLBACKEND": "Agg",
    }
