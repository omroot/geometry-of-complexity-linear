"""Sweep helpers shared by ``scripts/experiments/``.

* :func:`run_one_fit` — runs one rolling Grassmannian-IPCA fit and
  packages the metric row that every sweep CSV column-aligns with.
* :func:`load_real_panel` — loader for ``data/preprocessed/char_data.parquet``
  with optional top-N-by-mean-mcap restriction.
* Module-level constants: paper §8.1 DGP defaults, common rolling-fit
  knobs, and reusable RFF / seed grids.
"""

from src.sweep.defaults import (
    BASELINE_RFF,
    FORECAST_START,
    LARGE_RFF,
    PAPER_DGP_DEFAULTS,
    PAPER_DGP_SHORT_WINDOW,
    RFF_GAMMA,
    SEEDS_LARGE,
    SEEDS_SMALL,
    SMALL_RFF,
    TRAIN_WINDOW_MONTHS,
)
from src.sweep.parallel import SweepCell, run_sweep
from src.sweep.runner import DEFAULT_PANEL_PATH, load_real_panel, run_one_fit

__all__ = [
    "BASELINE_RFF",
    "DEFAULT_PANEL_PATH",
    "FORECAST_START",
    "LARGE_RFF",
    "PAPER_DGP_DEFAULTS",
    "PAPER_DGP_SHORT_WINDOW",
    "RFF_GAMMA",
    "SEEDS_LARGE",
    "SEEDS_SMALL",
    "SMALL_RFF",
    "TRAIN_WINDOW_MONTHS",
    "SweepCell",
    "load_real_panel",
    "run_one_fit",
    "run_sweep",
]
