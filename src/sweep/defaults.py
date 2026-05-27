"""Shared constants for the sweep scripts under ``scripts/experiments/``.

Two kinds of constants live here:

* Paper §8.1 DGP defaults consumed by every synthetic sweep
  (:data:`PAPER_DGP_DEFAULTS` and the ``k0=18`` short-window variant).
* Common rolling-fit knobs (:data:`TRAIN_WINDOW_MONTHS`,
  :data:`RFF_GAMMA`, :data:`FORECAST_START`) and reusable
  RFF-dim / seed grids used by the §8.5 + §9.10 broad backtests.
"""

from __future__ import annotations


# Paper §8.1 DGP defaults consumed by every synthetic sweep.
PAPER_DGP_DEFAULTS: dict = {
    "T": 360, "N": 50, "m": 30, "k0": 12, "seed": 42,
    "snr_base": 5.0, "eigenvalue_decay": 0.82, "sigma_eps": 1.0,
}

# Paper §8.5 short-window variant: same DGP with k0 raised to 18.
PAPER_DGP_SHORT_WINDOW: dict = {**PAPER_DGP_DEFAULTS, "k0": 18}


# Rolling-fit knobs shared across sweeps.
TRAIN_WINDOW_MONTHS: int = 24
RFF_GAMMA: float = 0.25
FORECAST_START: str = "1996-01-01"  # First OOS month after a 24m warmup from 1994-01.


# Reusable RFF + seed grids for §8.5 / §9.10 broad backtests.
BASELINE_RFF: int = 0  # Linear baseline (no RFF lift).
SMALL_RFF: tuple[int, ...] = (24, 32, 48, 64, 128, 256)
LARGE_RFF: tuple[int, ...] = (512, 1024, 2048, 4096)
SEEDS_SMALL: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
SEEDS_LARGE: tuple[int, ...] = (10, 20, 30)
