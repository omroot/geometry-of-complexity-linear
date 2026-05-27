#!/usr/bin/env python3
"""Generate the paper §8.1 synthetic panel and print a sanity summary.

Quick check that the unified DGP in :mod:`src.generate.synthetic` is wired
correctly. Prints the signal spectrum and saves the long-form workflow
panel for downstream sweeps to consume.

Usage:
    python scripts/experiments/synthetic/01_smoke_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate import generate_workflow_panel


# Paper §8.1 defaults.
T: int = 360
N: int = 50
M: int = 30
K0: int = 12
SEED: int = 42
SNR_BASE: float = 5.0
EIGENVALUE_DECAY: float = 0.82
SIGMA_EPS: float = 1.0

OUT_PATH = REPO_ROOT / "results" / "data" / "synthetic_panel.csv"


def main() -> None:
    print(f"Generating §8.1 panel: T={T}, N={N}, m={M}, k0={K0}, seed={SEED}")
    panel, truth = generate_workflow_panel(
        T=T, N=N, m=M, k0=K0, seed=SEED,
        snr_base=SNR_BASE, eigenvalue_decay=EIGENVALUE_DECAY, sigma_eps=SIGMA_EPS,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PATH, index=False)

    print(f"\nPanel shape: {panel.shape}")
    print(f"  assets: {panel['permno'].nunique()}, months: {panel['yyyymm'].nunique()}")
    print(f"  range:  {panel['yyyymm'].min().date()} .. {panel['yyyymm'].max().date()}")
    print(f"\nTrue signal rank k0: {truth['true_k0']}")
    print(f"Calibrated w_drift_scale: {truth['w_drift_scale']:.6f}")
    print(
        "\nSignal spectrum lambda_j = snr_base * decay^(j-1) + sigma_eps^2 "
        f"(snr_base={SNR_BASE}, decay={EIGENVALUE_DECAY}):"
    )
    for j in range(1, K0 + 1):
        lam = SNR_BASE * EIGENVALUE_DECAY ** (j - 1) + SIGMA_EPS ** 2
        print(f"  lambda_{j:2d} = {lam:.3f}")
    print(f"\nSaved panel -> {OUT_PATH}")


if __name__ == "__main__":
    main()
