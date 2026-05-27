#!/usr/bin/env python3
"""Load the LinearCase real-data parquet and print a sanity summary.

Expects ``data/preprocessed/char_data.parquet`` (produced by
``scripts/00_build_panel.py``): a long-form panel of OpenAP
characteristics merged with CRSP excess returns. The shared
:func:`src.sweep.load_real_panel` loader handles the optional
``Size``->``mcap`` alias.

Outputs:
    results/data/real_panel_preview.csv -- first 10k rows sorted by
        (yyyymm, permno) for quick inspection.

Usage:
    python scripts/experiments/real_data/01_load_panel.py
    CHAR_DATA_PARQUET=/abs/path/to/char_data.parquet python scripts/experiments/real_data/01_load_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sweep import load_real_panel


OUT_PATH = REPO_ROOT / "results" / "data" / "real_panel_preview.csv"

# Columns that are NOT characteristics.
_NON_CHAR = {
    "permno", "yyyymm", "excess_ret", "ret_adj", "ret",
    "y_ipca", "Size", "mcap", "Price",
    "prc", "shrout", "cfacpr", "cfacshr", "rf",
}


def main() -> None:
    panel = load_real_panel()

    char_cols = [c for c in panel.columns if c not in _NON_CHAR]

    print(f"\nPanel shape: {panel.shape}")
    print(f"  permnos: {panel['permno'].nunique()}")
    print(f"  months:  {panel['yyyymm'].nunique()}")
    print(f"  range:   {pd.to_datetime(panel['yyyymm']).min().date()} .. "
          f"{pd.to_datetime(panel['yyyymm']).max().date()}")
    print(f"  characteristics: {len(char_cols)}")
    for col in ("excess_ret", "y_ipca", "mcap"):
        if col in panel.columns:
            n = int(panel[col].notna().sum())
            print(f"  {col}: {n:,} non-null ({100 * n / len(panel):.1f}%)")
        else:
            print(f"  {col}: MISSING")

    print(f"\nFirst 5 characteristics: {char_cols[:5]}")
    print(f"Last 5 characteristics:  {char_cols[-5:]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.sort_values(["yyyymm", "permno"]).head(10000).to_csv(OUT_PATH, index=False)
    print(f"\nWrote preview (first 10k rows) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
