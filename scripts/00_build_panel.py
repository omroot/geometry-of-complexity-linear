#!/usr/bin/env python3
"""Build the real-data characteristics panel and write ``data/preprocessed/char_data.parquet``.

Thin wrapper around :func:`src.ingest.build_panel.main`. End-to-end:

1. Download OpenAP signals (cached at ``data/raw/openap_raw.parquet``).
2. Download CRSP S&P 500 returns from WRDS, point-in-time, delisting-adjusted
   (cached at ``data/raw/crsp_sp500_raw.parquet``).
3. Download Ken French monthly risk-free rate
   (cached at ``data/raw/rf_monthly.parquet``).
4. Hand the three raw DataFrames to :func:`src.preprocess.build_char_panel`,
   which drops mostly-NaN columns, merges on ``(permno, yyyymm)``, computes
   ``excess_ret`` and ``mcap``, imputes remaining NaNs (time-safe), and
   prunes low-variance / highly-correlated characteristics.
5. Write ``data/preprocessed/char_data.parquet``.

Re-running is cheap: each download step skips when its raw parquet
already exists under ``data/raw/``. Pass ``--force-refresh`` to
invalidate every raw download.

WRDS credentials must be available via one of:

* ``WRDS_USERNAME`` / ``WRDS_PASSWORD`` environment variables,
* a project-root ``.env`` file with the same keys, or
* an existing ``~/.pgpass`` entry for ``wrds-pgdata.wharton.upenn.edu``.

Usage:
    python scripts/00_build_panel.py
    python scripts/00_build_panel.py --start 199001 --end 202512
    python scripts/00_build_panel.py --force-refresh
    python scripts/00_build_panel.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingest.build_panel import main


if __name__ == "__main__":
    sys.exit(main())
