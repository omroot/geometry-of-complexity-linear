"""End-to-end characteristics-panel builder for the asset-pricing experiments.

Orchestrates the building blocks in :mod:`src.ingest.pipeline` plus the
Ken French risk-free rate from :mod:`src.ingest.fama_french` to produce
the parquet that the real-data experiments expect at
``data/preprocessed/char_data.parquet``.

Layout on disk:
    data/raw/          Raw upstream downloads (cached across runs).
        openap_raw.parquet
        crsp_sp500_raw.parquet
        rf_monthly.parquet
        .pgpass_runtime           # mode 600, materialized from .env
    data/preprocessed/
        char_data.parquet         # final modelling panel

Steps:
    1. Download OpenAP characteristics (cached at ``data/raw/openap_raw.parquet``)
    2. Download S&P 500 returns from WRDS (cached at ``data/raw/crsp_sp500_raw.parquet``)
    3. Download Ken French monthly risk-free rate (cached at ``data/raw/rf_monthly.parquet``)
    4. Preprocess the three raw inputs into the modelling panel via
       :func:`src.preprocess.build_char_panel` (merge, impute, prune)
    5. Write the final panel to ``data/preprocessed/char_data.parquet``

CLI:
    python -m src.ingest.build_panel --start 199001 --end 202512 \\
        --output data/preprocessed/char_data.parquet

Re-running is cheap: each download step skips when its raw parquet already
exists under ``data/raw/`` (use ``--force-refresh`` to invalidate).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingest.fama_french import load_risk_free_rate_cached
from src.ingest.pipeline import DataPipeline
from src.ingest.wrds_auth import have_wrds_credentials, materialize_wrds_pgpass
from src.preprocess import build_char_panel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "preprocessed" / "char_data.parquet"


def _download_openap_cached(
    pipeline: DataPipeline,
    cache_path: Path,
    start_yyyymm: str,
    end_yyyymm: str,
    force_refresh: bool,
) -> pd.DataFrame:
    """Cache-aware OpenAP signals download (whole universe, then date-filter)."""
    if cache_path.exists() and not force_refresh:
        print(f"  [cache hit] OpenAP signals -> {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print(f"  [download] OpenAP signals (this may take several minutes)...")
        t0 = time.monotonic()
        df = pipeline.download_openap_data(start_yyyymm, end_yyyymm)
        print(f"  [downloaded] {len(df):,} rows in {time.monotonic() - t0:.1f}s")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"  [cached] -> {cache_path}")

    df = df[
        (df[pipeline.time_col] >= pd.to_datetime(start_yyyymm, format="%Y%m"))
        & (df[pipeline.time_col] <= pd.to_datetime(end_yyyymm, format="%Y%m"))
    ].reset_index(drop=True)
    return df


def _download_crsp_cached(
    pipeline: DataPipeline,
    cache_path: Path,
    start_date: str,
    end_date: str,
    force_refresh: bool,
) -> pd.DataFrame:
    """Cache-aware CRSP S&P 500 returns download."""
    if cache_path.exists() and not force_refresh:
        print(f"  [cache hit] CRSP S&P 500 returns -> {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print(f"  [download] CRSP S&P 500 returns (requires WRDS credentials)...")
        t0 = time.monotonic()
        df = pipeline.download_sp500_returns_wrds(
            start_date=start_date, end_date=end_date
        )
        print(f"  [downloaded] {len(df):,} rows in {time.monotonic() - t0:.1f}s")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"  [cached] -> {cache_path}")

    df = df[
        (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ].reset_index(drop=True)
    return df


def _yyyymm_to_iso_date(yyyymm: str, day: str = "01") -> str:
    """Convert ``YYYYMM`` to ``YYYY-MM-DD``."""
    return f"{yyyymm[:4]}-{yyyymm[4:6]}-{day}"


def _ensure_wrds_credentials(raw_dir: Path) -> None:
    """Resolve WRDS credentials and set up ``PGPASSFILE`` accordingly.

    Tries (in order): ``WRDS_USERNAME``/``WRDS_PASSWORD`` env vars, the
    project-root ``.env`` file, then the user's existing ``~/.pgpass``.
    If a ``.env`` source is used, a runtime pgpass is materialized at
    ``raw_dir / ".pgpass_runtime"`` (mode 600) and ``PGPASSFILE`` is
    pointed at it. If none of these are present AND the OpenAP/CRSP raw
    parquets don't already exist, prints a setup hint and returns — the
    download will then fail with the usual interactive prompt.
    """
    have_openap_raw = (raw_dir / "openap_raw.parquet").exists()
    have_crsp_raw = (raw_dir / "crsp_sp500_raw.parquet").exists()
    if have_openap_raw and have_crsp_raw:
        return

    if materialize_wrds_pgpass(REPO_ROOT, raw_dir / ".pgpass_runtime"):
        source = have_wrds_credentials(REPO_ROOT)
        print(f"  [wrds] credentials loaded from {source!r} -> PGPASSFILE set")
        return

    if (Path.home() / ".pgpass").exists():
        return

    print(
        "WARNING: no WRDS credentials found. Set one of:\n"
        "  - WRDS_USERNAME and WRDS_PASSWORD env vars\n"
        f"  - WRDS_USERNAME and WRDS_PASSWORD in {REPO_ROOT / '.env'}\n"
        "  - ~/.pgpass (run from an interactive shell:\n"
        "      python -c 'import wrds; wrds.Connection()')\n"
        "Without one of these, the download will fail with EOFError.\n",
        file=sys.stderr,
        flush=True,
    )


def build_panel(
    start_yyyymm: str = "199001",
    end_yyyymm: str = "202512",
    output_path: Path = DEFAULT_OUTPUT_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    max_nan_frac: float = 0.6,
    min_std: float = 1e-6,
    max_corr: float = 0.98,
    force_refresh: bool = False,
    skip_clean: bool = False,
) -> pd.DataFrame:
    """Build the real-data characteristics panel end-to-end.

    Args:
        start_yyyymm: First month, ``YYYYMM``.
        end_yyyymm: Last month, ``YYYYMM``.
        output_path: Where to write the final parquet.
        raw_dir: Directory holding the raw upstream downloads (OpenAP,
            CRSP, monthly RF). Re-used across runs unless
            ``force_refresh=True``.
        max_nan_frac: NaN-fraction threshold passed to
            :func:`src.preprocess.remove_mostly_nan_columns`.
        min_std: Std threshold passed to
            :func:`src.preprocess.drop_low_std_and_high_corr`.
        max_corr: Pairwise-correlation threshold passed to the same.
        force_refresh: Invalidate every raw parquet and re-download from
            the upstream sources.
        skip_clean: If True, skip the std/correlation filtering step.
            Useful for diagnostics; not recommended for the final panel.

    Returns:
        The final DataFrame as written to ``output_path``.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_wrds_credentials(raw_dir)
    pipeline = DataPipeline()
    start_date = _yyyymm_to_iso_date(start_yyyymm)
    end_date = _yyyymm_to_iso_date(end_yyyymm, day="28")  # safe day-of-month

    print(f"\nBuild panel: {start_yyyymm} -> {end_yyyymm}")
    print(f"  raw:    {raw_dir}")
    print(f"  output: {output_path}\n")

    # --- 1. Download sources ---------------------------------------------
    print("Step 1: Download OpenAP signals")
    openap_df = _download_openap_cached(
        pipeline,
        raw_dir / "openap_raw.parquet",
        start_yyyymm,
        end_yyyymm,
        force_refresh,
    )
    print(f"  -> {len(openap_df):,} rows, {len(openap_df.columns):,} columns")

    print("\nStep 2: Download CRSP S&P 500 returns")
    crsp_df = _download_crsp_cached(
        pipeline,
        raw_dir / "crsp_sp500_raw.parquet",
        start_date,
        end_date,
        force_refresh,
    )
    print(f"  -> {len(crsp_df):,} rows")

    print("\nStep 3: Download Ken French risk-free rate")
    rf_df = load_risk_free_rate_cached(
        raw_dir / "rf_monthly.parquet",
        start_yyyymm=start_yyyymm,
        end_yyyymm=end_yyyymm,
        force_refresh=force_refresh,
    )
    print(f"  -> {len(rf_df):,} months")

    # --- 2. Preprocess: raw -> cleaned modelling panel -------------------
    print("\nStep 4: Preprocess raw inputs into modelling panel")
    merged = build_char_panel(
        openap_df,
        crsp_df,
        rf_df,
        max_nan_frac=max_nan_frac,
        min_std=min_std,
        max_corr=max_corr,
        skip_clean=skip_clean,
        verbose=True,
    )

    # --- 3. Write output --------------------------------------------------
    print(f"\nStep 5: Write {output_path}")
    merged.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"  -> wrote {len(merged):,} rows × {len(merged.columns):,} columns "
        f"({size_mb:.1f} MB)"
    )

    return merged


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the real-data characteristics panel for the asset-pricing experiments."
    )
    parser.add_argument("--start", default="199001", help="First month, YYYYMM. Default: 199001")
    parser.add_argument("--end", default="202512", help="Last month, YYYYMM. Default: 202512")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Raw upstream downloads directory. Default: {DEFAULT_RAW_DIR}",
    )
    parser.add_argument(
        "--max-nan-frac",
        type=float,
        default=0.6,
        help="Drop columns with NaN share above this threshold. Default: 0.6",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=1e-6,
        help="Drop characteristics with std below this threshold. Default: 1e-6",
    )
    parser.add_argument(
        "--max-corr",
        type=float,
        default=0.98,
        help="Drop one char from each pair above this |corr|. Default: 0.98",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Invalidate every raw parquet and re-download.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip the std/correlation filtering step.",
    )
    args = parser.parse_args(argv)

    build_panel(
        start_yyyymm=args.start,
        end_yyyymm=args.end,
        output_path=args.output,
        raw_dir=args.raw_dir,
        max_nan_frac=args.max_nan_frac,
        min_std=args.min_std,
        max_corr=args.max_corr,
        force_refresh=args.force_refresh,
        skip_clean=args.skip_clean,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
