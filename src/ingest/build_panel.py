"""End-to-end characteristics-panel builder for the asset-pricing experiments.

Orchestrates the building blocks in :mod:`src.ingest.pipeline` plus the
Ken French risk-free rate from :mod:`src.ingest.fama_french` to produce
the parquet that the real-data experiments expect at
``data/char_data.parquet``.

Steps:
    1. Download OpenAP characteristics (cached: ``cache/openap_raw.parquet``)
    2. Download S&P 500 returns from WRDS (cached: ``cache/crsp_sp500_raw.parquet``)
    3. Download Ken French monthly risk-free rate (cached: ``cache/rf_monthly.parquet``)
    4. Drop characteristics with too many NaNs
    5. Merge OpenAP × CRSP on ``(permno, yyyymm)``, adding ``mcap = abs(prc) * shrout``
    6. Merge in ``rf`` and compute ``excess_ret = ret_adj - rf``
    7. Impute remaining NaNs (cross-sectional median, time-safe)
    8. Drop low-std / highly-correlated characteristics
    9. Write the final panel to ``data/char_data.parquet``

CLI:
    python -m src.ingest.build_panel --start 199001 --end 202512 \\
        --output data/char_data.parquet

Re-running is cheap: each download step skips when its cache parquet
exists (use ``--force-refresh`` to invalidate caches).
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "cache" / "ingest"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "char_data.parquet"


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


def _merge_risk_free_rate(
    panel: pd.DataFrame,
    rf_df: pd.DataFrame,
    return_col: str,
    excess_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Left-join the monthly risk-free rate and compute ``excess_ret``."""
    rf_df = rf_df.rename(columns={"yyyymm": time_col})
    rf_df[time_col] = pd.to_datetime(rf_df[time_col]).dt.to_period(
        "M"
    ).dt.to_timestamp()
    merged = panel.merge(rf_df, on=time_col, how="left", validate="m:1")
    merged[excess_col] = merged[return_col] - merged["rf"]
    return merged


def _yyyymm_to_iso_date(yyyymm: str, day: str = "01") -> str:
    """Convert ``YYYYMM`` to ``YYYY-MM-DD``."""
    return f"{yyyymm[:4]}-{yyyymm[4:6]}-{day}"


def _ensure_wrds_credentials(cache_dir: Path) -> None:
    """Resolve WRDS credentials and set up ``PGPASSFILE`` accordingly.

    Tries (in order): ``WRDS_USERNAME``/``WRDS_PASSWORD`` env vars, the
    project-root ``.env`` file, then the user's existing ``~/.pgpass``.
    If a ``.env`` source is used, a runtime pgpass is materialized at
    ``cache_dir / ".pgpass_runtime"`` (mode 600) and ``PGPASSFILE`` is
    pointed at it. If none of these are present AND the OpenAP/CRSP
    caches don't already exist, prints a setup hint and returns — the
    download will then fail with the usual interactive prompt.
    """
    have_openap_cache = (cache_dir / "openap_raw.parquet").exists()
    have_crsp_cache = (cache_dir / "crsp_sp500_raw.parquet").exists()
    if have_openap_cache and have_crsp_cache:
        return

    if materialize_wrds_pgpass(REPO_ROOT, cache_dir / ".pgpass_runtime"):
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
    cache_dir: Path = DEFAULT_CACHE_DIR,
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
        cache_dir: Directory for intermediate caches (raw OpenAP, raw
            CRSP, monthly RF). Re-used across runs unless
            ``force_refresh=True``.
        max_nan_frac: NaN-fraction threshold for column drops in
            :meth:`DataPipeline.remove_mostly_nan_columns`.
        min_std: Std threshold for low-variance char drops in
            :meth:`DataPipeline.drop_low_std_and_high_corr`.
        max_corr: Pairwise-correlation threshold for the same.
        force_refresh: Invalidate every intermediate cache and re-download
            from the upstream sources.
        skip_clean: If True, skip the std/correlation filtering step.
            Useful for diagnostics; not recommended for the final panel.

    Returns:
        The final DataFrame as written to ``output_path``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_wrds_credentials(cache_dir)
    pipeline = DataPipeline()
    start_date = _yyyymm_to_iso_date(start_yyyymm)
    end_date = _yyyymm_to_iso_date(end_yyyymm, day="28")  # safe day-of-month

    print(f"\nBuild panel: {start_yyyymm} -> {end_yyyymm}")
    print(f"  cache: {cache_dir}")
    print(f"  output: {output_path}\n")

    # --- 1. Download sources ---------------------------------------------
    print("Step 1: Download OpenAP signals")
    openap_df = _download_openap_cached(
        pipeline,
        cache_dir / "openap_raw.parquet",
        start_yyyymm,
        end_yyyymm,
        force_refresh,
    )
    print(f"  -> {len(openap_df):,} rows, {len(openap_df.columns):,} columns")

    print("\nStep 2: Download CRSP S&P 500 returns")
    crsp_df = _download_crsp_cached(
        pipeline,
        cache_dir / "crsp_sp500_raw.parquet",
        start_date,
        end_date,
        force_refresh,
    )
    print(f"  -> {len(crsp_df):,} rows")

    print("\nStep 3: Download Ken French risk-free rate")
    rf_df = load_risk_free_rate_cached(
        cache_dir / "rf_monthly.parquet",
        start_yyyymm=start_yyyymm,
        end_yyyymm=end_yyyymm,
        force_refresh=force_refresh,
    )
    print(f"  -> {len(rf_df):,} months")

    # --- 2. Clean characteristics ----------------------------------------
    print("\nStep 4: Drop OpenAP columns with too many NaNs "
          f"(threshold = {max_nan_frac:.0%})")
    openap_df, dropped_nan_cols = pipeline.remove_mostly_nan_columns(
        openap_df, max_nan_frac=max_nan_frac
    )
    print(
        f"  -> dropped {len(dropped_nan_cols)} columns "
        f"({len(openap_df.columns):,} remain)"
    )

    # --- 3. Merge OpenAP × CRSP × RF -------------------------------------
    print("\nStep 5: Merge OpenAP × CRSP (inner join on permno, yyyymm)")
    merged = pipeline.merge_openap_with_crsp_returns(
        openap_df,
        crsp_df,
        crsp_return_col="ret_adj",
        output_return_col="ret_adj",
        how="inner",
        add_market_cap=True,
    )
    print(
        f"  -> {len(merged):,} rows, "
        f"{merged['permno'].nunique():,} permnos, "
        f"{merged['yyyymm'].nunique():,} months"
    )

    print("\nStep 6: Merge in monthly risk-free rate and compute excess_ret")
    merged = _merge_risk_free_rate(
        merged,
        rf_df,
        return_col="ret_adj",
        excess_col="excess_ret",
        time_col=pipeline.time_col,
    )
    n_missing_rf = merged["rf"].isna().sum()
    if n_missing_rf:
        print(f"  WARNING: {n_missing_rf:,} rows have no RF match")
    print(
        f"  -> excess_ret range: "
        f"[{merged['excess_ret'].min():.4f}, {merged['excess_ret'].max():.4f}]"
    )

    # --- 4. Fill remaining NaNs and prune low-info chars -----------------
    print("\nStep 7: Impute remaining NaNs (cross-sectional median, time-safe)")
    merged = pipeline.fill_remaining_missing(merged, use_past_only=True)

    if not skip_clean:
        print(
            f"\nStep 8: Drop low-std (<{min_std}) and "
            f"highly-correlated (|r|>{max_corr}) characteristics"
        )
        merged, kept_chars, dropped_chars = pipeline.drop_low_std_and_high_corr(
            merged, min_std=min_std, max_corr=max_corr
        )
        print(
            f"  -> dropped {len(dropped_chars)} chars "
            f"({len(kept_chars)} characteristics remain)"
        )

    # --- 5. Write output --------------------------------------------------
    print(f"\nStep 9: Write {output_path}")
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
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Intermediate cache directory. Default: {DEFAULT_CACHE_DIR}",
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
        help="Invalidate every intermediate cache and re-download.",
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
        cache_dir=args.cache_dir,
        max_nan_frac=args.max_nan_frac,
        min_std=args.min_std,
        max_corr=args.max_corr,
        force_refresh=args.force_refresh,
        skip_clean=args.skip_clean,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
