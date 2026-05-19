"""Ken French risk-free rate downloader.

Fetches the monthly 1-month T-bill rate from the Fama/French data library:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

The CSV is bundled inside a zip and has columns
``Date, Mkt-RF, SMB, HML, RF`` (returns in *percent*; we convert to decimals
so it composes cleanly with CRSP returns).

No new dependencies — uses stdlib :mod:`urllib` and :mod:`zipfile`.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

KEN_FRENCH_3FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)


def download_risk_free_rate(
    start_yyyymm: str | None = None,
    end_yyyymm: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download Ken French's monthly risk-free rate.

    Args:
        start_yyyymm: Optional first month to keep, ``YYYYMM``.
        end_yyyymm: Optional last month to keep, ``YYYYMM``.
        timeout: HTTP timeout in seconds.

    Returns:
        DataFrame with columns ``yyyymm`` (month-start timestamp) and
        ``rf`` (monthly risk-free return as a decimal, e.g., 0.0035 for
        35 bps).
    """
    request = urllib.request.Request(
        KEN_FRENCH_3FACTORS_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ingest pipeline)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(csv_name) as fh:
            text = fh.read().decode("latin-1")

    # The Ken French CSV has a free-text header, then a monthly block, then
    # an annual block at the bottom (separated by a blank line). Parse only
    # the monthly section: lines whose first token is exactly 6 digits.
    monthly_rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and len(parts[0]) == 6 and parts[0].isdigit():
            monthly_rows.append(parts[:5])

    if not monthly_rows:
        raise RuntimeError(
            "Ken French CSV produced no monthly rows — upstream format may have changed."
        )

    df = pd.DataFrame(
        monthly_rows, columns=["yyyymm_str", "mkt_rf", "smb", "hml", "rf"]
    )
    df["yyyymm"] = pd.to_datetime(df["yyyymm_str"], format="%Y%m")
    df["rf"] = pd.to_numeric(df["rf"], errors="coerce") / 100.0  # percent -> decimal
    df = df[["yyyymm", "rf"]].dropna(subset=["rf"]).reset_index(drop=True)

    if start_yyyymm is not None:
        df = df[df["yyyymm"] >= pd.to_datetime(start_yyyymm, format="%Y%m")]
    if end_yyyymm is not None:
        df = df[df["yyyymm"] <= pd.to_datetime(end_yyyymm, format="%Y%m")]

    return df.reset_index(drop=True)


def load_risk_free_rate_cached(
    cache_path: Path,
    start_yyyymm: str | None = None,
    end_yyyymm: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Cache-aware wrapper around :func:`download_risk_free_rate`.

    Args:
        cache_path: Local parquet cache. Re-used if it exists and
            ``force_refresh`` is False.
        start_yyyymm: Optional first month, applied after caching so the
            cache is universe-wide.
        end_yyyymm: Optional last month, applied after caching.
        force_refresh: Re-download even if cache exists.

    Returns:
        Same shape as :func:`download_risk_free_rate`.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
    else:
        df = download_risk_free_rate()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    if start_yyyymm is not None:
        df = df[df["yyyymm"] >= pd.to_datetime(start_yyyymm, format="%Y%m")]
    if end_yyyymm is not None:
        df = df[df["yyyymm"] <= pd.to_datetime(end_yyyymm, format="%Y%m")]
    return df.reset_index(drop=True)
