"""Transform raw OpenAP / CRSP / Fama-French inputs into the modelling panel.

This module is the *preprocessing* half of the real-data pipeline. Its
counterpart is :mod:`src.ingest`, which downloads the raw inputs and
writes them to ``data/raw/``. Preprocessing is pure: it takes DataFrames
and returns DataFrames — no IO.

Top-level entry point:

* :func:`build_char_panel` — end-to-end raw → cleaned, merged, imputed,
  low-info-pruned modelling panel.

Lower-level building blocks (exposed for callers needing finer control or
one-off use of an individual step):

* :func:`remove_mostly_nan_columns`
* :func:`merge_openap_with_crsp`
* :func:`merge_risk_free_rate`
* :func:`fill_remaining_missing`
* :func:`drop_low_std_and_high_corr`
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


ID_COL: str = "permno"
TIME_COL: str = "yyyymm"
RETURN_COL: str = "excess_ret"

# Columns never dropped by the NaN / std / correlation filters.
_PROTECTED_COLS: tuple[str, ...] = (
    ID_COL, TIME_COL,
    "ret", "ret_adj", "excess_ret", "year",
    "prc", "shrout", "cfacpr", "cfacshr", "mcap",
)

# Same as above plus model-panel internals (used inside :func:`drop_low_std_and_high_corr`).
_PROTECTED_COLS_MODEL: tuple[str, ...] = _PROTECTED_COLS + ("y_ipca", "i_idx", "t_idx")


def remove_mostly_nan_columns(
    df: pd.DataFrame,
    max_nan_frac: float = 0.6,
    protected_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns whose NaN share exceeds ``max_nan_frac``.

    Args:
        df: Input DataFrame.
        max_nan_frac: NaN-share threshold in ``[0, 1]``.
        protected_cols: Columns never dropped. Defaults to the module-wide
            protected set (id/time/return + CRSP raw fields).

    Returns:
        Tuple ``(filtered_df, dropped_cols)``.

    Raises:
        ValueError: If ``max_nan_frac`` is outside ``[0, 1]``.
    """
    if not 0.0 <= max_nan_frac <= 1.0:
        raise ValueError("max_nan_frac must be between 0 and 1.")
    protected = set(protected_cols if protected_cols is not None else _PROTECTED_COLS)
    nan_fracs = df.isna().mean()
    dropped = [c for c in df.columns if c not in protected and nan_fracs[c] > max_nan_frac]
    keep = [c for c in df.columns if c not in set(dropped)]
    return df[keep].copy(), dropped


def merge_openap_with_crsp(
    openap_df: pd.DataFrame,
    crsp_df: pd.DataFrame,
    crsp_return_col: str = "ret_adj",
    output_return_col: str = "ret_adj",
    how: str = "inner",
    keep_crsp_columns: Optional[List[str]] = None,
    add_market_cap: bool = True,
) -> pd.DataFrame:
    """Merge OpenAP characteristics with CRSP returns on ``(permno, yyyymm)``.

    Args:
        openap_df: OpenAP panel with ``permno`` and ``yyyymm`` columns.
        crsp_df: CRSP S&P 500 returns (output of
            :meth:`src.ingest.pipeline.DataPipeline.download_sp500_returns_wrds`).
        crsp_return_col: Source CRSP return column (e.g. ``"ret_adj"``).
        output_return_col: Column name in the merged panel for the return.
        how: Merge mode (``"inner"`` or ``"left"``).
        keep_crsp_columns: Extra CRSP columns to retain. Defaults to
            ``["prc", "shrout", "cfacpr", "cfacshr"]`` when available.
        add_market_cap: When True and ``prc`` / ``shrout`` are present,
            adds ``mcap = abs(prc) * shrout`` to the merged panel.

    Returns:
        Merged panel sorted by ``(permno, yyyymm)``.

    Raises:
        ValueError: If ``crsp_return_col`` is missing from ``crsp_df``, or
            if CRSP lacks a usable time column.
    """
    if crsp_return_col not in crsp_df.columns:
        raise ValueError(
            f"CRSP return column '{crsp_return_col}' not found. "
            f"Available: {list(crsp_df.columns)}"
        )

    left = openap_df.copy()
    right = crsp_df.copy()

    left[ID_COL] = pd.to_numeric(left[ID_COL], errors="coerce")
    right[ID_COL] = pd.to_numeric(right[ID_COL], errors="coerce")

    # Normalise both sides to month-start timestamps so the merge keys match.
    left[TIME_COL] = pd.to_datetime(left[TIME_COL], errors="coerce").dt.to_period(
        "M"
    ).dt.to_timestamp()

    if TIME_COL in right.columns:
        right[TIME_COL] = pd.to_datetime(right[TIME_COL], errors="coerce").dt.to_period(
            "M"
        ).dt.to_timestamp()
    elif "date" in right.columns:
        right[TIME_COL] = pd.to_datetime(right["date"], errors="coerce").dt.to_period(
            "M"
        ).dt.to_timestamp()
    else:
        raise ValueError(f"CRSP data must include '{TIME_COL}' or 'date' column.")

    crsp_keep = [ID_COL, TIME_COL, crsp_return_col]
    if keep_crsp_columns is None:
        keep_crsp_columns = ["prc", "shrout", "cfacpr", "cfacshr"]
    for col in keep_crsp_columns:
        if col in right.columns and col not in crsp_keep:
            crsp_keep.append(col)

    replace_cols = [output_return_col] + [c for c in keep_crsp_columns if c in left.columns]
    if add_market_cap and "mcap" in left.columns:
        replace_cols.append("mcap")
    replace_cols = [c for c in replace_cols if c in left.columns]
    if replace_cols:
        left = left.drop(columns=replace_cols)

    right = right[crsp_keep].rename(columns={crsp_return_col: output_return_col})
    merged = left.merge(right, on=[ID_COL, TIME_COL], how=how, validate="m:1")

    if output_return_col in merged.columns:
        merged[output_return_col] = pd.to_numeric(merged[output_return_col], errors="coerce")
    for col in ["prc", "shrout", "cfacpr", "cfacshr"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    if add_market_cap and {"prc", "shrout"}.issubset(merged.columns):
        merged["mcap"] = merged["prc"].abs() * merged["shrout"]

    return merged.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)


def merge_risk_free_rate(
    panel: pd.DataFrame,
    rf_df: pd.DataFrame,
    return_col: str = "ret_adj",
    excess_col: str = RETURN_COL,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Left-join the monthly risk-free rate and compute the excess return.

    Args:
        panel: Merged panel containing the raw return column ``return_col``.
        rf_df: Output of :func:`src.ingest.fama_french.load_risk_free_rate_cached`.
            Expects ``yyyymm`` and ``rf`` columns.
        return_col: Raw return column on ``panel``.
        excess_col: Output column name (``panel[return_col] - rf``).
        time_col: Name of the month column on ``panel``.

    Returns:
        Panel with ``rf`` and ``excess_col`` columns added.
    """
    rf_df = rf_df.rename(columns={"yyyymm": time_col})
    rf_df[time_col] = pd.to_datetime(rf_df[time_col]).dt.to_period(
        "M"
    ).dt.to_timestamp()
    merged = panel.merge(rf_df, on=time_col, how="left", validate="m:1")
    merged[excess_col] = merged[return_col] - merged["rf"]
    return merged


def fill_remaining_missing(
    df: pd.DataFrame,
    protected_cols: Optional[List[str]] = None,
    use_past_only: bool = False,
) -> pd.DataFrame:
    """Fill remaining numeric NaNs by same-month median, then a fallback.

    When ``use_past_only`` is False (default), the fallback is the
    full-sample global median for each column. When True, the fallback is
    a time-safe historical median built only from strictly earlier months
    — useful when imputation downstream of train/test splits must avoid
    look-ahead.

    Args:
        df: Input DataFrame containing the month column ``yyyymm``.
        protected_cols: Columns excluded from imputation. Defaults to the
            module-wide protected set.
        use_past_only: Time-safe mode (see above).

    Returns:
        Imputed DataFrame.
    """
    if protected_cols is None:
        protected_cols = list(_PROTECTED_COLS)

    out = df.copy()
    num_cols = [
        c for c in out.columns
        if c not in protected_cols and pd.api.types.is_numeric_dtype(out[c])
    ]
    if not num_cols:
        return out

    if not use_past_only:
        global_medians = out[num_cols].median()
        out[num_cols] = out.groupby(TIME_COL)[num_cols].transform(
            lambda s: s.fillna(s.median())
        )
        out[num_cols] = out[num_cols].fillna(global_medians)
        return out

    # Time-safe mode:
    #   1) fill by same-month cross-sectional median
    #   2) remaining NaNs use median from strictly earlier months only
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce").dt.to_period(
        "M"
    ).dt.to_timestamp()
    original_index = out.index
    sort_cols = [TIME_COL]
    if ID_COL in out.columns:
        sort_cols.append(ID_COL)
    out = out.sort_values(sort_cols).copy()

    out[num_cols] = out.groupby(TIME_COL)[num_cols].transform(
        lambda s: s.fillna(s.median())
    )

    monthly_medians = (
        out.groupby(TIME_COL, sort=True)[num_cols]
        .median()
        .sort_index()
    )
    historical_medians = monthly_medians.expanding(min_periods=1).median().shift(1)

    hist_lookup = historical_medians.add_suffix("__hist")
    out = out.join(hist_lookup, on=TIME_COL)
    for col in num_cols:
        hist_col = f"{col}__hist"
        out[col] = out[col].fillna(out[hist_col])
    out = out.drop(columns=list(hist_lookup.columns))

    return out.loc[original_index]


def drop_low_std_and_high_corr(
    df: pd.DataFrame,
    char_cols: Optional[List[str]] = None,
    min_std: float = 1e-6,
    max_corr: float = 0.98,
    protected_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Drop very-low-std and very-high-correlation characteristics.

    Steps:
        1. Remove characteristics with ``std <= min_std`` (or NaN std).
        2. On the remaining set, greedily drop one variable from each pair
           whose ``|correlation| > max_corr``.

    Args:
        df: Input DataFrame.
        char_cols: Candidate characteristic columns to consider. When None,
            uses every numeric column not in ``protected_cols``.
        min_std: Minimum-std threshold. Must be non-negative.
        max_corr: Maximum absolute pairwise correlation. Must be in
            ``[0, 1]``.
        protected_cols: Columns never dropped. Defaults to the module-wide
            protected set extended with the model-panel internals
            (``y_ipca``, ``i_idx``, ``t_idx``).

    Returns:
        Tuple ``(filtered_df, kept_char_cols, dropped_char_cols)``.

    Raises:
        ValueError: If ``min_std < 0`` or ``max_corr`` is outside ``[0, 1]``.
    """
    if min_std < 0:
        raise ValueError("min_std must be non-negative.")
    if not 0.0 <= max_corr <= 1.0:
        raise ValueError("max_corr must be between 0 and 1.")

    out = df.copy()
    if protected_cols is None:
        protected_cols = list(_PROTECTED_COLS_MODEL)

    if char_cols is None:
        candidate_cols = [
            c for c in out.columns
            if c not in protected_cols and pd.api.types.is_numeric_dtype(out[c])
        ]
    else:
        candidate_cols = [
            c for c in char_cols
            if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
        ]

    if not candidate_cols:
        return out, [], []

    std = out[candidate_cols].std(axis=0, ddof=0)
    low_std_cols = std[(std <= min_std) | (~np.isfinite(std))].index.tolist()
    remaining = [c for c in candidate_cols if c not in low_std_cols]

    dropped_high_corr: List[str] = []
    if remaining:
        corr = out[remaining].corr().abs()
        mean_corr = corr.mean(axis=0).fillna(0.0)
        nan_frac = out[remaining].isna().mean(axis=0)

        work_cols = remaining.copy()
        while len(work_cols) >= 2:
            corr_sub = corr.loc[work_cols, work_cols]
            upper = corr_sub.where(np.triu(np.ones(corr_sub.shape), k=1).astype(bool))
            stacked = upper.stack()
            if stacked.empty:
                break
            most_corr = stacked.max()
            if not np.isfinite(most_corr) or most_corr <= max_corr:
                break

            c1, c2 = stacked.idxmax()
            if nan_frac[c1] > nan_frac[c2]:
                drop_col = c1
            elif nan_frac[c2] > nan_frac[c1]:
                drop_col = c2
            elif mean_corr[c1] >= mean_corr[c2]:
                drop_col = c1
            else:
                drop_col = c2

            dropped_high_corr.append(drop_col)
            work_cols.remove(drop_col)

        remaining = work_cols

    dropped_char_cols = low_std_cols + dropped_high_corr
    keep_set = set(remaining)
    non_char_cols = [c for c in out.columns if c not in candidate_cols]
    final_cols = non_char_cols + [c for c in candidate_cols if c in keep_set]
    return out[final_cols].copy(), remaining, dropped_char_cols


def build_char_panel(
    openap_df: pd.DataFrame,
    crsp_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    *,
    max_nan_frac: float = 0.6,
    min_std: float = 1e-6,
    max_corr: float = 0.98,
    skip_clean: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """End-to-end: raw inputs → cleaned, merged, low-info-pruned char panel.

    Steps:
        1. Drop OpenAP columns with NaN share above ``max_nan_frac``.
        2. Inner-join OpenAP × CRSP on ``(permno, yyyymm)`` and add
           ``mcap = abs(prc) * shrout``.
        3. Left-join the monthly RF and compute
           ``excess_ret = ret_adj - rf``.
        4. Time-safe cross-sectional median imputation of remaining NaNs.
        5. Optionally drop low-std (``< min_std``) and highly correlated
           (``|r| > max_corr``) characteristics.

    Args:
        openap_df: Raw OpenAP signal panel.
        crsp_df: Raw CRSP S&P 500 returns with delisting adjustment.
        rf_df: Monthly risk-free rate (Ken French).
        max_nan_frac: NaN-share threshold for column drops in step 1.
        min_std: Low-variance drop threshold in step 5.
        max_corr: High-correlation drop threshold in step 5.
        skip_clean: When True, skip step 5. Useful for diagnostics; not
            recommended for the final modelling panel.
        verbose: When True, print a one-line ``[step]`` summary for each
            stage.

    Returns:
        The preprocessed panel — long-form, sorted by ``(permno, yyyymm)``.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Drop high-NaN OpenAP columns.
    openap_df, dropped_nan = remove_mostly_nan_columns(openap_df, max_nan_frac=max_nan_frac)
    log(
        f"  [nan-prune] dropped {len(dropped_nan)} cols "
        f"(threshold = {max_nan_frac:.0%}; {len(openap_df.columns):,} remain)"
    )

    # 2. OpenAP × CRSP merge.
    merged = merge_openap_with_crsp(
        openap_df, crsp_df,
        crsp_return_col="ret_adj", output_return_col="ret_adj",
        how="inner", add_market_cap=True,
    )
    log(
        f"  [merge]     OpenAP × CRSP -> {len(merged):,} rows, "
        f"{merged[ID_COL].nunique():,} permnos, "
        f"{merged[TIME_COL].nunique():,} months"
    )

    # 3. RF merge + excess return.
    merged = merge_risk_free_rate(merged, rf_df, return_col="ret_adj", excess_col=RETURN_COL)
    n_missing_rf = int(merged["rf"].isna().sum())
    if n_missing_rf and verbose:
        print(f"  [merge]     WARNING: {n_missing_rf:,} rows have no RF match")
    log(
        f"  [merge]     RF -> excess_ret range "
        f"[{merged[RETURN_COL].min():.4f}, {merged[RETURN_COL].max():.4f}]"
    )

    # 4. NaN imputation (time-safe).
    merged = fill_remaining_missing(merged, use_past_only=True)
    log("  [impute]    cross-sectional median (time-safe)")

    # 5. Optional low-info pruning.
    if not skip_clean:
        merged, kept_chars, dropped_chars = drop_low_std_and_high_corr(
            merged, min_std=min_std, max_corr=max_corr,
        )
        log(
            f"  [prune]     dropped {len(dropped_chars)} low-info chars "
            f"(std<{min_std}, |r|>{max_corr}; {len(kept_chars)} chars remain)"
        )

    return merged
