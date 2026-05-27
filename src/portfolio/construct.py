"""Portfolio construction from cross-sectional prediction signals.

Three builders, each taking a panel of (date, prediction, realized return)
and returning an asset-level DataFrame with weights:

* :func:`build_quantile_portfolios` — decile/quintile sort with long-short
  spread.
* :func:`build_long_only_portfolio` — top-N or threshold-filtered longs only.
* :func:`build_directional_portfolio` — long-short scaled by prediction
  magnitude.

Aggregate asset-level output into monthly returns via
:func:`src.portfolio.performance.compute_portfolio_returns`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_quantile_portfolios(
    data: pd.DataFrame,
    score_col: str = "forecast",
    ret_col: str = "excess_ret",
    date_col: str = "yyyymm",
    n_portfolios: int = 10,
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly quantile portfolios and the long-short spread.

    Args:
        data: Input panel containing at least date, score, and return columns.
        score_col: Column used to sort assets into portfolios.
        ret_col: Realised return column.
        date_col: Monthly date column; coercible to pandas datetime.
        n_portfolios: Number of portfolios (e.g., ``10`` for deciles). Must
            be at least 2.
        weight_col: Optional column for value-weighted portfolio returns.
            When omitted, returns are equal-weighted.

    Returns:
        Tuple ``(long_df, wide_df)``:

        * ``long_df`` — one row per ``(month, portfolio)`` with portfolio
          return.
        * ``wide_df`` — one row per month with columns ``P1 ... Pn`` and
          ``long_short = Pn - P1``.

    Raises:
        ValueError: If required columns are missing, ``n_portfolios < 2``,
            ``weight_col`` is missing from ``data``, no valid rows remain
            after coercion, or all weights are non-positive.
    """
    required = {date_col, score_col, ret_col}
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if n_portfolios < 2:
        raise ValueError("n_portfolios must be >= 2.")
    if weight_col is not None and weight_col not in data.columns:
        raise ValueError(f"Weight column '{weight_col}' not found.")

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    if weight_col is not None:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")

    keep = [date_col, score_col, ret_col] + ([weight_col] if weight_col else [])
    df = df[keep].dropna(subset=[date_col, score_col, ret_col])
    if df.empty:
        raise ValueError("No valid rows left after coercion/dropna.")

    pct_rank = df.groupby(date_col)[score_col].rank(method="first", pct=True)
    df["portfolio"] = np.ceil(pct_rank * n_portfolios).clip(1, n_portfolios).astype(int)

    if weight_col is None:
        long_df = (
            df.groupby([date_col, "portfolio"], as_index=False)[ret_col]
            .mean()
            .rename(columns={ret_col: "portfolio_return"})
        )
    else:
        tmp = df.dropna(subset=[weight_col]).copy()
        tmp = tmp[tmp[weight_col] > 0]
        if tmp.empty:
            raise ValueError("No positive weights available for portfolio construction.")
        # Weighted average = sum(w * r) / sum(w), per (month, portfolio).
        tmp = tmp.assign(_wr=tmp[weight_col] * tmp[ret_col])
        agg = tmp.groupby([date_col, "portfolio"], as_index=False).agg(
            _wr=("_wr", "sum"), _w=(weight_col, "sum"),
        )
        agg["portfolio_return"] = agg["_wr"] / agg["_w"]
        long_df = agg[[date_col, "portfolio", "portfolio_return"]]

    wide_df = long_df.pivot(index=date_col, columns="portfolio", values="portfolio_return").sort_index()
    wide_df = wide_df.rename(columns={k: f"P{k}" for k in wide_df.columns})
    wide_df["long_short"] = wide_df[f"P{n_portfolios}"] - wide_df["P1"]
    wide_df = wide_df.reset_index()

    return long_df.sort_values([date_col, "portfolio"]).reset_index(drop=True), wide_df


def build_long_only_portfolio(
    pred_df: pd.DataFrame,
    top_n: int | None = None,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Select stocks each month with positive predicted returns.

    Args:
        pred_df: Panel with at least ``yyyymm``, ``y_pred``, ``y_true``
            columns.
        top_n: When set, keep only the top ``N`` stocks by predicted return
            per month. Must be positive when supplied.
        threshold: Minimum predicted return required for inclusion.

    Returns:
        Asset-level DataFrame with an equal weight per month assigned to
        ``weight``. Columns: ``yyyymm``, ``y_pred``, ``y_true``, ``weight``.

    Raises:
        ValueError: If required columns are missing or ``top_n`` is non-positive.
    """
    required = {"yyyymm", "y_pred", "y_true"}
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be a positive integer when provided.")

    port = pred_df.copy()
    port["yyyymm"] = pd.to_datetime(port["yyyymm"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    port["y_pred"] = pd.to_numeric(port["y_pred"], errors="coerce")
    port["y_true"] = pd.to_numeric(port["y_true"], errors="coerce")
    port = port.dropna(subset=["yyyymm", "y_pred", "y_true"])
    port = port[port["y_pred"] > threshold].copy()

    if top_n is not None:
        port = (
            port.sort_values(["yyyymm", "y_pred"], ascending=[True, False])
            .groupby("yyyymm", as_index=False, group_keys=False)
            .head(top_n)
            .copy()
        )

    # Equal weight within each month.
    month_count = port.groupby("yyyymm")["y_pred"].transform("size")
    port["weight"] = 1.0 / month_count

    return port


def build_directional_portfolio(
    pred_df: pd.DataFrame,
    date_col: str = "yyyymm",
    pred_col: str = "y_pred",
    ret_col: str = "y_true",
    weighting: str = "prediction_scaled",
    normalize_weights: bool = True,
    gross_exposure: float = 1.0,
    drop_zero_preds: bool = True,
) -> pd.DataFrame:
    """Build a long-short portfolio from predicted returns.

    Args:
        pred_df: Input panel with at least date, prediction, and realised
            return columns.
        date_col: Monthly date column.
        pred_col: Predicted-return / signal column.
        ret_col: Realised-return column.
        weighting: ``"prediction_scaled"`` uses predicted returns as weights
            (positive predictions go long, negative go short).
            ``"equal_weight_sign"`` ignores magnitude and assigns each
            non-zero signal the same absolute weight.
        normalize_weights: When True and ``weighting='prediction_scaled'``,
            scale each month's weights so the sum of absolute weights
            equals ``gross_exposure``. When False, the asset weight is
            ``gross_exposure * y_pred``.
        gross_exposure: Target gross exposure when normalised. Must be
            positive.
        drop_zero_preds: When True, rows with zero predictions are
            excluded.

    Returns:
        Asset-level DataFrame with weights and return contributions.
        ``raw_signal_return_pct`` matches the ``y_pred * y_true * 100``
        style metric; ``contribution_pct`` is based on the portfolio weight
        actually used.

    Raises:
        ValueError: If required columns are missing, ``weighting`` is
            unknown, ``gross_exposure <= 0``, no valid rows remain after
            coercion, no non-zero predictions remain, or every month has
            zero gross signal under prediction-scaled weighting.
    """
    required = {date_col, pred_col, ret_col}
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if weighting not in {"prediction_scaled", "equal_weight_sign"}:
        raise ValueError(
            "weighting must be either 'prediction_scaled' or 'equal_weight_sign'."
        )
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive.")

    port = pred_df.copy()
    port[date_col] = pd.to_datetime(port[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    port[pred_col] = pd.to_numeric(port[pred_col], errors="coerce")
    port[ret_col] = pd.to_numeric(port[ret_col], errors="coerce")
    port = port.dropna(subset=[date_col, pred_col, ret_col]).copy()
    if port.empty:
        raise ValueError("No valid rows left after coercion/dropna.")

    port["position"] = np.sign(port[pred_col]).astype(float)
    if drop_zero_preds:
        port = port[port["position"] != 0].copy()
    if port.empty:
        raise ValueError("No non-zero predictions available for portfolio construction.")

    # Raw signal-return interaction, useful when you want y_pred * y_true * 100.
    port["raw_signal_return"] = port[pred_col] * port[ret_col]
    port["raw_signal_return_pct"] = 100.0 * port["raw_signal_return"]

    if weighting == "equal_weight_sign":
        n_active = port.groupby(date_col)["position"].transform("size")
        port["weight"] = gross_exposure * port["position"] / n_active
    else:
        if normalize_weights:
            port["gross_signal"] = port.groupby(date_col)[pred_col].transform(
                lambda s: np.abs(s).sum()
            )
            port = port[port["gross_signal"] > 0].copy()
            if port.empty:
                raise ValueError("All months have zero gross predicted signal.")
            port["weight"] = gross_exposure * port[pred_col] / port["gross_signal"]
        else:
            port["weight"] = gross_exposure * port[pred_col]

    port["contribution"] = port["weight"] * port[ret_col]
    port["contribution_pct"] = 100.0 * port["contribution"]

    return port.sort_values([date_col, pred_col], ascending=[True, False]).reset_index(drop=True)
