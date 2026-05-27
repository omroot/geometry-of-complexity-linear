"""Portfolio return aggregation and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_portfolio_returns(port: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly portfolio returns from asset-level weights.

    Args:
        port: Asset-level DataFrame with columns ``yyyymm``, ``weight``,
            ``y_true`` (as produced by the builders in
            :mod:`src.portfolio.construct`).

    Returns:
        DataFrame with columns ``yyyymm`` and ``port_ret``, one row per
        month, sorted by ``yyyymm``.

    Raises:
        ValueError: If any of ``yyyymm`` / ``weight`` / ``y_true`` is missing.
    """
    required = {"yyyymm", "weight", "y_true"}
    missing = [c for c in required if c not in port.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    contributions = port["weight"] * port["y_true"]
    return (
        contributions.groupby(port["yyyymm"])
        .sum()
        .rename("port_ret")
        .reset_index()
        .sort_values("yyyymm")
        .reset_index(drop=True)
    )


def portfolio_performance(
    monthly_ret: pd.DataFrame, annual_factor: int = 12
) -> dict[str, float]:
    """Compute annualised return, vol, Sharpe, max drawdown, and hit rate.

    Args:
        monthly_ret: Output of :func:`compute_portfolio_returns`; must have
            a ``port_ret`` column.
        annual_factor: Periods per year used to annualise (``12`` for
            monthly, ``252`` for daily).

    Returns:
        Dict with keys ``ann_ret``, ``ann_vol``, ``sharpe``, ``max_dd``,
        ``hit_rate``. ``sharpe`` is ``float('nan')`` when ``ann_vol`` is
        zero.

    Raises:
        ValueError: If ``port_ret`` is missing, or if no finite returns
            remain after coercion.
    """
    required = {"port_ret"}
    missing = [c for c in required if c not in monthly_ret.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    r = pd.to_numeric(monthly_ret["port_ret"], errors="coerce").dropna()
    if r.empty:
        raise ValueError("No valid portfolio returns found.")

    ann_ret = r.mean() * annual_factor
    ann_vol = r.std(ddof=1) * np.sqrt(annual_factor)
    sharpe = float("nan") if ann_vol == 0 else ann_ret / ann_vol

    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    hit_rate = (r > 0).mean()

    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "hit_rate": float(hit_rate),
    }
