"""Portfolio return aggregation and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_portfolio_returns(port: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly portfolio returns from asset-level weights.

    Expects an asset-level DataFrame with columns ``yyyymm``, ``weight``,
    ``y_true`` (as produced by ``src.portfolio.construct`` builders).
    """
    required = {"yyyymm", "weight", "y_true"}
    missing = [c for c in required if c not in port.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    monthly = (
        port.groupby("yyyymm", as_index=False)
        .apply(lambda g: pd.Series({"port_ret": (g["weight"] * g["y_true"]).sum()}))
        .reset_index(drop=True)
    )
    return monthly.sort_values("yyyymm").reset_index(drop=True)


def portfolio_performance(monthly_ret: pd.DataFrame, annual_factor: int = 12) -> dict[str, float]:
    """Compute annualized return, vol, Sharpe, max drawdown, hit rate.

    Parameters
    ----------
    monthly_ret : pd.DataFrame
        Output of ``compute_portfolio_returns``: needs a ``port_ret`` column.
    annual_factor : int
        Periods per year (12 for monthly, 252 for daily).
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
    sharpe = np.nan if ann_vol == 0 else ann_ret / ann_vol

    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    hit_rate = (r > 0).mean()

    print(f"Annualized Return : {ann_ret:.2%}")
    print(f"Annualized Vol    : {ann_vol:.2%}")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Max Drawdown      : {max_dd:.2%}")
    print(f"Hit Rate          : {hit_rate:.2%}")

    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "max_dd": float(max_dd),
        "hit_rate": float(hit_rate),
    }
