"""Portfolio plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative(monthly_ret: pd.DataFrame) -> None:
    """Plot the cumulative-return curve from monthly portfolio returns.

    Args:
        monthly_ret: Output of
            :func:`src.portfolio.performance.compute_portfolio_returns`;
            must have ``yyyymm`` and ``port_ret`` columns.

    Raises:
        ValueError: If either required column is missing.
    """
    required = {"yyyymm", "port_ret"}
    missing = [c for c in required if c not in monthly_ret.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = monthly_ret.copy()
    data["yyyymm"] = pd.to_datetime(data["yyyymm"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    data["port_ret"] = pd.to_numeric(data["port_ret"], errors="coerce")
    data = data.dropna(subset=["yyyymm", "port_ret"]).sort_values("yyyymm")

    cum = (1 + data["port_ret"]).cumprod()
    plt.figure(figsize=(12, 5))
    plt.plot(data["yyyymm"], cum)
    plt.title("IPCA Long-Only Portfolio — Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
