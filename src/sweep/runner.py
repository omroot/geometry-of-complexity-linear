"""Single-cell rolling-fit runner and real-data panel loader.

* :func:`run_one_fit` — runs one rolling Grassmannian-IPCA backtest and
  returns the metric row that all sweeps share.
* :func:`load_real_panel` — loads the real-data parquet (env-overridable
  path), aliases ``Size`` to ``mcap`` when needed, and optionally
  restricts to the top-N permnos by mean market cap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.diagnostics.aggregation import aggregate_window_diagnostics
from src.models.rolling_fit import rolling_fit
from src.portfolio import (
    build_directional_portfolio,
    compute_portfolio_returns,
    portfolio_performance,
)


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_PATH: Path = REPO_ROOT / "data" / "preprocessed" / "char_data.parquet"


def run_one_fit(panel: pd.DataFrame, **rolling_fit_kwargs) -> Optional[dict]:
    """Run one rolling Grassmannian-IPCA fit and return a metrics row.

    Pipeline:
        1. ``rolling_fit(panel, **rolling_fit_kwargs)``
        2. ``aggregate_window_diagnostics`` over the fitted bases.
        3. ``build_directional_portfolio`` from OOS predictions.
        4. ``portfolio_performance`` for annualised Sharpe.
        5. OOS R² as ``1 - SSE / SS_tot``.

    Args:
        panel: Long-form ``(permno, yyyymm, characteristics, target)`` panel.
        **rolling_fit_kwargs: Forwarded to
            :func:`src.models.rolling_fit.rolling_fit`. Common keys:
            ``k``, ``train_window_months``, ``forecast_start``,
            ``rff_n_components``, ``rff_gamma``, ``rff_seed``, ``alpha``,
            ``cross_sectional_normalize``, ``market_cap_col``,
            ``market_cap_top_n``, ``min_train_obs``.

    Returns:
        Metric-row dict with the 9 keys this project's plot pipeline
        consumes (``mean_subspace_stability``, ``mean_max_principal_angle``,
        ``mean_mean_principal_angle``, ``mean_anisotropy``,
        ``mean_erank``, ``mean_spectral_gap``, ``oos_r2``, ``sharpe``,
        ``n_oos_months``). Returns ``None`` when the fit produced no OOS
        predictions (e.g. ``min_train_obs`` filter dropped every window).

        The caller is expected to annotate the result with cell identity
        (e.g. ``"sweep": "sweep_C"``, ``"k"``, ``"P"``, ``"z"``, ``"seed"``).
    """
    result = rolling_fit(panel, **rolling_fit_kwargs)
    if len(result.predictions) == 0 or not result.fitted_bases:
        return None

    diag = aggregate_window_diagnostics(result.fitted_bases, result.factor_returns)
    port = build_directional_portfolio(result.predictions)
    perf = portfolio_performance(compute_portfolio_returns(port))

    y_true = result.predictions["y_true"].to_numpy()
    y_pred = result.predictions["y_pred"].to_numpy()
    ss_tot = float((y_true ** 2).sum())
    oos_r2 = (
        float("nan") if ss_tot == 0
        else 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot
    )

    return {
        "mean_subspace_stability": diag.mean_subspace_stability,
        "mean_max_principal_angle": diag.mean_max_principal_angle,
        "mean_mean_principal_angle": diag.mean_mean_principal_angle,
        "mean_anisotropy": diag.mean_anisotropy,
        "mean_erank": diag.mean_erank,
        "mean_spectral_gap": diag.mean_spectral_gap,
        "oos_r2": oos_r2,
        "sharpe": float(perf["sharpe"]),
        "n_oos_months": diag.n_windows,
    }


def load_real_panel(
    top_n: Optional[int] = None,
    *,
    panel_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the real-data characteristics panel.

    The parquet path resolves as: ``CHAR_DATA_PARQUET`` env var (if set)
    > ``panel_path`` arg > ``data/preprocessed/char_data.parquet``.
    Aliases the ``Size`` column to ``mcap`` when needed so downstream
    code can use the market-cap filter knob in
    :func:`src.models.rolling_fit.rolling_fit` directly.

    Args:
        top_n: When given, restrict to the top-N permnos by **mean**
            market cap over the full sample. This is the fixed-universe
            convention used by the §9.1-§9.9 baseline grid (small
            look-ahead by construction). When None, returns the full
            panel — used by the §9.10 extended backtest where the
            top-N filter happens per-month inside ``rolling_fit`` via
            ``market_cap_top_n`` (no look-ahead).
        panel_path: Override the default parquet path. Ignored when the
            ``CHAR_DATA_PARQUET`` env var is set.

    Returns:
        Long-form DataFrame with ``permno``, ``yyyymm``, ``mcap``, and
        the characteristic columns.

    Raises:
        FileNotFoundError: When the parquet is missing at the resolved
            path.
        ValueError: When the panel lacks both ``mcap`` and ``Size``.
    """
    resolved = Path(
        os.environ.get(
            "CHAR_DATA_PARQUET",
            str(panel_path) if panel_path is not None else str(DEFAULT_PANEL_PATH),
        )
    )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Real-data parquet not found at {resolved}. "
            "Run scripts/00_build_panel.py to produce it."
        )

    panel = pd.read_parquet(resolved, engine="pyarrow")
    if "mcap" not in panel.columns and "Size" in panel.columns:
        panel["mcap"] = panel["Size"]
    if "mcap" not in panel.columns:
        raise ValueError("Panel has neither 'mcap' nor 'Size' column.")

    if top_n is not None:
        top_permnos = panel.groupby("permno")["mcap"].mean().nlargest(top_n).index
        panel = panel[panel["permno"].isin(top_permnos)]

    return panel.reset_index(drop=True)
