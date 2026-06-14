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

import numpy as np
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

# VoC WRDS top-50 permnos search order:
#   1. data/reference/top50_permnos.csv  (bundled in this repo — works on Colab)
#   2. sibling VoC repo (local dev convenience)
_TOP50_CANDIDATES: list[Path] = [
    REPO_ROOT / "data" / "reference" / "top50_permnos.csv",
    (
        REPO_ROOT.parent
        / "TheVirtueOfComplexity_PaperReplication_Experimentation"
        / "data"
        / "top50_permnos.csv"
    ),
]


def load_voc_top50_permnos() -> Optional[list]:
    """Return the VoC WRDS-derived top-50 permno list, or None if not found.

    Searches data/reference/ first (works on Colab after cloning this repo),
    then falls back to the sibling VoC repo for local dev.
    """
    for path in _TOP50_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)["permno"].astype(int).tolist()
    return None


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
    # Filter to finite pairs only — y_true can contain NaN when a stock has
    # valid characteristics but a missing return in the panel.
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[finite_mask]
    y_p = y_pred[finite_mask]
    # Centered SS_tot matches the original backtest_GRASS_IPCA.py convention:
    # R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²  (null model predicts the sample mean)
    ss_tot = float(((y_t - y_t.mean()) ** 2).sum()) if len(y_t) > 1 else 0.0
    oos_r2 = (
        float("nan") if ss_tot == 0
        else 1.0 - float(((y_t - y_p) ** 2).sum()) / ss_tot
    )

    return {
        "mean_subspace_stability": diag.mean_subspace_stability,
        "mean_max_principal_angle": diag.mean_max_principal_angle,
        "mean_mean_principal_angle": diag.mean_mean_principal_angle,
        "mean_anisotropy": diag.mean_anisotropy,
        "mean_geodesic_accel": diag.mean_geodesic_accel,
        "mean_erank": diag.mean_erank,
        "mean_spectral_gap": diag.mean_spectral_gap,
        "oos_r2": oos_r2,
        "sharpe": float(perf["sharpe"]),
        "n_oos_months": diag.n_windows,
    }


def run_seed_averaged_fit(
    panel: pd.DataFrame,
    *,
    rff_seeds: tuple[int, ...],
    **rolling_fit_kwargs,
) -> Optional[dict]:
    """Run one grid cell over multiple RFF seeds and average metric rows."""
    seed_metrics: list[dict] = []
    for seed in rff_seeds:
        metrics = run_one_fit(panel, rff_seed=seed, **rolling_fit_kwargs)
        if metrics is not None:
            seed_metrics.append(metrics)
            print(
                f"      seed={seed}: sharpe={metrics['sharpe']:+.4f}  "
                f"r2={metrics['oos_r2']:.4f}  "
                f"erank={metrics['mean_erank']:.4f}  "
                f"gap={metrics['mean_spectral_gap']:.4f}",
                flush=True,
            )

    if not seed_metrics:
        return None

    return {
        key: float(np.mean([metrics[key] for metrics in seed_metrics]))
        for key in seed_metrics[0]
    } | {"n_seeds": len(seed_metrics)}


def load_real_panel(
    top_n: Optional[int] = None,
    *,
    permno_list: Optional[list] = None,
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
            market cap over the full sample (using nsmallest on Size proxy).
            Ignored when ``permno_list`` is provided.
        permno_list: Explicit list of permnos to keep. Takes precedence over
            ``top_n``. Use this to match the exact VoC reference universe
            (loaded from ``data/top50_permnos.csv``).
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

    if permno_list is not None:
        # Explicit permno list — matches exact VoC reference universe.
        panel = panel[panel["permno"].isin(permno_list)]
    elif top_n is not None:
        # Size (= mcap alias) is negatively correlated with market cap:
        # more negative Size → larger company.  Use nsmallest to select top-N.
        top_permnos = panel.groupby("permno")["mcap"].mean().nsmallest(top_n).index
        panel = panel[panel["permno"].isin(top_permnos)]

    # Align target: y_ipca[t] = excess_ret[t+1] so the model predicts
    # NEXT month's return from current-month characteristics.  Without this
    # shift, STreversal (= -excess_ret[t]) is a perfect same-period predictor
    # and inflates Sharpe by ~60x.  Matches the reference notebook convention:
    #   df["y_ipca"] = df.groupby("permno")["excess_ret"].shift(-1)
    # Calendar-month merge (not row-based shift) so gaps don't misalign labels.
    panel = panel.copy()
    panel["yyyymm"] = pd.to_datetime(panel["yyyymm"])
    next_ret = panel[["permno", "yyyymm", "excess_ret"]].copy()
    next_ret["yyyymm"] = next_ret["yyyymm"] - pd.DateOffset(months=1)
    next_ret = next_ret.rename(columns={"excess_ret": "_y_ipca_next"})
    panel = panel.merge(next_ret, on=["permno", "yyyymm"], how="left")
    panel["y_ipca"] = panel["_y_ipca_next"]
    panel = panel.drop(columns=["_y_ipca_next"])

    return panel.reset_index(drop=True)
