"""Simple no-leakage rolling Grassmannian-IPCA fit for the synthetic experiments.

Replaces ``src.workflows.grassmannian_ipca.GrassmannianIPCAWorkflow.rolling_ipca_predictions``
with a stripped-down loop sufficient for the §8 synthetic scripts. Differences:

* No market-cap filter, no warm-start factor alignment, no rolling-normalization
  toggle, no progress bar, no per-(seed, RFF) caching.
* RFF lift is sampled once per call (``rff_seed``) and applied with frozen
  normalization stats taken from the first training window.
* Returns the fitted Grassmannian bases and factor returns per OOS month so
  downstream diagnostics (effective rank, spectral gap, principal angles,
  anisotropy) can be computed without re-fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.models.grassmannian_ipca import GrassmannianManifoldIPCAEstimator
from src.models.rff import RandomFourierFeatures


_NON_CHAR_COLUMNS: frozenset[str] = frozenset(
    {
        "yyyymm", "permno",
        "excess_ret", "ret", "ret_adj", "y_ipca",
        "mcap", "Price",
    }
)


@dataclass
class RollingFitResult:
    """Per-OOS-month outputs of a single rolling Grassmannian-IPCA run.

    Attributes:
        predictions: Long-form DataFrame with one row per (asset, OOS month).
            Columns: ``yyyymm``, ``permno``, ``y_true``, ``y_pred``,
            ``train_end``, ``n_train``, ``n_test``.
        fitted_bases: One fitted basis ``W`` (shape ``(m_eff, k)``) per OOS
            month, in chronological order.
        factor_returns: One per-OOS-month factor history ``f_hat`` of shape
            ``(k, train_window_months)``. Used by spectral diagnostics.
        oos_months: OOS month ``Timestamp`` matching each entry of
            ``fitted_bases`` / ``factor_returns``.
    """

    predictions: pd.DataFrame
    fitted_bases: list[np.ndarray] = field(default_factory=list)
    factor_returns: list[np.ndarray] = field(default_factory=list)
    oos_months: list[pd.Timestamp] = field(default_factory=list)


def _detect_char_cols(panel: pd.DataFrame, target_col: str) -> list[str]:
    """Pick numeric columns that look like characteristics."""
    exclude = _NON_CHAR_COLUMNS | {target_col}
    return [
        c for c in panel.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(panel[c])
    ]


def _panel_to_rt_z(
    X: np.ndarray, y: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reshape long-form (X, y, indices) into the estimator's (rets, Z) layout.

    Requires unique (asset, month) rows. Mirrors the helper in
    ``src.workflows.grassmannian_ipca._panel_to_grassmannian_data``.
    """
    asset_codes, asset_inv = np.unique(indices[:, 0], return_inverse=True)
    date_codes, date_inv = np.unique(indices[:, 1], return_inverse=True)

    pair_codes = np.ravel_multi_index(
        (date_inv, asset_inv),
        dims=(date_codes.size, asset_codes.size),
    )
    if np.unique(pair_codes).size != X.shape[0]:
        raise ValueError("Duplicate (asset, month) rows detected.")

    rets = np.zeros((date_codes.size, asset_codes.size), dtype=np.float64)
    Z = np.zeros((date_codes.size, asset_codes.size, X.shape[1]), dtype=np.float64)
    rets[date_inv, asset_inv] = y
    Z[date_inv, asset_inv, :] = X
    metad = {
        "ids": asset_codes,
        "dates": date_codes,
        "N": int(asset_codes.size),
        "T": int(date_codes.size),
        "L": int(X.shape[1]),
    }
    return rets, Z, metad


def rolling_fit(
    panel: pd.DataFrame,
    *,
    k: int,
    train_window_months: int,
    forecast_start: str | pd.Timestamp,
    rff_n_components: int = 0,
    rff_gamma: float = 0.25,
    rff_seed: int = 0,
    alpha: float = 100.0,
    char_cols: Optional[list[str]] = None,
    target_col: str = "y_ipca",
    id_col: str = "permno",
    time_col: str = "yyyymm",
    min_train_obs: int = 1,
    normalize: bool = True,
    cross_sectional_normalize: bool = False,
    market_cap_col: Optional[str] = None,
    market_cap_top_n: Optional[int] = None,
    warm_start: bool = True,
    max_iter: int = 200,
    iter_tol: float = 1e-6,
) -> RollingFitResult:
    """Slide a Grassmannian-IPCA fit across the panel and predict each OOS month.

    For each month at or after ``forecast_start``, fits on the trailing
    ``train_window_months`` and predicts that month using the mean training
    factor. Normalization statistics are frozen from the first training
    window. When ``rff_n_components > 0``, characteristics are lifted to
    ``2 * rff_n_components``-dimensional random Fourier features (omega
    frozen per ``rff_seed``).

    Args:
        panel: Long-form DataFrame with ``permno``, ``yyyymm``, ``target_col``,
            and characteristic columns.
        k: Number of latent factors.
        train_window_months: Rolling training window length.
        forecast_start: First OOS month (inclusive).
        rff_n_components: ``0`` for the linear baseline; otherwise the number
            of random frequencies (output dim is ``2 * rff_n_components``).
        rff_gamma: Bandwidth of the random frequencies.
        rff_seed: Seed for the omega draw.
        alpha: Ridge shrinkage passed to the IPCA factor step.
        char_cols: Override the auto-detected characteristic columns.
        target_col: Column holding next-month excess return.
        id_col: Asset identifier column.
        time_col: Month column.
        min_train_obs: Skip an OOS month if the training window has fewer
            than this many finite observations.
        normalize: Standardize characteristics with stats from the first
            training window.
        cross_sectional_normalize: When True, override ``normalize`` and
            z-score characteristics within each month before any
            train/test slicing (paper §9.1 convention for real data:
            "each characteristic is demeaned and scaled to unit
            cross-sectional standard deviation"). Mutually exclusive
            with ``normalize`` semantically; takes precedence when True.
        market_cap_col: Column holding a market-cap proxy. When set with
            ``market_cap_top_n``, the universe at each OOS month is
            restricted to the top-N stocks by **previous-month** market
            cap (no look-ahead).
        market_cap_top_n: Number of stocks to retain per OOS month.
            Ignored when ``market_cap_col`` is None.
        warm_start: Reuse the previous month's fitted basis as the optimizer
            initial point.
        max_iter: Maximum optimizer iterations.
        iter_tol: Optimizer gradient-norm tolerance.

    Returns:
        :class:`RollingFitResult` with predictions and per-OOS-month bases /
        factors. The ``predictions`` DataFrame is empty if no OOS month
        meets ``min_train_obs``.

    Raises:
        ValueError: If ``market_cap_top_n`` is given without ``market_cap_col``,
            or if ``market_cap_col`` is missing from the panel.
    """
    if market_cap_top_n is not None and market_cap_col is None:
        raise ValueError("market_cap_top_n requires market_cap_col.")
    if market_cap_col is not None and market_cap_col not in panel.columns:
        raise ValueError(f"market_cap_col '{market_cap_col}' missing from panel.")
    work = panel.copy()
    work[time_col] = (
        pd.to_datetime(work[time_col], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    work = work.dropna(subset=[id_col, time_col]).copy()

    if target_col not in work.columns:
        raise ValueError(f"target_col '{target_col}' missing from panel.")

    if char_cols is None:
        char_cols = _detect_char_cols(work, target_col)
    if not char_cols:
        raise ValueError("No characteristic columns detected.")

    work = work.sort_values([time_col, id_col]).reset_index(drop=True)
    work["i_idx"] = work[id_col].factorize(sort=True)[0].astype(np.int64)
    work["t_idx"] = work[time_col].factorize(sort=True)[0].astype(np.int64)

    months = np.sort(work[time_col].dropna().unique())
    start_ts = pd.Timestamp(forecast_start).to_period("M").to_timestamp()
    pred_positions = np.flatnonzero(months >= start_ts.to_datetime64())
    if pred_positions.size == 0:
        return RollingFitResult(predictions=_empty_predictions(id_col, time_col))

    id_all = work[id_col].to_numpy()
    time_all = work[time_col].to_numpy()
    X_raw_all = work[char_cols].to_numpy(np.float64)
    y_all = work[target_col].to_numpy(np.float64)
    idx_all = work[["i_idx", "t_idx"]].to_numpy(np.int64)
    mcap_all = (
        work[market_cap_col].to_numpy(np.float64) if market_cap_col is not None
        else None
    )

    t_codes = idx_all[:, 1]
    change_points = np.flatnonzero(np.diff(t_codes)) + 1
    month_starts = np.r_[0, change_points]
    month_ends = np.r_[change_points, len(work)]

    # Precompute the train-window start position for each test month.
    months_ns = months.astype("datetime64[ns]")
    train_start_pos = np.zeros(months.size, dtype=np.int64)
    for pos, month_val in enumerate(months):
        cutoff = (
            pd.Timestamp(month_val) - pd.DateOffset(months=train_window_months)
        ).to_datetime64()
        train_start_pos[pos] = int(np.searchsorted(months_ns, cutoff, side="left"))

    # Cross-sectional z-score per month (paper §9.1 convention) takes precedence
    # over the fixed-first-window normalization. When enabled, we pre-normalize
    # X_raw_all once so the downstream code path is uniform.
    if cross_sectional_normalize:
        X_raw_all = _cross_sectional_normalize(
            X_raw_all, month_starts, month_ends
        )
        norm_mu = np.zeros(len(char_cols), dtype=np.float64)
        norm_sd = np.ones(len(char_cols), dtype=np.float64)
    else:
        # Fixed normalization stats from the first training window.
        norm_mu = None
        norm_sd = None
        if normalize:
            first_test = int(pred_positions[0])
            base_start = int(train_start_pos[first_test])
            if base_start < first_test:
                base_slice = slice(month_starts[base_start], month_ends[first_test - 1])
                X_base = X_raw_all[base_slice]
                X_base = X_base[np.isfinite(X_base).all(axis=1)]
                if X_base.size > 0:
                    norm_mu = X_base.mean(axis=0)
                    sd = X_base.std(axis=0, ddof=0)
                    norm_sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        if norm_mu is None:
            norm_mu = np.zeros(len(char_cols), dtype=np.float64)
            norm_sd = np.ones(len(char_cols), dtype=np.float64)

    # Previous-month top-N universe per OOS month (no look-ahead).
    prev_universe: Optional[list[np.ndarray]] = None
    if market_cap_top_n is not None:
        prev_universe = _build_prev_month_universe(
            id_all=id_all,
            month_starts=month_starts,
            month_ends=month_ends,
            mcap_all=mcap_all,
            top_n=int(market_cap_top_n),
            n_months=months.size,
        )

    # RFF projection (omega frozen across the run).
    rff_omega: Optional[np.ndarray] = None
    if rff_n_components > 0:
        rng = np.random.default_rng(rff_seed)
        rff_omega = (
            rng.standard_normal((len(char_cols), rff_n_components))
            * float(rff_gamma)
        )

    def _lift(X_norm: np.ndarray) -> np.ndarray:
        if rff_omega is None:
            return X_norm
        proj = X_norm @ rff_omega
        # Raw sin/cos — matches the reference §8.2 pipeline and the canonical
        # RandomFourierFeatures class in src/models/rff.py. The sqrt(2/m)
        # normalization that previously lived here changed the effective
        # ridge scale relative to the paper's setup and shifted the
        # Sharpe-vs-z optimum.
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=1)

    prediction_rows: list[pd.DataFrame] = []
    fitted_bases: list[np.ndarray] = []
    factor_returns: list[np.ndarray] = []
    oos_months: list[pd.Timestamp] = []
    prev_basis: Optional[np.ndarray] = None

    for test_pos in pred_positions:
        test_pos = int(test_pos)
        test_month = months[test_pos]
        start_pos = int(train_start_pos[test_pos])
        if start_pos >= test_pos:
            continue

        train_slice = slice(month_starts[start_pos], month_ends[test_pos - 1])
        test_slice = slice(month_starts[test_pos], month_ends[test_pos])

        X_train_raw = X_raw_all[train_slice]
        y_train = y_all[train_slice]
        idx_train = idx_all[train_slice]
        ids_train = id_all[train_slice]
        finite_train = (
            np.isfinite(X_train_raw).all(axis=1) & np.isfinite(y_train)
        )
        if prev_universe is not None:
            selected = prev_universe[test_pos]
            if selected.size == 0:
                continue
            finite_train &= np.isin(ids_train, selected)
        if int(finite_train.sum()) < min_train_obs:
            continue

        X_train = (X_train_raw[finite_train] - norm_mu) / norm_sd
        y_train_f = y_train[finite_train]
        idx_train_f = idx_train[finite_train]

        X_train_lift = _lift(X_train)

        X_test_raw = X_raw_all[test_slice]
        y_test = y_all[test_slice]
        ids_test = id_all[test_slice]
        finite_test = np.isfinite(X_test_raw).all(axis=1)
        if prev_universe is not None:
            finite_test &= np.isin(ids_test, prev_universe[test_pos])
        if not finite_test.any():
            continue
        X_test = (X_test_raw[finite_test] - norm_mu) / norm_sd
        X_test_lift = _lift(X_test)

        rets_train, Z_train, _metad = _panel_to_rt_z(X_train_lift, y_train_f, idx_train_f)
        m_eff = Z_train.shape[2]
        if m_eff < k:
            continue

        estimator = GrassmannianManifoldIPCAEstimator(
            n_assets=rets_train.shape[1],
            k=k,
            m=m_eff,
            t_train=rets_train.shape[0],
            shrinkage=max(float(alpha), 1e-8),
        )

        initial_point = None
        if warm_start and prev_basis is not None and prev_basis.shape == (m_eff, k):
            initial_point = prev_basis

        Wopt, f_hat, _history = estimator.fit(
            data=(rets_train, Z_train),
            optimizer="ConjugateGradient",
            max_iterations=max_iter,
            iter_tol=iter_tol,
            verbosity=0,
            initial_point=initial_point,
        )
        prev_basis = np.asarray(Wopt, dtype=np.float64)
        # Convert (T_train, k) -> (k, T_train) for downstream diagnostics.
        factor_history = np.asarray(f_hat, dtype=np.float64).T

        # Predict OOS using the mean training factor: y_pred = X @ W @ f_bar.
        factor_mean = factor_history.mean(axis=1)
        y_pred = X_test_lift @ Wopt @ factor_mean

        prediction_rows.append(
            pd.DataFrame(
                {
                    id_col: ids_test[finite_test],
                    time_col: np.full(finite_test.sum(), test_month, dtype="datetime64[ns]"),
                    "y_true": y_test[finite_test],
                    "y_pred": y_pred,
                    "train_end": months[test_pos - 1],
                    "n_train": int(finite_train.sum()),
                    "n_test": int(finite_test.sum()),
                }
            )
        )
        fitted_bases.append(prev_basis)
        factor_returns.append(factor_history)
        oos_months.append(pd.Timestamp(test_month))

    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else _empty_predictions(id_col, time_col)
    )
    return RollingFitResult(
        predictions=predictions,
        fitted_bases=fitted_bases,
        factor_returns=factor_returns,
        oos_months=oos_months,
    )


def _empty_predictions(id_col: str, time_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[id_col, time_col, "y_true", "y_pred", "train_end", "n_train", "n_test"]
    )


def _cross_sectional_normalize(
    X: np.ndarray, month_starts: np.ndarray, month_ends: np.ndarray
) -> np.ndarray:
    """Z-score each month's cross-section independently (paper §9.1).

    For each month block, replaces ``X[start:end]`` with
    ``(X[start:end] - mu_m) / sd_m`` where ``mu_m``/``sd_m`` ignore NaNs.
    Zero/NaN standard deviations are replaced by 1.0.
    """
    out = X.astype(np.float64, copy=True)
    for start, end in zip(month_starts, month_ends):
        block = out[start:end]
        mu = np.nanmean(block, axis=0)
        sd = np.nanstd(block, axis=0, ddof=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        out[start:end] = (block - mu) / sd
    return out


def _build_prev_month_universe(
    *,
    id_all: np.ndarray,
    month_starts: np.ndarray,
    month_ends: np.ndarray,
    mcap_all: np.ndarray,
    top_n: int,
    n_months: int,
) -> list[np.ndarray]:
    """For each month, the top-N permnos by the **previous** month's mcap.

    No look-ahead: the universe applied at test month ``t`` is built from the
    cross-section observed at ``t - 1``. The first month gets an empty
    universe (no prior cross-section to rank).

    Args:
        id_all: Per-row permno (object array).
        month_starts: Per-month start row index.
        month_ends: Per-month end row index (exclusive).
        mcap_all: Per-row market-cap values; NaNs treated as missing.
        top_n: Number of stocks to retain per OOS month.
        n_months: Total month count.

    Returns:
        List of length ``n_months`` of permno arrays; entry ``t`` is the
        universe to apply at test month ``t``.
    """
    universes: list[np.ndarray] = [np.array([], dtype=id_all.dtype)]
    for pos in range(1, n_months):
        prev_slice = slice(month_starts[pos - 1], month_ends[pos - 1])
        prev_ids = id_all[prev_slice]
        prev_mcap = mcap_all[prev_slice]
        mask = np.isfinite(prev_mcap)
        if not mask.any():
            universes.append(np.array([], dtype=id_all.dtype))
            continue
        prev_ids = prev_ids[mask]
        prev_mcap = prev_mcap[mask]
        if prev_ids.size <= top_n:
            universes.append(np.unique(prev_ids))
            continue
        order = np.argsort(prev_mcap)[::-1]  # descending
        top_ids = prev_ids[order[:top_n]]
        universes.append(np.unique(top_ids))
    return universes
