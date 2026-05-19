"""Unified rolling-window Grassmannian-IPCA backtest.

Consolidates ``run_ipca_grassmannian_v2`` (equal-weighted Sharpe) and
``run_ipca_grassmannian_mktcap`` (value-weighted Sharpe) into a single
function :func:`run_backtest` and re-uses the canonical diagnostics from
:mod:`src.diagnostics`.

The numerical conventions for the seven geometric diagnostics
(``d_proj``, ``geodesic_accel``, ``erank``, ``spectral_gap``,
``d_proj_norm``, ``d_proj_seeds``, principal angles) are bit-exact
matches of the originals. Two compatibility shims,
:func:`run_ipca_grassmannian_v2` and :func:`run_ipca_grassmannian_mktcap`,
preserve the legacy entry points.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.diagnostics import effective_rank, spectral_gap
from src.models.grassmannian_ipca import GrassmannianManifoldIPCAEstimator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _principal_angles_desc(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles sorted descending, assuming orthonormal columns.

    Bypasses the QR safety net of :func:`src.diagnostics.principal_angles`
    so that already-orthonormal inputs (as produced by the estimator) are
    SVD'd directly. The QR step is a no-op in principle but introduces
    ~1e-8 floating-point noise; skipping it preserves bit-exact agreement
    with the legacy implementation.
    """
    cos_angles = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    return np.sort(np.arccos(cos_angles))[::-1]


def _projection_distance_from_angles(angles: np.ndarray) -> float:
    """Frobenius projection distance derived from principal angles.

    Uses the identity ``||P_a − P_b||_F = sqrt(2 sum sin²(ψ))`` to avoid
    forming the ``m × m`` projector matrices. Bit-exact match with the
    legacy backtest, which uses the same shortcut.
    """
    return float(np.sqrt(2.0 * np.sum(np.sin(angles) ** 2)))


def _projection_distance_from_gram(
    basis_a: np.ndarray, basis_b: np.ndarray, k: int
) -> float:
    """Frobenius projection distance via the ``k × k`` Gram-matrix shortcut.

    Uses the identity ``||P_a − P_b||²_F = 2 (k − ||W_a^T W_b||²_F)``,
    O(P k²) instead of O(P² k). Matches the legacy backtest's seed-variance
    computation bit-exactly.
    """
    gram = basis_a.T @ basis_b
    return float(np.sqrt(max(0.0, 2.0 * (k - np.sum(gram ** 2)))))


def _erank_with_legacy_fallback(eigvals: np.ndarray) -> float:
    """Effective rank with the legacy ``erank == 1.0`` sentinel for an all-zero spectrum.

    :func:`src.diagnostics.effective_rank` returns 0.0 when the spectrum
    sums to zero; the legacy backtest returned 1.0. Preserved here for
    bit-exact reproducibility.
    """
    eigvals = np.maximum(np.asarray(eigvals, dtype=float), 0.0)
    if eigvals.sum() <= 0:
        return 1.0
    return effective_rank(eigvals)


def _spectral_gap_with_boundary_fallback(
    eigvals: np.ndarray, num_fact: int
) -> float:
    """Spectral-gap diagnostic with the legacy ``len(eigvals) <= k`` fallback.

    The fitted ``Σ̂_f`` is ``k × k``, so there is no ``(k+1)``-th eigenvalue.
    The legacy backtest in that case returns ``λ_k / λ_1`` (clipped to
    ``[0, 1]``), interpreting the smallest retained eigenvalue itself as a
    noise-floor proxy.
    """
    eigvals_desc = np.sort(np.asarray(eigvals, dtype=float))[::-1]
    leading = eigvals_desc[0] if eigvals_desc[0] > 0 else 1.0
    if len(eigvals_desc) > num_fact:
        return spectral_gap(eigvals_desc, num_fact, normalize=True)
    gap = eigvals_desc[-1]
    return float(np.clip(gap / leading, 0.0, 1.0))


def _haar_baseline_projection_distance(m: int, k: int) -> float:
    """Expected ``d_proj`` between two Haar-uniform points on ``Gr(m, k)``.

    Equals ``sqrt(2 k (m - k) / (m + 1))``; derived from
    ``E[sin²(ψ_i)] = (m - k)/(m + 1)`` under the Haar measure.
    """
    if m <= k:
        return 0.0
    return float(np.sqrt(2.0 * k * (m - k) / (m + 1)))


def _build_panels(
    df_trunc: pd.DataFrame,
    char_feats: list[str],
    weighting: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame | None]:
    """Unstack the multi-index frame into return / characteristic / mktcap panels."""
    df_panel = df_trunc.copy().sort_index()
    rets_panel = df_panel["ret"].unstack("permno").sort_index()
    char_panels = [df_panel[col].unstack("permno").sort_index() for col in char_feats]
    if weighting in ("mktcap", "both"):
        mktcap_panel = df_panel["mktcap"].unstack("permno").sort_index()
    else:
        mktcap_panel = None
    return rets_panel, char_panels, mktcap_panel


def _clean_align_impute(
    rets_panel: pd.DataFrame,
    char_panels: list[pd.DataFrame],
    mktcap_panel: pd.DataFrame | None,
    max_nan: float,
    max_zero: float,
    min_non_nan_frac: float,
    impute: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame | None]:
    """Asset alignment, NaN/zero filtering, burn-in, imputation, char standardization.

    Mirrors the original v2 (when ``mktcap_panel is None``) and mktcap variants
    bit-for-bit, except that ``impute`` is now explicit. With ``impute="mean"``
    behavior matches v2; with ``impute="median"`` cross-sectional medians are
    used for the returns and characteristics. The mktcap panel always uses
    a positive-median fallback (matching the legacy mktcap variant).
    """
    common_assets = rets_panel.columns
    if mktcap_panel is not None:
        common_assets = common_assets.intersection(mktcap_panel.columns)
    for x in char_panels:
        common_assets = common_assets.intersection(x.columns)

    rets_panel = rets_panel[common_assets]
    char_panels = [x[common_assets] for x in char_panels]
    if mktcap_panel is not None:
        mktcap_panel = mktcap_panel[common_assets]

    zero_frac = (rets_panel == 0).mean()
    nan_frac = rets_panel.isna().mean()
    good_assets = (nan_frac <= max_nan) & (zero_frac <= max_zero)
    for x in char_panels:
        good_assets &= x.isna().mean() <= max_nan
        good_assets &= (x == 0).mean() <= max_zero

    rets_panel = rets_panel.loc[:, good_assets]
    char_panels = [x.loc[:, good_assets] for x in char_panels]
    if mktcap_panel is not None:
        mktcap_panel = mktcap_panel.loc[:, good_assets]

    min_non_nan = int(min_non_nan_frac * rets_panel.shape[1])
    valid_time = rets_panel.notna().sum(axis=1) >= min_non_nan
    for x in char_panels:
        valid_time &= x.notna().sum(axis=1) >= min_non_nan

    rets_panel = rets_panel.loc[valid_time]
    char_panels = [x.loc[valid_time] for x in char_panels]
    if mktcap_panel is not None:
        mktcap_panel = mktcap_panel.loc[valid_time]
        # Lag mktcap by one period (beginning-of-period weights, no look-ahead).
        mktcap_panel = mktcap_panel.shift(1)

    if impute == "median":
        impute_fn = lambda s: s.fillna(s.median())
    else:
        impute_fn = lambda s: s.fillna(s.mean())
    rets_panel = rets_panel.apply(impute_fn, axis=1)
    char_panels = [x.apply(impute_fn, axis=1) for x in char_panels]

    if mktcap_panel is not None:
        mktcap_panel = mktcap_panel.apply(
            lambda s: s.where(s > 0).fillna(s[s > 0].median()), axis=1
        ).fillna(0.0)

    char_panels = [
        (x.sub(x.mean(axis=1), axis=0))
        .div(x.std(axis=1).replace(0, np.nan), axis=0)
        for x in char_panels
    ]
    char_panels = [x.fillna(0.0) for x in char_panels]

    return rets_panel, char_panels, mktcap_panel


def _market_timing_sharpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Annualized equal-weighted market-timing Sharpe.

    Bit-exact reproduction of the legacy formula:

    ``pos = y_pred / (mean(|y_pred|, axis=1) + 1e-12)``,
    ``port_ret = mean(pos * y_true, axis=1)``,
    ``Sharpe = mean / (std_{ddof=1} + 1e-12) * sqrt(12)``.
    """
    pos = y_pred / (np.mean(np.abs(y_pred), axis=1, keepdims=True) + 1e-12)
    port_ret = np.mean(pos * y_true, axis=1)
    return float(
        np.mean(port_ret) / (np.std(port_ret, ddof=1) + 1e-12) * np.sqrt(12)
    )


def _mktcap_sharpe(
    y_true: np.ndarray, y_pred: np.ndarray, mktcap_lagged: np.ndarray
) -> float:
    """Annualized value-weighted Sharpe (lagged-mktcap weights, S&P style).

    Bit-exact reproduction of the legacy mktcap formula.
    """
    mc_clipped = np.maximum(mktcap_lagged, 0.0)
    mc_sum = mc_clipped.sum(axis=1, keepdims=True)
    weights = mc_clipped / (mc_sum + 1e-12)
    signal_scale = np.sum(weights * np.abs(y_pred), axis=1, keepdims=True)
    positions = y_pred / (signal_scale + 1e-12)
    port_ret = np.sum(weights * positions * y_true, axis=1)
    return float(
        np.mean(port_ret) / (np.std(port_ret, ddof=1) + 1e-12) * np.sqrt(12)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    df_trunc: pd.DataFrame,
    char_feats: list[str],
    num_fact: int = 2,
    window_len: int = 12,
    max_nan: float = 0.3,
    max_zero: float = 0.3,
    min_non_nan_frac: float = 0.7,
    shrinkage: float = 0.0,
    min_gradient_norm: float = 1e-6,
    compute_seed_variance: bool = True,
    weighting: Literal["equal", "mktcap", "both"] = "equal",
    impute: Literal["mean", "median"] = "mean",
    verbose: bool = True,
) -> dict:
    """Rolling-window OOS backtest for Grassmannian-IPCA with geometric diagnostics.

    At each time step ``t > window_len`` the function:

      1. Fits ``W_t ∈ Gr(m, k)`` on the trailing ``window_len`` observations.
      2. Predicts ``r̂_t = (Z_t W_t) f̄`` where ``f̄`` is the mean in-window
         factor return.
      3. Records the seven geometric diagnostics from §4–§6 of the paper
         (projection distance, principal angles, geodesic acceleration,
         effective rank, spectral gap, Haar-normalized d_proj, and
         between-seed d_proj when ``compute_seed_variance=True``).

    Args:
        df_trunc: Multi-index ``(date, permno)`` DataFrame. Must contain
            ``"ret"`` (return, already lagged/aligned) and each name in
            ``char_feats``. Must also contain ``"mktcap"`` when
            ``weighting`` is ``"mktcap"`` or ``"both"``.
        char_feats: Names of characteristic columns to use as instruments
            ``Z``.
        num_fact: Number of latent factors ``k``.
        window_len: Rolling estimation window in periods.
        max_nan: Maximum fraction of NaN values allowed per asset.
        max_zero: Maximum fraction of exact-zero values allowed per asset.
        min_non_nan_frac: Minimum fraction of assets with non-NaN values
            required for a time period to be kept (burn-in filter).
        shrinkage: Ridge-like shrinkage parameter passed to the estimator.
        min_gradient_norm: Convergence tolerance forwarded to the
            estimator as ``iter_tol``.
        compute_seed_variance: Whether to refit each window from a fresh
            random initialization to compute the between-seed
            ``d_proj_seeds`` diagnostic.
        weighting: Which Sharpe to compute. ``"equal"`` reproduces the
            legacy ``run_ipca_grassmannian_v2``; ``"mktcap"`` reproduces
            the legacy ``run_ipca_grassmannian_mktcap`` (requires
            ``"mktcap"`` column); ``"both"`` returns both.
        impute: Cross-sectional NaN imputation for returns and
            characteristics (``"mean"`` reproduces v2, ``"median"`` is
            more robust). The mktcap panel always uses a positive-median
            fallback.
        verbose: Whether to print a summary at the end.

    Returns:
        Dict with the following keys (a superset of both legacy variants):

          * Performance: ``r2_oos``, ``sharpe``, ``sharpe_ew``,
            ``sharpe_mktcap`` (latter two when applicable).
          * Subspace stability: ``mean_subspace_stability``,
            ``stability_dist_series``.
          * Principal angles: ``mean_max_principal_angle``,
            ``mean_mean_principal_angle``, ``principal_angles_series``,
            ``std_mean_principal_angle``, ``std_max_principal_angle``,
            ``sr_mean_principal_angle``, ``sr_max_principal_angle``.
          * Geodesic acceleration: ``mean_geodesic_accel``,
            ``geodesic_accel_series``.
          * Effective rank: ``mean_erank``, ``erank_series``,
            ``erank_collapse_frac``.
          * Spectral gap: ``mean_spectral_gap``, ``spectral_gap_series``.
          * Haar-normalized d_proj: ``mean_d_proj_norm``,
            ``d_proj_norm_series``, ``d_proj_random_baseline``.
          * Between-seed d_proj: ``mean_d_proj_seeds``,
            ``d_proj_seeds_series``, ``d_proj_seeds_flat_frac``.
    """
    if weighting not in ("equal", "mktcap", "both"):
        raise ValueError(
            f"weighting must be 'equal', 'mktcap', or 'both'; got {weighting!r}"
        )
    if impute not in ("mean", "median"):
        raise ValueError(f"impute must be 'mean' or 'median'; got {impute!r}")
    if weighting in ("mktcap", "both") and "mktcap" not in df_trunc.columns:
        raise ValueError(
            f"weighting={weighting!r} requires a 'mktcap' column on df_trunc"
        )

    # -------- Panel build and clean --------
    rets_panel, char_panels, mktcap_panel = _build_panels(
        df_trunc, char_feats, weighting
    )
    rets_panel, char_panels, mktcap_panel = _clean_align_impute(
        rets_panel,
        char_panels,
        mktcap_panel,
        max_nan=max_nan,
        max_zero=max_zero,
        min_non_nan_frac=min_non_nan_frac,
        impute=impute,
    )

    Z = np.stack([x.to_numpy(dtype=float) for x in char_panels], axis=2)
    rets_stock = rets_panel.to_numpy(dtype=float)
    mktcap_arr = (
        mktcap_panel.to_numpy(dtype=float) if mktcap_panel is not None else None
    )

    T, N = rets_stock.shape
    m_dim = Z.shape[2]
    preds_stock = np.full_like(rets_stock, np.nan, dtype=float)
    d_proj_random_baseline = _haar_baseline_projection_distance(m_dim, num_fact)

    # -------- Rolling loop --------
    W_prev: np.ndarray | None = None
    stability_dists: list[float] = []
    erank_series: list[float] = []
    principal_angles_series: list[np.ndarray] = []
    spectral_gap_series: list[float] = []
    d_proj_norm_series: list[float] = []
    d_proj_seeds_series: list[float] = []

    for t in range(window_len, T):
        rets_tr = rets_stock[t - window_len : t]
        Z_tr = Z[t - window_len : t]

        est = GrassmannianManifoldIPCAEstimator(
            num_assets=N,
            num_fact=num_fact,
            num_charact=m_dim,
            win_len=rets_tr.shape[0],
            shrinkage=shrinkage,
        )
        W_t, f_tr, _ = est.fit(
            data=[rets_tr, Z_tr],
            optimizer="ConjugateGradient",
            max_iterations=200,
            iter_tol=min_gradient_norm,
            verbosity=0,
            log_verbosity=0,
            initial_point=W_prev,
        )

        if W_prev is not None:
            angles = _principal_angles_desc(W_t, W_prev)
            principal_angles_series.append(angles)
            d_proj = _projection_distance_from_angles(angles)
            stability_dists.append(d_proj)
            d_proj_norm = (
                d_proj / d_proj_random_baseline
                if d_proj_random_baseline > 0
                else np.nan
            )
            d_proj_norm_series.append(float(d_proj_norm))

        if compute_seed_variance:
            est_rand = GrassmannianManifoldIPCAEstimator(
                num_assets=N,
                num_fact=num_fact,
                num_charact=m_dim,
                win_len=rets_tr.shape[0],
                shrinkage=shrinkage,
            )
            W_t_rand, _, _ = est_rand.fit(
                data=[rets_tr, Z_tr],
                optimizer="ConjugateGradient",
                max_iterations=200,
                iter_tol=min_gradient_norm,
                verbosity=0,
                log_verbosity=0,
                initial_point=None,
            )
            d_seeds = _projection_distance_from_gram(W_t, W_t_rand, num_fact)
            d_proj_seeds_series.append(d_seeds)

        W_prev = W_t

        Sigma_f = (f_tr.T @ f_tr) / f_tr.shape[0]
        eigvals = np.maximum(np.linalg.eigvalsh(Sigma_f), 0.0)
        erank_series.append(_erank_with_legacy_fallback(eigvals))
        spectral_gap_series.append(
            _spectral_gap_with_boundary_fallback(eigvals, num_fact)
        )

        f_next = f_tr.mean(axis=0)
        Lambda_t = Z[t] @ W_t
        preds_stock[t] = Lambda_t @ f_next

    # -------- OOS performance --------
    valid = ~np.isnan(preds_stock).any(axis=1)
    y_true = rets_stock[valid]
    y_pred = preds_stock[valid]

    r2_oos = 1.0 - (
        np.sum((y_true - y_pred) ** 2)
        / np.sum((y_true - y_true.mean()) ** 2)
    )

    sharpe_ew = _market_timing_sharpe(y_true, y_pred)
    if weighting in ("mktcap", "both"):
        mc_oos = mktcap_arr[valid]
        sharpe_mktcap = _mktcap_sharpe(y_true, y_pred, mc_oos)
    else:
        sharpe_mktcap = None

    if weighting == "equal":
        sharpe_primary = sharpe_ew
    elif weighting == "mktcap":
        sharpe_primary = sharpe_mktcap
    else:  # both
        sharpe_primary = sharpe_mktcap

    # -------- Aggregate diagnostics --------
    mean_stability = float(np.mean(stability_dists)) if stability_dists else np.nan

    if principal_angles_series:
        max_angles = [a[0] for a in principal_angles_series]
        mean_angles = [float(np.mean(a)) for a in principal_angles_series]
        mean_max_principal_angle = float(np.mean(max_angles))
        mean_mean_principal_angle = float(np.mean(mean_angles))
        std_mean_pa = (
            float(np.std(mean_angles, ddof=1)) if len(mean_angles) > 1 else np.nan
        )
        std_max_pa = (
            float(np.std(max_angles, ddof=1)) if len(max_angles) > 1 else np.nan
        )
        sr_mean_principal_angle = (
            mean_mean_principal_angle / std_mean_pa
            if std_mean_pa and std_mean_pa > 1e-12
            else np.nan
        )
        sr_max_principal_angle = (
            mean_max_principal_angle / std_max_pa
            if std_max_pa and std_max_pa > 1e-12
            else np.nan
        )
    else:
        mean_max_principal_angle = np.nan
        mean_mean_principal_angle = np.nan
        std_mean_pa = std_max_pa = np.nan
        sr_mean_principal_angle = sr_max_principal_angle = np.nan

    if len(stability_dists) >= 2:
        accel_series = list(np.diff(stability_dists))
        mean_geodesic_accel = float(np.mean(np.abs(accel_series)))
    else:
        accel_series = []
        mean_geodesic_accel = np.nan

    mean_erank = float(np.mean(erank_series)) if erank_series else np.nan
    erank_collapse_frac = (
        float(np.mean(np.array(erank_series) < 0.5 * num_fact))
        if erank_series
        else np.nan
    )
    mean_spectral_gap = (
        float(np.mean(spectral_gap_series)) if spectral_gap_series else np.nan
    )
    mean_d_proj_norm = (
        float(np.mean(d_proj_norm_series)) if d_proj_norm_series else np.nan
    )
    mean_d_proj_seeds = (
        float(np.mean(d_proj_seeds_series)) if d_proj_seeds_series else np.nan
    )
    d_proj_seeds_flat_frac = (
        float(
            np.mean(
                np.array(d_proj_seeds_series) > 0.5 * np.sqrt(2 * num_fact)
            )
        )
        if d_proj_seeds_series
        else np.nan
    )

    if verbose:
        _print_summary(
            r2_oos=r2_oos,
            sharpe_ew=sharpe_ew,
            sharpe_mktcap=sharpe_mktcap,
            weighting=weighting,
            mean_stability=mean_stability,
            mean_max_principal_angle=mean_max_principal_angle,
            mean_mean_principal_angle=mean_mean_principal_angle,
            sr_mean_principal_angle=sr_mean_principal_angle,
            sr_max_principal_angle=sr_max_principal_angle,
            mean_geodesic_accel=mean_geodesic_accel,
            mean_erank=mean_erank,
            erank_collapse_frac=erank_collapse_frac,
            num_fact=num_fact,
            mean_spectral_gap=mean_spectral_gap,
            d_proj_random_baseline=d_proj_random_baseline,
            m_dim=m_dim,
            mean_d_proj_norm=mean_d_proj_norm,
            compute_seed_variance=compute_seed_variance,
            mean_d_proj_seeds=mean_d_proj_seeds,
            d_proj_seeds_flat_frac=d_proj_seeds_flat_frac,
        )

    results: dict = {
        "r2_oos": r2_oos,
        "sharpe": sharpe_primary,
        "mean_subspace_stability": mean_stability,
        "stability_dist_series": stability_dists,
        "mean_max_principal_angle": mean_max_principal_angle,
        "mean_mean_principal_angle": mean_mean_principal_angle,
        "principal_angles_series": principal_angles_series,
        "std_mean_principal_angle": std_mean_pa,
        "std_max_principal_angle": std_max_pa,
        "sr_mean_principal_angle": sr_mean_principal_angle,
        "sr_max_principal_angle": sr_max_principal_angle,
        "mean_geodesic_accel": mean_geodesic_accel,
        "geodesic_accel_series": accel_series,
        "mean_erank": mean_erank,
        "erank_series": erank_series,
        "erank_collapse_frac": erank_collapse_frac,
        "mean_spectral_gap": mean_spectral_gap,
        "spectral_gap_series": spectral_gap_series,
        "mean_d_proj_norm": mean_d_proj_norm,
        "d_proj_norm_series": d_proj_norm_series,
        "d_proj_random_baseline": d_proj_random_baseline,
        "mean_d_proj_seeds": mean_d_proj_seeds,
        "d_proj_seeds_series": d_proj_seeds_series,
        "d_proj_seeds_flat_frac": d_proj_seeds_flat_frac,
    }
    if weighting in ("mktcap", "both"):
        results["sharpe_ew"] = sharpe_ew
        results["sharpe_mktcap"] = sharpe_mktcap
    return results


def _print_summary(**kw) -> None:
    print("=" * 65)
    print("  Grassmannian-IPCA Backtest Results")
    print("=" * 65)
    print("\n-- Performance --")
    print(f"  OOS R^2                              : {kw['r2_oos']:.4f}")
    if kw["weighting"] == "equal":
        print(f"  Market-timing Sharpe (annualised)    : {kw['sharpe_ew']:.3f}")
    else:
        print(f"  Sharpe (equal-weighted)              : {kw['sharpe_ew']:.3f}")
        print(f"  Sharpe (mktcap-weighted)             : {kw['sharpe_mktcap']:.3f}")
    print(f"\n-- Subspace stability --")
    print(f"  Mean d_proj                          : {kw['mean_stability']:.4f}")
    print(f"  Mean max  principal angle (rad)      : {kw['mean_max_principal_angle']:.4f}")
    print(f"  Mean mean principal angle (rad)      : {kw['mean_mean_principal_angle']:.4f}")
    print(f"  SR of mean principal angle            : {kw['sr_mean_principal_angle']:.4f}")
    print(f"  SR of max  principal angle            : {kw['sr_max_principal_angle']:.4f}")
    print(f"  Mean |delta d_proj|                  : {kw['mean_geodesic_accel']:.4f}")
    print(f"\n-- Spectrum --")
    print(f"  Mean erank                           : {kw['mean_erank']:.4f}  (max = {kw['num_fact']})")
    print(f"  erank collapse fraction              : {kw['erank_collapse_frac']:.4f}")
    print(f"  Mean spectral gap                    : {kw['mean_spectral_gap']:.4f}")
    print(f"\n-- Ambient-normalized d_proj --")
    print(f"  Haar random baseline                 : {kw['d_proj_random_baseline']:.4f}  (Gr({kw['m_dim']},{kw['num_fact']}))")
    print(f"  Mean d_proj_norm                     : {kw['mean_d_proj_norm']:.4f}")
    if kw["compute_seed_variance"]:
        print(f"\n-- Between-seed d_proj --")
        print(f"  Mean d_proj (warm vs random init)    : {kw['mean_d_proj_seeds']:.4f}  (max = {np.sqrt(2*kw['num_fact']):.4f})")
        print(f"  Flat-Hessian fraction                : {kw['d_proj_seeds_flat_frac']:.4f}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Backward-compatibility shims
# ---------------------------------------------------------------------------


def run_ipca_grassmannian_v2(*args, **kwargs) -> dict:
    """Legacy entry point for the equal-weighted backtest.

    Forwards to :func:`run_backtest` with ``weighting="equal"`` and
    ``impute="mean"`` (the v2 defaults). Kept for backward compatibility
    with existing call sites; new code should call :func:`run_backtest`
    directly.
    """
    kwargs.setdefault("weighting", "equal")
    kwargs.setdefault("impute", "mean")
    return run_backtest(*args, **kwargs)


def run_ipca_grassmannian_mktcap(*args, **kwargs) -> dict:
    """Legacy entry point for the value-weighted backtest.

    Forwards to :func:`run_backtest` with ``weighting="mktcap"``. Also
    fixes a latent bug in the original implementation, which passed the
    wrong kwarg name (``min_gradient_norm``) to the estimator's
    ``fit(iter_tol=...)``.
    """
    kwargs.setdefault("weighting", "mktcap")
    kwargs.setdefault("impute", "mean")
    return run_backtest(*args, **kwargs)
