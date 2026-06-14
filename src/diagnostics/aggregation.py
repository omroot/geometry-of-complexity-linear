"""Aggregate per-window geometric diagnostics for a rolling fit.

Takes the ``fitted_bases`` and ``factor_returns`` lists produced by
:func:`src.models.rolling_fit.rolling_fit` and reduces them to scalar
cross-section means matching the schema the legacy broad-grid summary CSVs
expose. Composes primitives already in :mod:`src.diagnostics`
(:func:`principal_angles`, :func:`anisotropy_ratio`,
:func:`effective_rank`, :func:`spectral_gap`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.diagnostics.grassmann import (
    anisotropy_ratio,
    principal_angles,
    projection_distance,
)
from src.diagnostics.spectrum import effective_rank


@dataclass
class WindowDiagnostics:
    """Cross-window means of the geometric diagnostics from §4.

    Field names mirror the legacy ``synthetic_v3_sweep_*_summary.csv`` schema
    (with the ``avg_`` prefix dropped) so downstream plotting code can rename
    or read directly.

    Attributes:
        mean_subspace_stability: Mean rolling-window projection distance
            ``d_proj`` between consecutive fitted bases.
        mean_max_principal_angle: Mean of ``max(principal_angles)`` between
            consecutive fitted bases.
        mean_mean_principal_angle: Mean of ``mean(principal_angles)``
            between consecutive fitted bases.
        mean_anisotropy: Mean of the per-window anisotropy ratio
            ``mean(angles) / max(angles)``.
        mean_geodesic_accel: Mean absolute first-difference of consecutive
            ``d_proj`` values: ``mean(|d_proj_t - d_proj_{t-1}|)``.
            Matches VoC's ``avg_geodesic_accel`` column in all_rff_results.csv.
        mean_erank: Mean effective rank of the per-window factor covariance
            ``F F^T / T_train``.
        mean_spectral_gap: Mean of ``lambda_k / lambda_1`` over the per-window
            factor covariance. (For a fitted rank-``k`` factor model there is
            no ``lambda_{k+1}`` noise eigenvalue, so the standard
            ``(lambda_k - lambda_{k+1}) / lambda_1`` definition reduces to
            ``lambda_k / lambda_1``, matching the legacy summary CSV column.)
        n_windows: Number of rolling windows contributing to the means.
    """

    mean_subspace_stability: float
    mean_max_principal_angle: float
    mean_mean_principal_angle: float
    mean_anisotropy: float
    mean_geodesic_accel: float
    mean_erank: float
    mean_spectral_gap: float
    n_windows: int


def _factor_covariance_eigvals(factor_history: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending, clipped at 0) of Sigma_f = F F^T / T_train.

    Args:
        factor_history: Factor matrix of shape ``(k, T_train)``.

    Returns:
        Length-k array of non-negative eigenvalues in descending order.
    """
    T_train = max(int(factor_history.shape[1]), 1)
    Sigma_f = (factor_history @ factor_history.T) / T_train
    eigvals = np.linalg.eigvalsh(Sigma_f)
    return np.sort(np.clip(eigvals, 0.0, None))[::-1]


def aggregate_window_diagnostics(
    fitted_bases: list[np.ndarray],
    factor_returns: list[np.ndarray] | None = None,
) -> WindowDiagnostics:
    """Reduce a sequence of fitted bases/factors to scalar means.

    Args:
        fitted_bases: List of orthonormal bases ``W`` (shape ``(m_eff, k)``)
            in chronological order.
        factor_returns: Optional list of per-window factor histories ``F``
            (shape ``(k, T_train)``). Required for ``mean_erank`` and
            ``mean_spectral_gap``; NaN-filled when omitted.

    Returns:
        :class:`WindowDiagnostics`. Distance / angle fields require at least
        two bases; when fewer are supplied they fall back to NaN.
    """
    n = len(fitted_bases)
    if n == 0:
        return WindowDiagnostics(
            mean_subspace_stability=float("nan"),
            mean_max_principal_angle=float("nan"),
            mean_mean_principal_angle=float("nan"),
            mean_anisotropy=float("nan"),
            mean_geodesic_accel=float("nan"),
            mean_erank=float("nan"),
            mean_spectral_gap=float("nan"),
            n_windows=0,
        )

    d_proj_vals: list[float] = []
    angle_max_vals: list[float] = []
    angle_mean_vals: list[float] = []
    anisotropy_vals: list[float] = []
    for prev, curr in zip(fitted_bases[:-1], fitted_bases[1:]):
        if prev.shape != curr.shape:
            continue
        d_proj_vals.append(projection_distance(curr, prev))
        angles = principal_angles(curr, prev)
        if angles.size == 0:
            continue
        angle_max_vals.append(float(np.max(angles)))
        angle_mean_vals.append(float(np.mean(angles)))
        anisotropy_vals.append(float(anisotropy_ratio(angles)))

    erank_vals: list[float] = []
    spectral_gap_vals: list[float] = []
    if factor_returns is not None:
        for factor_history in factor_returns:
            eigvals = _factor_covariance_eigvals(factor_history)
            if eigvals.size == 0 or float(eigvals[0]) <= 0:
                continue
            erank_vals.append(effective_rank(eigvals))
            # Legacy convention for a rank-k factor model: lambda_k / lambda_1.
            spectral_gap_vals.append(float(eigvals[-1] / eigvals[0]))

    # Geodesic acceleration = mean |Δd_proj| across consecutive windows.
    # Matches VoC avg_geodesic_accel: measures how much the subspace speed changes.
    if len(d_proj_vals) >= 2:
        accel_vals = [abs(d_proj_vals[i + 1] - d_proj_vals[i])
                      for i in range(len(d_proj_vals) - 1)]
        mean_geodesic_accel = float(np.mean(accel_vals))
    else:
        mean_geodesic_accel = float("nan")

    return WindowDiagnostics(
        mean_subspace_stability=_mean_or_nan(d_proj_vals),
        mean_max_principal_angle=_mean_or_nan(angle_max_vals),
        mean_mean_principal_angle=_mean_or_nan(angle_mean_vals),
        mean_anisotropy=_mean_or_nan(anisotropy_vals),
        mean_geodesic_accel=mean_geodesic_accel,
        mean_erank=_mean_or_nan(erank_vals),
        mean_spectral_gap=_mean_or_nan(spectral_gap_vals),
        n_windows=n,
    )


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")
