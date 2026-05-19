"""Curvature diagnostics for Grassmannian return-subspace estimation.

Implements two complementary notions of curvature discussed in §6 of
"From Parameters to Subspaces: The Geometry of Complexity in Asset Pricing":

1. **Eigengap curvature**: at a fitted subspace, the local curvature of the
   Grassmann projection objective in the j-th direction is controlled by the
   gap :math:`\\lambda_j - \\lambda_{k+1}` between the j-th retained
   eigenvalue and the largest noise eigenvalue. Flat (near-zero) gaps mark
   weakly identified directions that rotate under perturbation.

2. **Geodesic acceleration**: along a trajectory of rolling-window subspaces,
   the discrete second difference on the Grassmann manifold measures how
   "kinked" the path is. Useful for distinguishing structured regime shifts
   from diffuse noise-driven rotation.
"""

from __future__ import annotations

import numpy as np

from .grassmann import projection_distance
from .spectrum import _eigenvalues_from_input


def eigengap_curvature(
    eigenvalues_or_matrix: np.ndarray,
    k: int,
    normalize: bool = True,
) -> np.ndarray:
    """Per-direction eigengap curvature at a signal/noise boundary.

    Returns the vector :math:`(\\lambda_j - \\lambda_{k+1})_{j=1}^{k}`,
    optionally normalized by :math:`\\lambda_1`. Per §6, this vector
    approximates the local curvature of the Grassmann projection objective
    along the j-th principal direction. Small entries flag weakly identified
    directions whose fitted estimates will rotate under sampling variation;
    large entries flag well-identified directions.

    Args:
        eigenvalues_or_matrix: Either a 1-D array of (non-negative) eigenvalues
            or a 2-D symmetric PSD matrix. Eigenvalues are sorted descending
            internally.
        k: Signal-dimension boundary, ``1 <= k < n``.
        normalize: If True, divides by :math:`\\lambda_1` so all entries lie
            in ``[0, 1]``.

    Returns:
        Array of length ``k`` of non-negative per-direction curvatures, sorted
        descending (largest gap first).

    Raises:
        ValueError: If ``k`` is out of range.
    """
    eigenvalues = _eigenvalues_from_input(eigenvalues_or_matrix)
    n = eigenvalues.size
    if not (1 <= k < n):
        raise ValueError(f"k must satisfy 1 <= k < {n}, got {k}")
    noise_floor = eigenvalues[k]
    gaps = eigenvalues[:k] - noise_floor
    gaps = np.maximum(gaps, 0.0)
    if not normalize:
        return gaps
    if eigenvalues[0] == 0.0:
        return np.zeros_like(gaps)
    return np.clip(gaps / eigenvalues[0], 0.0, 1.0)


def geodesic_acceleration(
    basis_prev: np.ndarray,
    basis_curr: np.ndarray,
    basis_next: np.ndarray,
) -> float:
    """Discrete geodesic acceleration of a subspace trajectory.

    Treats the projector representation :math:`P_t = U_t U_t^\\top` as living
    in a flat Frobenius inner-product space and computes the chord-normalized
    second difference

    .. math:: a_t = \\sqrt{2\\, d(V_{t-1}, V_t)^2 + 2\\, d(V_t, V_{t+1})^2
        - d(V_{t-1}, V_{t+1})^2},

    where :math:`d` is the projection (chordal) Grassmann distance
    :func:`~.grassmann.projection_distance`. This equals
    :math:`\\|P_{t+1} - 2 P_t + P_{t-1}\\|_F / \\sqrt{2}`. For a straight
    (geodesic) path through the three samples, ``a_t`` is zero; for a sharp
    regime shift it is large. Per §6, large geodesic acceleration co-occurs
    with effective-rank collapse and spectral-gap narrowing in the weakly
    identified case.

    Args:
        basis_prev: Subspace basis at time ``t-1``, shape ``(n, k)``.
        basis_curr: Subspace basis at time ``t``, shape ``(n, k)``.
        basis_next: Subspace basis at time ``t+1``, shape ``(n, k)``.

    Returns:
        Non-negative scalar.
    """
    d_back = projection_distance(basis_prev, basis_curr)
    d_fwd = projection_distance(basis_curr, basis_next)
    d_skip = projection_distance(basis_prev, basis_next)
    accel_sq = 2.0 * (d_back ** 2 + d_fwd ** 2) - d_skip ** 2
    return float(np.sqrt(max(accel_sq, 0.0)))
