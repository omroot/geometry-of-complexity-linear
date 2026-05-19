"""Spectral diagnostics for fitted factor covariance matrices.

Implements the intrinsic-dimensionality and spectral-separation diagnostics
from §4 and §6 of "From Parameters to Subspaces: The Geometry of Complexity
in Asset Pricing".
"""

from __future__ import annotations

import numpy as np


def _eigenvalues_from_input(
    eigenvalues_or_matrix: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """Coerce input to a non-negative, descending-sorted eigenvalue vector.

    Args:
        eigenvalues_or_matrix: Either a 1-D array of eigenvalues, or a 2-D
            symmetric PSD matrix whose eigenvalues will be computed.
        tol: Eigenvalues below this threshold (after taking absolute value to
            absorb numerical noise) are floored to zero.

    Returns:
        1-D array of eigenvalues, non-negative, sorted in descending order.
    """
    arr = np.asarray(eigenvalues_or_matrix, dtype=float)
    if arr.ndim == 1:
        eigenvalues = arr
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        eigenvalues = np.linalg.eigvalsh((arr + arr.T) / 2.0)
    else:
        raise ValueError(
            "spectrum input must be a 1-D eigenvalue array or a square 2-D "
            f"matrix; got shape {arr.shape}"
        )
    eigenvalues = np.where(np.abs(eigenvalues) < tol, 0.0, eigenvalues)
    if np.any(eigenvalues < 0):
        raise ValueError(
            "eigenvalues must be non-negative (input is expected to be PSD)"
        )
    return np.sort(eigenvalues)[::-1]


def effective_rank(eigenvalues_or_matrix: np.ndarray) -> float:
    """Entropic effective rank :math:`\\mathrm{erank}(\\Sigma) = \\exp(H(p))`.

    Defined in §4.1, eq. (effective-rank) of the paper. With
    :math:`p_i = \\lambda_i / \\mathrm{tr}(\\Sigma)` and
    :math:`H(p) = -\\sum_i p_i \\log p_i`, the effective rank is

    .. math:: \\mathrm{erank}(\\Sigma) = \\exp(H(p)).

    It measures how many directions are effectively active in the covariance
    and equals the integer rank for orthogonal projectors with equal nonzero
    eigenvalues. The paper uses this quantity as the empirical proxy for the
    intrinsic dimension of a fitted return representation; see
    :func:`intrinsic_dimension`.

    Args:
        eigenvalues_or_matrix: Either a 1-D array of (non-negative) eigenvalues
            or a 2-D symmetric PSD matrix.

    Returns:
        Effective rank in ``[0, n]`` where ``n`` is the number of eigenvalues.
        Returns ``0.0`` if all eigenvalues are zero.
    """
    eigenvalues = _eigenvalues_from_input(eigenvalues_or_matrix)
    total = float(np.sum(eigenvalues))
    if total == 0.0:
        return 0.0
    probabilities = eigenvalues / total
    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return float(np.exp(entropy))


def intrinsic_dimension(eigenvalues_or_matrix: np.ndarray) -> float:
    """Intrinsic dimension of a fitted return representation.

    Per §4.1 of the paper, the intrinsic dimension is *defined* as the
    effective rank of the estimated factor covariance matrix. This function
    is a named alias for :func:`effective_rank`.

    Args:
        eigenvalues_or_matrix: Eigenvalues or PSD matrix; see
            :func:`effective_rank`.

    Returns:
        Effective rank value.
    """
    return effective_rank(eigenvalues_or_matrix)


def spectral_gap(
    eigenvalues_or_matrix: np.ndarray,
    k: int,
    normalize: bool = True,
) -> float:
    """Spectral-gap diagnostic at signal/noise boundary ``k``.

    Computes :math:`(\\lambda_k - \\lambda_{k+1}) / \\lambda_1` (or the raw
    difference when ``normalize=False``). Discussed in §6: large gaps indicate
    well-identified signal directions; collapsing gaps indicate weakly
    identified directions that rotate under sampling variation.

    Args:
        eigenvalues_or_matrix: Either a 1-D array of eigenvalues or a 2-D
            symmetric PSD matrix. Eigenvalues are sorted descending internally.
        k: Signal-dimension boundary, ``1 <= k < n``. The gap is taken between
            the ``k``-th and ``(k+1)``-th eigenvalues (1-indexed).
        normalize: If True, divides by :math:`\\lambda_1` and clips to
            ``[0, 1]``; if False, returns the raw difference.

    Returns:
        Non-negative scalar.

    Raises:
        ValueError: If ``k`` is out of range.
    """
    eigenvalues = _eigenvalues_from_input(eigenvalues_or_matrix)
    n = eigenvalues.size
    if not (1 <= k < n):
        raise ValueError(f"k must satisfy 1 <= k < {n}, got {k}")
    gap = float(eigenvalues[k - 1] - eigenvalues[k])
    if not normalize:
        return gap
    if eigenvalues[0] == 0.0:
        return 0.0
    ratio = gap / eigenvalues[0]
    return float(np.clip(ratio, 0.0, 1.0))
