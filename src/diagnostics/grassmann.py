"""Grassmannian geometry diagnostics for fitted return subspaces.

Implements the subspace-stability diagnostics from §4 of
"From Parameters to Subspaces: The Geometry of Complexity in Asset Pricing".

All functions take orthonormal bases ``U`` of shape ``(n, k)`` representing
points on the Grassmann manifold :math:`Gr(k, n)`. Non-orthonormal inputs are
orthonormalized via QR.
"""

from __future__ import annotations

import numpy as np


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis spanning the same column space as ``basis``.

    Args:
        basis: Array of shape ``(n, k)`` with ``k <= n``.

    Returns:
        Array of shape ``(n, k)`` with orthonormal columns.
    """
    q, _ = np.linalg.qr(basis)
    return q


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces.

    For orthonormal bases :math:`U_a, U_b \\in \\mathbb{R}^{n \\times k}`, the
    principal angles :math:`\\psi_1 \\le \\dots \\le \\psi_k \\in [0, \\pi/2]`
    are defined by :math:`\\cos\\psi_j = \\sigma_j(U_a^\\top U_b)` where
    :math:`\\sigma_j` denotes the j-th singular value (paper eq. just after
    §1, "Principal-angle diagnostics decompose that motion across directions").

    Args:
        basis_a: Subspace basis of shape ``(n, k_a)``.
        basis_b: Subspace basis of shape ``(n, k_b)``. ``k_b`` may differ from
            ``k_a``; the returned array has length ``min(k_a, k_b)``.

    Returns:
        Array of principal angles in radians, sorted ascending.
    """
    qa = _orthonormalize(basis_a)
    qb = _orthonormalize(basis_b)
    cos_angles = np.linalg.svd(qa.T @ qb, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    return np.sort(angles)


def grassmann_distance(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    metric: str = "geodesic",
) -> float:
    """Distance between two equal-dimension subspaces on the Grassmann manifold.

    Args:
        basis_a: Subspace basis of shape ``(n, k)``.
        basis_b: Subspace basis of shape ``(n, k)``.
        metric: One of ``"geodesic"`` (arc-length, :math:`\\sqrt{\\sum \\psi_j^2}`),
            ``"projection"`` (chordal, :math:`\\sqrt{\\sum \\sin^2 \\psi_j}`),
            or ``"chordal_2"`` (Frobenius distance between projectors divided by
            :math:`\\sqrt{2}`, equal to ``"projection"``).

    Returns:
        Non-negative scalar distance.

    Raises:
        ValueError: If ``basis_a`` and ``basis_b`` have different column counts,
            or if ``metric`` is unknown.
    """
    if basis_a.shape[1] != basis_b.shape[1]:
        raise ValueError(
            f"grassmann_distance requires equal-dim subspaces, "
            f"got {basis_a.shape[1]} and {basis_b.shape[1]}"
        )
    angles = principal_angles(basis_a, basis_b)
    if metric == "geodesic":
        return float(np.sqrt(np.sum(angles ** 2)))
    if metric in ("projection", "chordal_2"):
        return float(np.sqrt(np.sum(np.sin(angles) ** 2)))
    raise ValueError(f"Unknown metric {metric!r}")


def projection_distance(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Projection (Frobenius) distance between two subspaces.

    Computes :math:`\\|P_a - P_b\\|_F / \\sqrt{2}` where
    :math:`P = U U^\\top` is the orthogonal projector onto the column span. For
    equal-dimension subspaces this equals
    :math:`\\sqrt{\\sum \\sin^2 \\psi_j}` (the chordal Grassmann distance).

    Unlike :func:`grassmann_distance`, this works for unequal-dimension
    subspaces.

    Args:
        basis_a: Subspace basis of shape ``(n, k_a)``.
        basis_b: Subspace basis of shape ``(n, k_b)``.

    Returns:
        Non-negative scalar distance.
    """
    qa = _orthonormalize(basis_a)
    qb = _orthonormalize(basis_b)
    proj_a = qa @ qa.T
    proj_b = qb @ qb.T
    return float(np.linalg.norm(proj_a - proj_b, ord="fro") / np.sqrt(2.0))


def anisotropy_ratio(angles: np.ndarray) -> float:
    """Anisotropy ratio :math:`\\rho_\\psi = \\bar\\psi / \\psi_{\\max}`.

    Defined in §1 of the paper as the ratio of mean to max principal angle.
    Values near 1 indicate subspace motion spread diffusely across many
    directions; values near 0 indicate motion concentrated in a small number
    of directions (the "structured" case associated with regime separators).

    Args:
        angles: Principal angles in radians, e.g. from
            :func:`principal_angles`. Must be non-empty.

    Returns:
        Ratio in ``[0, 1]``. Returns ``0.0`` when the max angle is zero
        (identical subspaces).
    """
    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        raise ValueError("anisotropy_ratio requires at least one angle")
    psi_max = float(np.max(angles))
    if psi_max == 0.0:
        return 0.0
    return float(np.mean(angles) / psi_max)
