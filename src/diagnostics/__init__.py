"""Grassmannian diagnostics for the virtue-of-complexity geometric framework.

Reference implementations of the diagnostics introduced in
"From Parameters to Subspaces: The Geometry of Complexity in Asset Pricing"
(Deep, Lesniewski, Missaoui, Pakala). Each function corresponds to a
quantity defined in §4 (intrinsic dimensionality and subspace stability) or
§6 (curvature and eigenvalue gaps) of the paper.

This module is independent of the IPCA estimator and the rolling-window
backtest harness. It operates directly on orthonormal subspace bases,
eigenvalue vectors, or PSD covariance matrices.
"""

from .curvature import eigengap_curvature, geodesic_acceleration
from .grassmann import (
    anisotropy_ratio,
    grassmann_distance,
    principal_angles,
    projection_distance,
)
from .spectrum import effective_rank, intrinsic_dimension, spectral_gap

__all__ = [
    "anisotropy_ratio",
    "effective_rank",
    "eigengap_curvature",
    "geodesic_acceleration",
    "grassmann_distance",
    "intrinsic_dimension",
    "principal_angles",
    "projection_distance",
    "spectral_gap",
]
