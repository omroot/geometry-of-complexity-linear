"""Unit tests for src/diagnostics/.

Each diagnostic is exercised on inputs whose answer is known by construction
(identical subspaces, orthogonal subspaces, rotated subspaces with known
angles, projectors with equal eigenvalues, signal+noise spectra, etc.).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.diagnostics import (
    anisotropy_ratio,
    effective_rank,
    eigengap_curvature,
    geodesic_acceleration,
    grassmann_distance,
    intrinsic_dimension,
    principal_angles,
    projection_distance,
    spectral_gap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_2d(theta: float) -> np.ndarray:
    return np.array(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta),  math.cos(theta)]]
    )


def _two_subspaces_with_known_angles(angles: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Build two k-dim subspaces in R^{2k} with prescribed principal angles.

    Construction: in R^{2k}, take ``basis_a`` = first k axes; ``basis_b``
    spans the columns ``(cos(psi_j) e_j + sin(psi_j) e_{k+j})``. Then the
    principal angles between span(basis_a) and span(basis_b) are exactly
    the prescribed ``angles``.
    """
    k = len(angles)
    n = 2 * k
    basis_a = np.zeros((n, k))
    basis_b = np.zeros((n, k))
    for j, psi in enumerate(angles):
        basis_a[j, j] = 1.0
        basis_b[j, j] = math.cos(psi)
        basis_b[k + j, j] = math.sin(psi)
    return basis_a, basis_b


# ---------------------------------------------------------------------------
# Grassmann diagnostics
# ---------------------------------------------------------------------------


class TestPrincipalAngles:
    def test_identical_subspaces_have_zero_angles(self) -> None:
        rng = np.random.default_rng(0)
        basis = np.linalg.qr(rng.standard_normal((10, 3)))[0]
        angles = principal_angles(basis, basis)
        np.testing.assert_allclose(angles, np.zeros(3), atol=1e-10)

    def test_orthogonal_subspaces_have_right_angle(self) -> None:
        basis_a = np.eye(6)[:, :3]
        basis_b = np.eye(6)[:, 3:]
        angles = principal_angles(basis_a, basis_b)
        np.testing.assert_allclose(angles, np.full(3, math.pi / 2), atol=1e-10)

    def test_known_angles_recovered(self) -> None:
        prescribed = [0.1, 0.4, 0.9]
        basis_a, basis_b = _two_subspaces_with_known_angles(prescribed)
        angles = principal_angles(basis_a, basis_b)
        np.testing.assert_allclose(angles, sorted(prescribed), atol=1e-10)

    def test_unequal_k_truncates_to_min(self) -> None:
        rng = np.random.default_rng(1)
        basis_a = np.linalg.qr(rng.standard_normal((8, 3)))[0]
        basis_b = np.linalg.qr(rng.standard_normal((8, 5)))[0]
        angles = principal_angles(basis_a, basis_b)
        assert angles.shape == (3,)

    def test_invariant_to_basis_rotation(self) -> None:
        rng = np.random.default_rng(2)
        basis = np.linalg.qr(rng.standard_normal((6, 3)))[0]
        rotation = np.linalg.qr(rng.standard_normal((3, 3)))[0]
        basis_rotated = basis @ rotation
        prescribed = [0.2, 0.5]
        basis_a, basis_b = _two_subspaces_with_known_angles(prescribed)
        rotation_a = np.linalg.qr(np.random.default_rng(3).standard_normal((2, 2)))[0]
        rotation_b = np.linalg.qr(np.random.default_rng(4).standard_normal((2, 2)))[0]
        angles_direct = principal_angles(basis_a, basis_b)
        angles_rotated = principal_angles(basis_a @ rotation_a, basis_b @ rotation_b)
        np.testing.assert_allclose(angles_direct, angles_rotated, atol=1e-10)


class TestGrassmannDistance:
    def test_zero_for_identical_subspaces(self) -> None:
        # arccos amplifies float noise near 1, so the geodesic metric loses
        # ~7 digits relative to the projection metric on identical inputs.
        rng = np.random.default_rng(10)
        basis = np.linalg.qr(rng.standard_normal((8, 4)))[0]
        assert grassmann_distance(basis, basis) == pytest.approx(0.0, abs=1e-6)
        assert grassmann_distance(basis, basis, metric="projection") == pytest.approx(
            0.0, abs=1e-6
        )

    def test_geodesic_matches_l2_of_angles(self) -> None:
        prescribed = [0.1, 0.3, 1.2]
        basis_a, basis_b = _two_subspaces_with_known_angles(prescribed)
        expected = math.sqrt(sum(a * a for a in prescribed))
        assert grassmann_distance(basis_a, basis_b, metric="geodesic") == pytest.approx(
            expected, abs=1e-10
        )

    def test_projection_matches_sin_of_angles(self) -> None:
        prescribed = [0.1, 0.3, 1.2]
        basis_a, basis_b = _two_subspaces_with_known_angles(prescribed)
        expected = math.sqrt(sum(math.sin(a) ** 2 for a in prescribed))
        assert grassmann_distance(basis_a, basis_b, metric="projection") == pytest.approx(
            expected, abs=1e-10
        )

    def test_unknown_metric_raises(self) -> None:
        basis_a, basis_b = _two_subspaces_with_known_angles([0.1])
        with pytest.raises(ValueError, match="Unknown metric"):
            grassmann_distance(basis_a, basis_b, metric="nope")

    def test_dimension_mismatch_raises(self) -> None:
        rng = np.random.default_rng(11)
        basis_a = np.linalg.qr(rng.standard_normal((8, 3)))[0]
        basis_b = np.linalg.qr(rng.standard_normal((8, 4)))[0]
        with pytest.raises(ValueError, match="equal-dim"):
            grassmann_distance(basis_a, basis_b)


class TestProjectionDistance:
    def test_zero_for_identical_subspaces(self) -> None:
        rng = np.random.default_rng(20)
        basis = np.linalg.qr(rng.standard_normal((8, 3)))[0]
        assert projection_distance(basis, basis) == pytest.approx(0.0, abs=1e-10)

    def test_matches_chordal_grassmann_distance(self) -> None:
        prescribed = [0.2, 0.6, 1.1]
        basis_a, basis_b = _two_subspaces_with_known_angles(prescribed)
        chordal = grassmann_distance(basis_a, basis_b, metric="projection")
        assert projection_distance(basis_a, basis_b) == pytest.approx(chordal, abs=1e-10)

    def test_handles_unequal_dim(self) -> None:
        rng = np.random.default_rng(21)
        basis_a = np.linalg.qr(rng.standard_normal((8, 2)))[0]
        basis_b = np.linalg.qr(rng.standard_normal((8, 5)))[0]
        assert projection_distance(basis_a, basis_b) >= 0.0


class TestAnisotropyRatio:
    def test_uniform_angles_give_one(self) -> None:
        assert anisotropy_ratio(np.array([0.4, 0.4, 0.4])) == pytest.approx(1.0)

    def test_single_concentrated_direction_gives_near_zero(self) -> None:
        angles = np.array([1e-8, 1e-8, 1e-8, 1.0])
        ratio = anisotropy_ratio(angles)
        assert 0.0 <= ratio < 0.3

    def test_zero_max_returns_zero(self) -> None:
        assert anisotropy_ratio(np.zeros(4)) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            anisotropy_ratio(np.array([]))


# ---------------------------------------------------------------------------
# Spectrum diagnostics
# ---------------------------------------------------------------------------


class TestEffectiveRank:
    def test_equal_eigenvalues_give_full_rank(self) -> None:
        eigenvalues = np.ones(5)
        assert effective_rank(eigenvalues) == pytest.approx(5.0, abs=1e-10)

    def test_single_nonzero_eigenvalue_gives_one(self) -> None:
        eigenvalues = np.array([1.0, 0.0, 0.0, 0.0])
        assert effective_rank(eigenvalues) == pytest.approx(1.0, abs=1e-10)

    def test_two_equal_nonzero_with_zeros_gives_two(self) -> None:
        eigenvalues = np.array([1.0, 1.0, 0.0, 0.0])
        assert effective_rank(eigenvalues) == pytest.approx(2.0, abs=1e-10)

    def test_accepts_matrix_input(self) -> None:
        diag = np.diag([3.0, 1.0, 0.5])
        rng = np.random.default_rng(30)
        rotation = np.linalg.qr(rng.standard_normal((3, 3)))[0]
        matrix = rotation @ diag @ rotation.T
        from_matrix = effective_rank(matrix)
        from_eigenvalues = effective_rank(np.array([3.0, 1.0, 0.5]))
        assert from_matrix == pytest.approx(from_eigenvalues, abs=1e-8)

    def test_all_zero_eigenvalues_returns_zero(self) -> None:
        assert effective_rank(np.zeros(4)) == 0.0

    def test_negative_eigenvalues_raise(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            effective_rank(np.array([1.0, -0.1, 0.5]))

    def test_intrinsic_dimension_is_alias(self) -> None:
        eigenvalues = np.array([2.0, 1.0, 0.1])
        assert intrinsic_dimension(eigenvalues) == effective_rank(eigenvalues)


class TestSpectralGap:
    def test_clear_signal_noise_gap(self) -> None:
        eigenvalues = np.array([10.0, 8.0, 0.1, 0.05])
        gap = spectral_gap(eigenvalues, k=2, normalize=False)
        assert gap == pytest.approx(8.0 - 0.1)

    def test_normalization_clips_to_unit_interval(self) -> None:
        eigenvalues = np.array([10.0, 8.0, 0.1, 0.05])
        gap = spectral_gap(eigenvalues, k=2, normalize=True)
        assert gap == pytest.approx((8.0 - 0.1) / 10.0)
        assert 0.0 <= gap <= 1.0

    def test_collapsed_gap(self) -> None:
        eigenvalues = np.array([5.0, 5.0, 5.0, 5.0])
        assert spectral_gap(eigenvalues, k=2) == pytest.approx(0.0, abs=1e-12)

    def test_out_of_range_k_raises(self) -> None:
        eigenvalues = np.array([2.0, 1.0])
        with pytest.raises(ValueError, match="k must satisfy"):
            spectral_gap(eigenvalues, k=2)
        with pytest.raises(ValueError, match="k must satisfy"):
            spectral_gap(eigenvalues, k=0)


# ---------------------------------------------------------------------------
# Curvature diagnostics
# ---------------------------------------------------------------------------


class TestEigengapCurvature:
    def test_returns_per_direction_gaps_descending(self) -> None:
        eigenvalues = np.array([10.0, 6.0, 2.0, 0.5])
        gaps = eigengap_curvature(eigenvalues, k=3, normalize=False)
        np.testing.assert_allclose(gaps, [9.5, 5.5, 1.5])

    def test_normalized_in_unit_interval(self) -> None:
        eigenvalues = np.array([10.0, 6.0, 2.0, 0.5])
        gaps = eigengap_curvature(eigenvalues, k=3, normalize=True)
        assert np.all(gaps >= 0.0) and np.all(gaps <= 1.0)
        np.testing.assert_allclose(gaps, [0.95, 0.55, 0.15])

    def test_flat_spectrum_gives_zero_curvature(self) -> None:
        eigenvalues = np.ones(5)
        gaps = eigengap_curvature(eigenvalues, k=3)
        np.testing.assert_allclose(gaps, np.zeros(3))

    def test_out_of_range_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must satisfy"):
            eigengap_curvature(np.array([2.0, 1.0]), k=2)


class TestGeodesicAcceleration:
    def test_zero_for_constant_trajectory(self) -> None:
        rng = np.random.default_rng(40)
        basis = np.linalg.qr(rng.standard_normal((6, 2)))[0]
        accel = geodesic_acceleration(basis, basis, basis)
        assert accel == pytest.approx(0.0, abs=1e-10)

    def test_small_step_along_geodesic_gives_small_acceleration(self) -> None:
        # A Grassmann geodesic still curves in projector space, so the chord
        # second difference is nonzero — but it vanishes as O(step^2) as the
        # spacing shrinks.
        def basis_at(t: float) -> np.ndarray:
            b = np.zeros((4, 2))
            b[0, 0] = 1.0
            b[1, 1] = math.cos(t)
            b[2, 1] = math.sin(t)
            return b

        step_large = geodesic_acceleration(basis_at(0.0), basis_at(0.2), basis_at(0.4))
        step_small = geodesic_acceleration(basis_at(0.0), basis_at(0.02), basis_at(0.04))
        # Shrinking the step by 10x should shrink acceleration by ~100x.
        assert step_small < step_large / 50.0

    def test_large_for_sharp_kink(self) -> None:
        # Path goes far, then snaps back to start: large second difference.
        basis_prev = np.eye(4)[:, :2]
        basis_curr = np.eye(4)[:, 2:]
        basis_next = np.eye(4)[:, :2]
        accel = geodesic_acceleration(basis_prev, basis_curr, basis_next)
        assert accel > 1.0

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        bases = [np.linalg.qr(rng.standard_normal((6, 2)))[0] for _ in range(3)]
        accel = geodesic_acceleration(*bases)
        assert accel >= 0.0
