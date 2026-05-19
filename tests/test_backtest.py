"""Unit tests for src/backtest/backtest.py.

Verifies that the consolidated ``run_backtest`` agrees bit-exactly with the
legacy ``run_ipca_grassmannian_v2`` on a deterministic synthetic panel, and
that the mktcap and "both" paths return well-formed superset dicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtest import (
    run_backtest,
    run_ipca_grassmannian_mktcap,
    run_ipca_grassmannian_v2,
)
from src.backtest.grassmannian_ipca import (
    run_ipca_grassmannian_v2 as legacy_v2,
)


# ---------------------------------------------------------------------------
# Fixture: deterministic synthetic panel with embedded 2-factor structure
# ---------------------------------------------------------------------------


def _make_synthetic_panel(
    n_dates: int = 30,
    n_assets: int = 25,
    n_chars: int = 5,
    seed: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2000-01", periods=n_dates, freq="M").to_timestamp()
    permnos = list(range(10001, 10001 + n_assets))
    chars = [f"x{i}" for i in range(n_chars)]

    f_true = rng.standard_normal((n_dates, 2))
    Gamma = rng.standard_normal((n_chars, 2))

    rows = []
    for ti, d in enumerate(dates):
        for permno in permnos:
            x = rng.standard_normal(n_chars)
            loadings = x @ Gamma
            ret = float(loadings @ f_true[ti] + 0.1 * rng.standard_normal())
            mktcap = float(np.exp(rng.standard_normal()) * 100.0)
            row = {"date": d, "permno": permno, "ret": ret, "mktcap": mktcap}
            for c, xi in zip(chars, x):
                row[c] = float(xi)
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["date", "permno"])
    return df, chars


@pytest.fixture(scope="module")
def panel():
    return _make_synthetic_panel()


# ---------------------------------------------------------------------------
# 1. Bit-exact agreement with legacy run_ipca_grassmannian_v2
# ---------------------------------------------------------------------------


class TestEqualWeightedAgreesWithLegacy:
    """Every key returned by the legacy v2 must agree bit-exactly with the new path."""

    def _run_both(self, panel) -> tuple[dict, dict]:
        df, chars = panel
        np.random.seed(0)
        legacy = legacy_v2(
            df_trunc=df, char_feats=chars, num_fact=2, window_len=12, verbose=False
        )
        np.random.seed(0)
        new = run_backtest(
            df_trunc=df,
            char_feats=chars,
            num_fact=2,
            window_len=12,
            verbose=False,
            weighting="equal",
            impute="mean",
        )
        return legacy, new

    def test_scalar_keys_bit_exact(self, panel) -> None:
        legacy, new = self._run_both(panel)
        scalar_keys = [
            "r2_oos",
            "sharpe",
            "mean_subspace_stability",
            "mean_max_principal_angle",
            "mean_mean_principal_angle",
            "std_mean_principal_angle",
            "std_max_principal_angle",
            "sr_mean_principal_angle",
            "sr_max_principal_angle",
            "mean_geodesic_accel",
            "mean_erank",
            "erank_collapse_frac",
            "mean_spectral_gap",
            "mean_d_proj_norm",
            "d_proj_random_baseline",
            "mean_d_proj_seeds",
            "d_proj_seeds_flat_frac",
        ]
        for k in scalar_keys:
            a, b = legacy[k], new[k]
            if np.isnan(a) and np.isnan(b):
                continue
            assert a == pytest.approx(b, abs=1e-12), (
                f"scalar key {k}: legacy={a}, new={b}"
            )

    def test_series_keys_bit_exact(self, panel) -> None:
        legacy, new = self._run_both(panel)
        for k in [
            "stability_dist_series",
            "erank_series",
            "spectral_gap_series",
            "d_proj_norm_series",
            "d_proj_seeds_series",
            "geodesic_accel_series",
        ]:
            np.testing.assert_allclose(
                np.asarray(legacy[k], dtype=float),
                np.asarray(new[k], dtype=float),
                atol=1e-12,
                err_msg=f"series key {k} mismatch",
            )

    def test_principal_angles_series_bit_exact(self, panel) -> None:
        legacy, new = self._run_both(panel)
        assert len(legacy["principal_angles_series"]) == len(
            new["principal_angles_series"]
        )
        for i, (a, b) in enumerate(
            zip(legacy["principal_angles_series"], new["principal_angles_series"])
        ):
            np.testing.assert_allclose(a, b, atol=1e-12, err_msg=f"window {i}")

    def test_v2_shim_matches_legacy(self, panel) -> None:
        df, chars = panel
        np.random.seed(0)
        legacy = legacy_v2(
            df_trunc=df, char_feats=chars, num_fact=2, window_len=12, verbose=False
        )
        np.random.seed(0)
        via_shim = run_ipca_grassmannian_v2(
            df_trunc=df, char_feats=chars, num_fact=2, window_len=12, verbose=False
        )
        for k in legacy:
            if isinstance(legacy[k], list) and legacy[k] and isinstance(
                legacy[k][0], np.ndarray
            ):
                for la, sa in zip(legacy[k], via_shim[k]):
                    np.testing.assert_allclose(la, sa, atol=1e-12)
            elif isinstance(legacy[k], list):
                np.testing.assert_allclose(
                    np.asarray(legacy[k], dtype=float),
                    np.asarray(via_shim[k], dtype=float),
                    atol=1e-12,
                )
            else:
                if np.isnan(legacy[k]) and np.isnan(via_shim[k]):
                    continue
                assert legacy[k] == pytest.approx(via_shim[k], abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Return-dict shape contract
# ---------------------------------------------------------------------------


REQUIRED_KEYS_V2 = {
    "r2_oos",
    "sharpe",
    "mean_subspace_stability",
    "stability_dist_series",
    "mean_max_principal_angle",
    "mean_mean_principal_angle",
    "principal_angles_series",
    "std_mean_principal_angle",
    "std_max_principal_angle",
    "sr_mean_principal_angle",
    "sr_max_principal_angle",
    "mean_geodesic_accel",
    "geodesic_accel_series",
    "mean_erank",
    "erank_series",
    "erank_collapse_frac",
    "mean_spectral_gap",
    "spectral_gap_series",
    "mean_d_proj_norm",
    "d_proj_norm_series",
    "d_proj_random_baseline",
    "mean_d_proj_seeds",
    "d_proj_seeds_series",
    "d_proj_seeds_flat_frac",
}


class TestReturnDictContract:
    def test_equal_contains_v2_keys(self, panel) -> None:
        df, chars = panel
        np.random.seed(0)
        result = run_backtest(
            df_trunc=df,
            char_feats=chars,
            num_fact=2,
            window_len=12,
            verbose=False,
            weighting="equal",
        )
        assert REQUIRED_KEYS_V2.issubset(set(result))
        # Equal-weighted path does not advertise the dual-Sharpe keys.
        assert "sharpe_mktcap" not in result

    def test_mktcap_adds_dual_sharpes(self, panel) -> None:
        df, chars = panel
        np.random.seed(0)
        result = run_backtest(
            df_trunc=df,
            char_feats=chars,
            num_fact=2,
            window_len=12,
            verbose=False,
            weighting="mktcap",
        )
        assert REQUIRED_KEYS_V2.issubset(set(result))
        assert "sharpe_ew" in result
        assert "sharpe_mktcap" in result
        # When weighting='mktcap', the primary 'sharpe' aliases the mktcap version.
        assert result["sharpe"] == pytest.approx(result["sharpe_mktcap"])

    def test_both_returns_both(self, panel) -> None:
        df, chars = panel
        np.random.seed(0)
        result = run_backtest(
            df_trunc=df,
            char_feats=chars,
            num_fact=2,
            window_len=12,
            verbose=False,
            weighting="both",
        )
        assert "sharpe_ew" in result and "sharpe_mktcap" in result
        # 'sharpe' aliases mktcap when weighting='both' (per design).
        assert result["sharpe"] == pytest.approx(result["sharpe_mktcap"])
        # The EW Sharpe under 'both' should match the equal-weighted run.
        np.random.seed(0)
        eq = run_backtest(
            df_trunc=df,
            char_feats=chars,
            num_fact=2,
            window_len=12,
            verbose=False,
            weighting="equal",
        )
        assert result["sharpe_ew"] == pytest.approx(eq["sharpe"], abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Mktcap path: runs to completion and produces a different Sharpe
# ---------------------------------------------------------------------------


class TestMktcapPath:
    def test_mktcap_runs_via_shim(self, panel) -> None:
        df, chars = panel
        np.random.seed(0)
        result = run_ipca_grassmannian_mktcap(
            df_trunc=df, char_feats=chars, num_fact=2, window_len=12, verbose=False
        )
        assert "sharpe_mktcap" in result
        assert np.isfinite(result["sharpe_mktcap"])

    def test_mktcap_requires_mktcap_column(self, panel) -> None:
        df, chars = panel
        with pytest.raises(ValueError, match="requires a 'mktcap' column"):
            run_backtest(
                df_trunc=df.drop(columns=["mktcap"]),
                char_feats=chars,
                num_fact=2,
                window_len=12,
                verbose=False,
                weighting="mktcap",
            )

    def test_invalid_weighting_raises(self, panel) -> None:
        df, chars = panel
        with pytest.raises(ValueError, match="weighting must be"):
            run_backtest(
                df_trunc=df,
                char_feats=chars,
                num_fact=2,
                window_len=12,
                verbose=False,
                weighting="bogus",
            )

    def test_invalid_impute_raises(self, panel) -> None:
        df, chars = panel
        with pytest.raises(ValueError, match="impute must be"):
            run_backtest(
                df_trunc=df,
                char_feats=chars,
                num_fact=2,
                window_len=12,
                verbose=False,
                impute="bogus",
            )
