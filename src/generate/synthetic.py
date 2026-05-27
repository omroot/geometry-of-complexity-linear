"""Unified synthetic-data generator for paper §8 experiments.

Implements the paper §8.1 DGP and exposes two entry points:

* :func:`generate_synthetic_panel` — the raw generator. Returns a
  ``(date, permno)`` MultiIndex DataFrame with ``ret`` and the
  characteristic columns, plus a ``truth`` dict (true k0, signal
  eigenvalues, W*(t) series, factor returns, etc.). Used by the broad-grid
  sweeps (paper §8.2-§8.4).

* :func:`generate_workflow_panel` — convenience wrapper that runs the DGP
  and reshapes the output for the rolling
  :class:`src.workflows.grassmannian_ipca.GrassmannianIPCAWorkflow`. Adds
  ``yyyymm``/``permno`` columns, renames ``ret -> excess_ret``, computes
  ``y_ipca`` (next-month excess return per asset), and synthesizes a
  persistent ``mcap`` column for the workflow's market-cap filter. Used
  by the short-window runs (paper §8.5).

Single DGP across both paths: paper §8.1 with k0 swappable.

The DGP at a glance (paper §8.1):

    r_{i,t} = (Z_{i,t} @ W*(t)) @ f_t + ε_{i,t}

where ``Z_{i,t} ~ N(0, Toeplitz(z_corr))``; ``W*(t)`` is an isotropic
Grassmannian random walk on Gr(m, k0) with step ``w_drift_scale``
auto-calibrated to a target ``d_proj`` (default 0.6, matching real-data
backtest at k=12, z=50, P=1024); ``f_t`` is AR(1) with the geometrically
decaying signal spectrum ``λ_j = snr_base · decay^(j-1) + σ²``; and
``ε_{i,t}`` is Gaussian with log-normal cross-sectional heteroskedasticity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _grassmannian_step(
    W: np.ndarray, delta_scale: float, rng: np.random.Generator
) -> np.ndarray:
    """One isotropic step on Gr(m, k) starting from ``W``.

    Generates a horizontal tangent (orthogonal to the current subspace) of
    scale ``delta_scale`` and retracts via QR. Equivalent to the W*(t)
    update rule described in paper §8.1.

    Args:
        W: Current orthonormal basis, shape ``(m, k)``.
        delta_scale: Step size on the Grassmannian.
        rng: NumPy random generator.

    Returns:
        New orthonormal basis of the same shape.
    """
    m, k = W.shape
    Z = rng.normal(size=(m, k)) * delta_scale
    Z = Z - W @ (W.T @ Z)  # project to horizontal space
    W_new, _ = np.linalg.qr(W + Z)
    return W_new[:, :k]


def _calibrate_drift_scale(k0: int, window_len: int, target_dproj: float) -> float:
    """Back out ``w_drift_scale`` from a target rolling-window d_proj.

    Approximation: over a window of length ``T``, the subspace accumulates
    ``T`` independent Grassmannian steps; each step of size ``δ`` contributes
    roughly ``δ`` to each of the ``k0`` principal angles, so
    ``d_proj ≈ sqrt(2 · k0 · T) · δ``. Inverting gives the returned ``δ``.

    Args:
        k0: True signal rank.
        window_len: Rolling-window length used as the calibration reference.
        target_dproj: Target projection distance to hit at that window length.

    Returns:
        Per-step Grassmannian drift scale.
    """
    return target_dproj / np.sqrt(2 * k0 * window_len)


# ---------------------------------------------------------------------------
# DGP (paper §8.1) — MultiIndex output
# ---------------------------------------------------------------------------


def generate_synthetic_panel(
    T: int = 360,
    N: int = 50,
    m: int = 30,
    k0: int = 12,
    seed: int = 42,
    snr_base: float = 5.0,
    eigenvalue_decay: float = 0.82,
    sigma_eps: float = 1.0,
    factor_ar1: float = 0.5,
    factor_vol: float = 0.3,
    w_drift_scale: float | None = None,
    target_dproj: float = 0.6,
    calibration_window: int = 24,
    z_corr: float = 0.3,
    z_scale: float = 1.0,
    hetero_strength: float = 0.3,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Generate the paper §8.1 synthetic IPCA panel with known true rank.

    The signal eigenvalues are designed as
    ``λ_j = snr_base · eigenvalue_decay^(j-1) + sigma_eps^2`` so the spectrum
    has the geometric cliff at ``j = k0`` that the §8.3 figures validate.
    Subspace dynamics use an isotropic Grassmannian random walk; ``f_t``
    follows an AR(1) with structured covariance matched to the signal
    spectrum.

    Args:
        T: Number of months.
        N: Number of assets.
        m: Number of raw characteristics. Must be ``>= k0``.
        k0: True signal rank.
        seed: RNG seed.
        snr_base: Excess of the strongest signal eigenvalue over the noise
            floor: ``λ_1 = snr_base + sigma_eps^2``.
        eigenvalue_decay: Geometric decay rate for signal eigenvalues.
        sigma_eps: Idiosyncratic noise standard deviation.
        factor_ar1: AR(1) persistence of factor returns.
        factor_vol: Factor innovation volatility multiplier.
        w_drift_scale: Per-step Grassmannian drift scale. ``None`` =>
            auto-calibrate from ``target_dproj``.
        target_dproj: Target projection distance at the calibration window
            (only used when ``w_drift_scale is None``).
        calibration_window: Window length used by the calibration formula.
        z_corr: Toeplitz cross-sectional correlation for characteristics.
        z_scale: Overall characteristic scale.
        hetero_strength: Log-normal spread of idiosyncratic vol across assets.

    Returns:
        Tuple ``(df, truth, char_cols)``:

        * ``df`` — MultiIndex ``(date, permno)`` DataFrame with columns
          ``ret``, ``Price`` (=1.0), and ``char_1 ... char_m``.
        * ``truth`` — dict with ``true_k0``, ``signal_eigenvalues``,
          ``W_star_series``, ``factor_returns``, ``sigma_i``, ``params``,
          and the calibrated ``w_drift_scale``.
        * ``char_cols`` — list of characteristic column names.
    """
    rng = np.random.default_rng(seed)
    assert m >= k0, f"Need m >= k0 ({m} < {k0})"

    sigma_sq = sigma_eps ** 2
    signal_eigenvalues = np.array(
        [snr_base * (eigenvalue_decay ** j) + sigma_sq for j in range(k0)]
    )
    excess = np.maximum(signal_eigenvalues - sigma_sq, 1e-8)
    L_f = np.diag(np.sqrt(excess))

    if w_drift_scale is None:
        w_drift_scale = _calibrate_drift_scale(k0, calibration_window, target_dproj)

    idx = np.arange(m)
    Sigma_z = z_corr ** np.abs(idx[:, None] - idx[None, :])
    Lz = np.linalg.cholesky(Sigma_z + 1e-10 * np.eye(m))

    A0 = rng.normal(size=(m, k0))
    W_star_0, _ = np.linalg.qr(A0)
    W_star_0 = W_star_0[:, :k0]

    W_star_series = [W_star_0]
    W_t = W_star_0.copy()
    for _ in range(1, T):
        W_t = _grassmannian_step(W_t, w_drift_scale, rng)
        W_star_series.append(W_t.copy())

    u = rng.normal(size=N)
    sigma_i = sigma_eps * np.exp(hetero_strength * u)
    sigma_i = np.clip(sigma_i, 0.1, np.percentile(sigma_i, 99))

    dates = pd.date_range("1994-01-01", periods=T, freq="MS")
    permnos = np.arange(10001, 10001 + N)
    char_cols = [f"char_{j+1}" for j in range(m)]

    rows: list[dict] = []
    f_prev = np.zeros(k0)
    all_f = np.zeros((T, k0))

    for t in range(T):
        W_t = W_star_series[t]
        innov = L_f @ rng.normal(size=k0) * factor_vol
        f_t = factor_ar1 * f_prev + innov
        f_prev = f_t
        all_f[t] = f_t

        Z_t = (rng.normal(size=(N, m)) @ Lz.T) * z_scale
        Lambda_t = Z_t @ W_t
        signal = Lambda_t @ f_t
        eps = rng.normal(size=N) * sigma_i
        ret_t = signal + eps

        for i in range(N):
            row = {"date": dates[t], "permno": permnos[i], "ret": ret_t[i], "Price": 1.0}
            for j in range(m):
                row[char_cols[j]] = Z_t[i, j]
            rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "permno"]).sort_index()

    truth = {
        "true_k0": k0,
        "signal_eigenvalues": signal_eigenvalues,
        "snr_base": snr_base,
        "eigenvalue_decay": eigenvalue_decay,
        "sigma_eps": sigma_eps,
        "w_drift_scale": w_drift_scale,
        "W_star_series": W_star_series,
        "factor_returns": all_f,
        "sigma_i": sigma_i,
        "params": {
            "T": T, "N": N, "m": m, "k0": k0,
            "snr_base": snr_base,
            "eigenvalue_decay": eigenvalue_decay,
            "factor_ar1": factor_ar1, "factor_vol": factor_vol,
            "w_drift_scale": w_drift_scale,
            "target_dproj": target_dproj,
            "z_corr": z_corr, "z_scale": z_scale,
            "sigma_eps": sigma_eps, "hetero_strength": hetero_strength,
        },
    }

    return df, truth, char_cols


# ---------------------------------------------------------------------------
# Workflow-shape adapter (used by §8.5 short-window runs)
# ---------------------------------------------------------------------------


def generate_workflow_panel(
    *,
    mcap_loc: float = 10.0,
    mcap_rho: float = 0.98,
    mcap_noise: float = 0.15,
    mcap_return_loading: float = 0.25,
    mcap_seed_offset: int = 1,
    **dgp_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Generate a paper §8.1 panel reshaped for the rolling IPCA workflow.

    Runs :func:`generate_synthetic_panel` with the given DGP kwargs, then:

    * Flattens the ``(date, permno)`` MultiIndex into ``yyyymm`` and
      ``permno`` columns.
    * Renames ``ret -> excess_ret`` and duplicates it as ``ret_adj`` (both
      column names are used by different parts of the workflow).
    * Drops the dummy ``Price`` column (the workflow treats unknown columns
      as characteristics).
    * Adds ``y_ipca = next-month excess_ret`` per asset and drops the final
      month per asset (no target available).
    * Synthesizes a persistent log-AR(1) ``mcap`` column with feedback from
      lagged returns; the workflow's market-cap filter (``mcap_filter_top_n``)
      uses this when present and skips filtering when absent.

    Args:
        mcap_loc: Long-run mean of log market cap.
        mcap_rho: AR(1) persistence of log market cap.
        mcap_noise: Innovation standard deviation of log market cap.
        mcap_return_loading: Feedback from lagged excess return into log
            market cap.
        mcap_seed_offset: Added to ``dgp_kwargs['seed']`` to seed the mcap
            RNG independently of the DGP RNG.
        **dgp_kwargs: Forwarded to :func:`generate_synthetic_panel`.

    Returns:
        Tuple ``(panel_df, truth)``:

        * ``panel_df`` — long-form DataFrame with columns
          ``yyyymm``, ``permno``, ``excess_ret``, ``ret_adj``, ``mcap``,
          ``char_1 ... char_m``, ``y_ipca``. Sorted by ``(yyyymm, permno)``.
        * ``truth`` — same dict returned by
          :func:`generate_synthetic_panel`, augmented with ``permnos``,
          ``dates``, and ``char_cols``.
    """
    df, truth, char_cols = generate_synthetic_panel(**dgp_kwargs)

    # Pull dates and permnos from the MultiIndex; they are the canonical
    # ordering for the mcap simulation below.
    dates = df.index.get_level_values("date").unique().sort_values()
    permnos = df.index.get_level_values("permno").unique().sort_values()
    T, N = len(dates), len(permnos)

    # (T, N) panel of excess returns in date-major order. ``unstack`` keeps
    # dates on rows and permnos on columns; reindex enforces the order so
    # later .reshape(-1) lines up with np.tile/np.repeat below.
    rets_wide = (
        df["ret"].unstack(level="permno").reindex(index=dates, columns=permnos)
    )
    rets = rets_wide.to_numpy()

    # Synthesize a persistent log market cap with feedback from lagged returns.
    seed = int(dgp_kwargs.get("seed", 42))
    mcap_rng = np.random.default_rng(seed + mcap_seed_offset)
    log_mcap = np.empty((T, N), dtype=float)
    log_mcap[0] = mcap_rng.normal(loc=mcap_loc, scale=1.0, size=N)
    for t in range(1, T):
        log_mcap[t] = (
            mcap_rho * log_mcap[t - 1]
            + (1.0 - mcap_rho) * mcap_loc
            + mcap_return_loading * rets[t - 1]
            + mcap_rng.normal(scale=mcap_noise, size=N)
        )
    mcap = np.exp(np.clip(log_mcap, -10.0, 20.0))

    # Pull characteristics in the same date-major order so reshape matches.
    char_wide = {
        col: df[col]
        .unstack(level="permno")
        .reindex(index=dates, columns=permnos)
        .to_numpy()
        for col in char_cols
    }

    panel_df = pd.DataFrame(
        {
            "yyyymm": np.repeat(dates.to_numpy(), N),
            "permno": np.tile(permnos.to_numpy(), T),
            "excess_ret": rets.reshape(-1),
            "ret_adj": rets.reshape(-1),
            "mcap": mcap.reshape(-1),
        }
    )
    for col in char_cols:
        panel_df[col] = char_wide[col].reshape(-1)

    panel_df = panel_df.sort_values(["permno", "yyyymm"]).reset_index(drop=True)
    panel_df["y_ipca"] = panel_df.groupby("permno")["excess_ret"].shift(-1)
    panel_df = (
        panel_df.dropna(subset=["y_ipca"])
        .sort_values(["yyyymm", "permno"])
        .reset_index(drop=True)
    )

    truth = dict(truth)
    truth["permnos"] = permnos.to_numpy()
    truth["dates"] = dates
    truth["char_cols"] = char_cols

    return panel_df, truth
