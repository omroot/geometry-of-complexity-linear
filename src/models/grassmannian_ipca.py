"""Grassmannian-IPCA estimator backed by pymanopt.

Provides:

* :class:`ConfigurableGrassmannian` — extends ``pymanopt.manifolds.Grassmann``
  with a selectable retraction backend (``svd`` / ``qr``).
* :class:`GrassmannianManifoldIPCAEstimator` — fits the profiled IPCA loss
  ``L(W) = (1/T) sum_t min_f || r_t - (Z_t W) f ||^2`` over orthonormal
  ``W in R^{m x k}`` on the Grassmann manifold ``Gr(m, k)``.

The factor step is solved in closed form per cross-section (ridge-regularised
when ``shrinkage > 0``). The subspace step is the only non-trivial
optimisation and is handled by pymanopt on the manifold.
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
from pymanopt import Problem
from pymanopt.function import autograd as pymanopt_autograd
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions


class ConfigurableGrassmannian(Grassmann):
    """Grassmann manifold with selectable retraction backend.

    Subclasses :class:`pymanopt.manifolds.Grassmann` to expose a ``qr``
    retraction in addition to the pymanopt default ``svd``/``polar``. The
    ``qr`` backend sign-fixes the returned representative (positive diagonal
    of ``R``) so warm-started runs stay on the same continuous branch.

    Args:
        n: Ambient dimension.
        p: Subspace dimension (``p <= n``).
        retraction_method: One of ``"auto"``, ``"svd"``, ``"polar"``,
            ``"qr"``, ``"thin_qr"`` (case / hyphen / underscore agnostic).
            ``auto``, ``svd``, and ``polar`` map to the pymanopt default.

    Raises:
        ValueError: If ``retraction_method`` is unknown.
    """

    def __init__(self, n: int, p: int, *, retraction_method: str = "svd") -> None:
        super().__init__(n, p)
        method = str(retraction_method).lower().replace("_", "").replace("-", "")
        aliases = {
            "auto": "svd",
            "svd": "svd",
            "polar": "svd",
            "qr": "qr",
            "thinqr": "qr",
        }
        try:
            self._retraction_method = aliases[method]
        except KeyError as exc:
            raise ValueError(
                "retraction_method must be one of: auto, svd, polar, qr, thin_qr"
            ) from exc

    def retraction(
        self, point: np.ndarray, tangent_vector: np.ndarray
    ) -> np.ndarray:
        if self._retraction_method == "qr":
            y = point + tangent_vector
            q, r = np.linalg.qr(y, mode="reduced")
            # Fix QR sign ambiguity so the returned representative is stable
            # across warm-started iterations.
            signs = np.sign(np.diagonal(r, axis1=-2, axis2=-1))
            signs = np.where(signs == 0, 1.0, signs)
            return q * np.expand_dims(signs, axis=-2)
        return super().retraction(point, tangent_vector)


class GrassmannianManifoldIPCAEstimator:
    """Profiled IPCA estimator with the loading matrix on a Grassmannian.

    Optimisation variable:
        ``W in R^{m x k}`` with orthonormal columns (a representative of a
        point on ``Gr(m, k)``).

    Profiled loss:
        ``L(W) = (1/T) sum_t || r_t - (Z_t W) f_hat_t(W) ||^2``,
        with the ridge-regularised per-cross-section factor step
        ``f_hat_t(W) = (L_t^T L_t + shrinkage * I)^{-1} L_t^T r_t``
        where ``L_t = Z_t W``.

    The subspace step uses pymanopt's :class:`ConjugateGradient` /
    :class:`SteepestDescent` / :class:`TrustRegions` on the manifold. The
    factor step is in closed form.

    Args:
        n_assets: Number of assets per cross-section (``N``).
        k: Number of latent factors (subspace dimension).
        m: Ambient characteristic dimension (= number of rows of ``W``).
        t_train: Training window length in periods (``T``).
        shrinkage: Ridge regularisation added to the per-cross-section factor
            step. Default ``0`` (no shrinkage).
    """

    def __init__(
        self,
        n_assets: int,
        k: int,
        m: int,
        t_train: int,
        shrinkage: float = 0.0,
    ) -> None:
        self.n_assets = n_assets
        self.k = k
        self.m = m
        self.t_train = t_train
        self.shrinkage = float(shrinkage)

    def estimate_f(
        self,
        W: np.ndarray,
        data: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Closed-form ridge factor step ``f_hat`` given a fixed basis ``W``.

        Args:
            W: Loading matrix of shape ``(m, k)`` (orthonormal columns are
                assumed but not enforced).
            data: Tuple ``(rets, Z)`` with ``rets`` of shape ``(T, N)`` and
                ``Z`` of shape ``(T, N, m)``.

        Returns:
            Array of shape ``(T, k)`` with the per-period factor estimates.
        """
        rets, Z = data
        k = self.k
        z = self.shrinkage
        # Vectorised: single batched matmul + solve, no Python loop.
        L = Z @ W                                          # (T, N, k)
        Lt = L.transpose(0, 2, 1)                           # (T, k, N)
        LtL = Lt @ L                                         # (T, k, k)
        LtL_reg = LtL + z * np.eye(k)                       # (T, k, k)
        Ltr = (Lt @ rets[..., None]).squeeze(-1)             # (T, k)
        f_hat = np.linalg.solve(LtL_reg, Ltr[..., None]).squeeze(-1)  # (T, k)
        return f_hat

    def fit(
        self,
        data: tuple[np.ndarray, np.ndarray],
        optimizer: str = "ConjugateGradient",
        max_iterations: int = 200,
        iter_tol: float = 1e-6,
        verbosity: int = 1,
        log_verbosity: int = 0,
        initial_point: np.ndarray | None = None,
        reuse_line_searcher: bool = False,
        cg_beta_rule: str = "PolakRibiere",
        retraction_method: str = "svd",
        return_pymanopt_result: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, list[float]]
        | tuple[np.ndarray, np.ndarray, list[float], object]
    ):
        """Fit the profiled IPCA loss on ``Gr(m, k)`` via pymanopt.

        Args:
            data: Tuple ``(rets, Z)`` with shapes ``(T, N)`` and ``(T, N, m)``.
            optimizer: Manifold solver. One of ``"ConjugateGradient"``,
                ``"SteepestDescent"``, ``"TrustRegions"`` (case / underscore
                / abbreviation agnostic; ``"cg"``, ``"sd"``, ``"tr"`` accepted).
            max_iterations: Maximum manifold-optimiser iterations.
            iter_tol: Gradient-norm convergence tolerance. Must be positive.
            verbosity: Solver verbosity (``0`` = silent, ``2`` = per-iter).
            log_verbosity: Per-iter history retention in the solver log
                (``0`` = none, ``2`` = full).
            initial_point: Optional warm-start basis of shape ``(m, k)``. When
                omitted, the solver draws a random orthonormal start.
            reuse_line_searcher: ConjugateGradient-only knob; persists the
                line-searcher state across warm-started restarts.
            cg_beta_rule: ConjugateGradient beta rule, e.g. ``"PolakRibiere"``
                or ``"FletcherReeves"``.
            retraction_method: Forwarded to :class:`ConfigurableGrassmannian`.
            return_pymanopt_result: When True, additionally returns the raw
                pymanopt result object as a fourth tuple element.

        Returns:
            ``(W_opt, f_hat, history)``, where ``W_opt`` has shape ``(m, k)``,
            ``f_hat`` has shape ``(T, k)``, and ``history`` is the per-iter
            cost trace from ``result.log`` (a one-element list when no log
            was kept). When ``return_pymanopt_result`` is True, the raw
            pymanopt result is appended as a fourth element.

        Raises:
            ValueError: If ``iter_tol <= 0``, if ``rets`` / ``Z`` shapes do
                not match ``(t_train, n_assets[, m])``, if ``initial_point``
                has the wrong shape, or if ``optimizer`` is unknown.
        """
        if iter_tol <= 0:
            raise ValueError(f"iter_tol must be > 0, got {iter_tol}")

        rets, Z = data
        expected_rets = (self.t_train, self.n_assets)
        expected_Z = (self.t_train, self.n_assets, self.m)
        if rets.shape != expected_rets:
            raise ValueError(
                f"rets shape {rets.shape} does not match expected {expected_rets}"
            )
        if Z.shape != expected_Z:
            raise ValueError(
                f"Z shape {Z.shape} does not match expected {expected_Z}"
            )

        # Capture scalars before autograd tracing.
        shrinkage_ = self.shrinkage
        k_ = self.k

        # NOTE: A Python for-loop is intentionally used here rather than
        # batched matmul/solve. Benchmarking shows the loop is 1.6-5x FASTER
        # at the actual problem sizes (P in {1024, 4096}, k=24) because
        # autograd's gradient tape for batched 3-D operations has higher
        # overhead than small 2-D operations in a loop.
        def ipca_profiled_loss_autograd(W, rets_, Z_):
            T_, _ = rets_.shape
            obj = 0.0
            for t in range(T_):
                Zt = Z_[t]                              # (N, m)
                rt = rets_[t]                           # (N,)
                Lt = anp.dot(Zt, W)                     # (N, k)
                XtX = anp.dot(Lt.T, Lt)                 # (k, k)
                XtX_reg = XtX + shrinkage_ * anp.eye(k_)
                Xty = anp.dot(Lt.T, rt)                 # (k,)
                bt = anp.linalg.solve(XtX_reg, Xty)     # ridge factor step
                resid = rt - anp.dot(Lt, bt)
                obj = obj + anp.dot(resid, resid)
            return obj / T_

        manifold = ConfigurableGrassmannian(
            self.m, self.k, retraction_method=retraction_method,
        )
        rets_ag = anp.asarray(rets)
        Z_ag = anp.asarray(Z)

        @pymanopt_autograd(manifold)
        def cost(W):
            return ipca_profiled_loss_autograd(W, rets_ag, Z_ag)

        problem = Problem(manifold=manifold, cost=cost)

        opt = optimizer.lower().replace("_", "")
        if opt in {"cg", "conjugategradient"}:
            solver = ConjugateGradient(
                beta_rule=cg_beta_rule,
                max_iterations=max_iterations,
                min_gradient_norm=float(iter_tol),
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs: dict = {"reuse_line_searcher": reuse_line_searcher}
        elif opt in {"sd", "steepestdescent"}:
            solver = SteepestDescent(
                max_iterations=max_iterations,
                min_gradient_norm=float(iter_tol),
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs = {}
        elif opt in {"tr", "trustregions"}:
            solver = TrustRegions(
                max_iterations=max_iterations,
                min_gradient_norm=float(iter_tol),
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs = {}
        else:
            raise ValueError(
                "optimizer must be one of: ConjugateGradient, SteepestDescent, TrustRegions"
            )

        if initial_point is not None:
            initial_point = np.asarray(initial_point)
            expected_init = (self.m, self.k)
            if initial_point.shape != expected_init:
                raise ValueError(
                    f"initial_point shape {initial_point.shape} does not match "
                    f"expected {expected_init}"
                )
            result = solver.run(problem, initial_point=initial_point, **run_kwargs)
        else:
            result = solver.run(problem, **run_kwargs)

        W_opt = np.asarray(result.point)

        # pymanopt stores per-iter costs in result.log; fall back to a single
        # final-cost entry when no log was kept (log_verbosity == 0).
        history: list[float] = []
        if getattr(result, "log", None) is not None:
            iters = result.log.get("iterations", None)
            if isinstance(iters, dict) and "cost" in iters:
                history = [float(c) for c in iters["cost"]]
        if not history:
            history = [float(cost(W_opt))]

        f_hat = self.estimate_f(W_opt, data)

        if return_pymanopt_result:
            return W_opt, f_hat, history, result
        return W_opt, f_hat, history
