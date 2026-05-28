"""Process-level parallelism for sweep cells.

Each synthetic / real-data sweep iterates a Cartesian product of
hyperparameters (``k``, ``rff_n_components``, ``alpha``, ``rff_seed``,
...) and runs one :func:`src.sweep.runner.run_one_fit` per cell. Cells
are independent, so the natural parallelization is process-level over
cells with each worker pinned to a single BLAS thread (to avoid
oversubscription between joblib workers and NumPy's thread pool).

Typical use from a sweep script::

    cells = [
        SweepCell(
            identity={"sweep": "sweep_A", "k": k, "P": m, "z": z, "seed": seed},
            fit_kwargs=dict(k=k, rff_n_components=m, alpha=z, rff_seed=seed, ...),
        )
        for k, m, z, seed in product(...)
    ]
    rows = run_sweep(panel, cells, n_jobs=-1)
    pd.DataFrame(rows).to_csv(out_path, index=False)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import pandas as pd
from joblib import Parallel, delayed

from src.sweep.runner import run_one_fit


logger = logging.getLogger(__name__)


_BLAS_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class SweepCell:
    """One independent unit of work in a sweep.

    Attributes:
        identity: Row-level metadata describing which cell this is
            (e.g. ``{"sweep": "sweep_A", "k": 12, "P": 1024, "z": 50.0,
            "seed": 10}``). Copied into the output row verbatim so the
            resulting CSV is self-describing.
        fit_kwargs: Keyword arguments forwarded to
            :func:`src.sweep.runner.run_one_fit`. The identity dict and
            the fit kwargs typically share keys with different names
            (e.g. ``identity["P"]`` vs ``fit_kwargs["rff_n_components"]``)
            — that mapping is the caller's responsibility.
    """

    identity: dict
    fit_kwargs: dict


def _fit_one(
    panel: pd.DataFrame,
    cell: SweepCell,
    on_error: Literal["raise", "skip"],
) -> Optional[dict]:
    """Run a single cell and return ``{**identity, **metrics}`` or ``None``."""
    try:
        metrics = run_one_fit(panel, **cell.fit_kwargs)
    except Exception as exc:
        if on_error == "raise":
            raise
        logger.warning("sweep cell %s failed: %r", cell.identity, exc)
        return None
    if metrics is None:
        return None
    return {**cell.identity, **metrics}


def run_sweep(
    panel: pd.DataFrame,
    cells: Sequence[SweepCell],
    *,
    n_jobs: int = -1,
    blas_threads_per_worker: int = 1,
    verbose: int = 10,
    on_error: Literal["raise", "skip"] = "skip",
) -> list[dict]:
    """Run all cells in parallel and return the rows for a sweep CSV.

    Workers are spawned via ``joblib.Parallel`` with the ``loky``
    backend. Before workers spawn, BLAS thread-count env vars are
    pinned to ``blas_threads_per_worker`` so each worker's NumPy stays
    single-threaded and the ``n_jobs`` processes don't oversubscribe
    physical cores. The previous env values are restored on return.

    Args:
        panel: Long-form panel forwarded to every cell. Pickled once
            per worker by ``loky``.
        cells: Independent units of work. See :class:`SweepCell`.
        n_jobs: Forwarded to ``joblib.Parallel``. ``-1`` uses all
            available cores; ``1`` runs sequentially (useful for
            debugging or very small grids).
        blas_threads_per_worker: Value written to ``OMP_NUM_THREADS``,
            ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
            ``NUMEXPR_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS`` for
            the duration of the call. Set to ``1`` to keep each worker
            single-threaded (the safe default when ``n_jobs > 1``).
        verbose: Forwarded to ``joblib.Parallel``. ``verbose=10``
            prints one line per completed task with elapsed time.
        on_error: ``"skip"`` (the default) catches exceptions in each
            cell, logs a warning, and drops the row. ``"raise"``
            propagates the first failure and aborts the sweep — useful
            during development.

    Returns:
        List of ``{**cell.identity, **metrics}`` dicts, in the order
        the cells were submitted. Cells that returned ``None``
        (e.g. ``min_train_obs`` filtered every OOS window) or
        that raised under ``on_error="skip"`` are omitted.
    """
    saved_env: dict[str, Optional[str]] = {
        var: os.environ.get(var) for var in _BLAS_THREAD_ENV_VARS
    }
    try:
        for var in _BLAS_THREAD_ENV_VARS:
            os.environ[var] = str(blas_threads_per_worker)
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
            delayed(_fit_one)(panel, cell, on_error) for cell in cells
        )
    finally:
        for var, prev in saved_env.items():
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev

    return [row for row in results if row is not None]
