"""The Virtue of Complexity in Asset Pricing — linear case.

Geometric framework for distinguishing virtuous from illusory complexity
in factor-model return prediction. See paper/geometry_of_complexity_linear.tex.

Active subpackages:
    config: project-wide paths
    generate: synthetic panel generator (paper §8.1 DGP + workflow adapter)
    models: Grassmannian-IPCA estimator, RFF features, rolling-fit primitive
    diagnostics: geometric primitives (principal angles, anisotropy, erank,
        spectral gap, curvature) and per-window aggregation
    portfolio: directional/long-only/quantile portfolios + performance metrics
    ingest: real-data download + cleaning (OpenAP/WRDS pipeline)
"""

from src.config import (
    DATA_DIR,
    DOCS_DIR,
    FIGURES_DIR,
    ROOT_DIR,
)
from src.ingest.pipeline import IPCAMatrices
from src.models.rff import RandomFourierFeatures, RFF
from src.portfolio import (
    build_directional_portfolio,
    build_long_only_portfolio,
    build_quantile_portfolios,
    compute_portfolio_returns,
    plot_cumulative,
    portfolio_performance,
)

__all__ = [
    "DATA_DIR",
    "DOCS_DIR",
    "FIGURES_DIR",
    "ROOT_DIR",
    "RandomFourierFeatures",
    "RFF",
    "IPCAMatrices",
    "build_quantile_portfolios",
    "build_long_only_portfolio",
    "build_directional_portfolio",
    "compute_portfolio_returns",
    "portfolio_performance",
    "plot_cumulative",
]

__version__ = "0.1.0"
