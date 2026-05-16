"""The Virtue of Complexity in Asset Pricing — linear case.

Geometric framework for distinguishing virtuous from illusory complexity
in factor-model return prediction. See paper/VirtueOfComplGeometry.tex.

Subpackages:
    config: project-wide paths
    models: IPCA-Grassmannian estimator, RFF features, legacy InstrumentedPCA
    workflows: orchestration tying models to data
    backtest: rolling out-of-sample backtest harnesses (Grassmannian-IPCA + mktcap)
    sweeps: parallel-sweep workers (joblib-checkpointed)
    ingest: real-data download + cleaning (OpenAP/WRDS pipeline)
    generate: synthetic panel generator
    portfolio: portfolio construction + performance metrics
"""

from src.config import (
    DATA_DIR,
    DOCS_DIR,
    FIGURES_DIR,
    ROOT_DIR,
)
from src.ingest.pipeline import IPCAMatrices
from src.workflows.ipca import IPCAWorkflow
from src.io.naming import build_experiment_excel_name
from src.portfolio import (
    build_directional_portfolio,
    build_long_only_portfolio,
    build_quantile_portfolios,
    compute_portfolio_returns,
    plot_cumulative,
    portfolio_performance,
)
from src.models.rff import RandomFourierFeatures, RFF

__all__ = [
    "DATA_DIR",
    "DOCS_DIR",
    "FIGURES_DIR",
    "ROOT_DIR",
    "RandomFourierFeatures",
    "RFF",
    "IPCAWorkflow",
    "IPCAMatrices",
    "build_experiment_excel_name",
    "build_quantile_portfolios",
    "build_long_only_portfolio",
    "build_directional_portfolio",
    "compute_portfolio_returns",
    "portfolio_performance",
    "plot_cumulative",
]

__version__ = "0.1.0"
