"""Data ingestion: OpenAP signals, CRSP returns, Ken French risk-free rate.

End-to-end real-data panel build:
    python -m src.ingest.build_panel --start 199001 --end 202512

See :mod:`src.ingest.build_panel` for the orchestrator and
:class:`src.ingest.pipeline.DataPipeline` for the WRDS / OpenAP download
methods plus the model-panel helpers. The cleaning / merging primitives
that turn raw downloads into the modelling panel live in
:mod:`src.preprocess`.
"""

from src.ingest.build_panel import build_panel
from src.ingest.fama_french import (
    download_risk_free_rate,
    load_risk_free_rate_cached,
)
from src.ingest.pipeline import DataPipeline, IPCAMatrices
from src.ingest.wrds_auth import have_wrds_credentials, materialize_wrds_pgpass

__all__ = [
    "DataPipeline",
    "IPCAMatrices",
    "build_panel",
    "download_risk_free_rate",
    "have_wrds_credentials",
    "load_risk_free_rate_cached",
    "materialize_wrds_pgpass",
]
