"""Preprocessing: raw OpenAP/CRSP/RF inputs -> modelling characteristics panel.

Pure DataFrame-in / DataFrame-out transformations. All IO (downloads,
parquet reads/writes) lives in :mod:`src.ingest`.
"""

from src.preprocess.panel import (
    build_char_panel,
    drop_low_std_and_high_corr,
    fill_remaining_missing,
    merge_openap_with_crsp,
    merge_risk_free_rate,
    remove_mostly_nan_columns,
)

__all__ = [
    "build_char_panel",
    "drop_low_std_and_high_corr",
    "fill_remaining_missing",
    "merge_openap_with_crsp",
    "merge_risk_free_rate",
    "remove_mostly_nan_columns",
]
