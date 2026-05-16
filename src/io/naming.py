"""Output-file naming conventions used across the experiment scripts."""

from __future__ import annotations


def build_experiment_excel_name(
    base_name: str,
    n_factors: int | str,
    train_window_months: int | str,
    data_label: str | None = None,
) -> str:
    """Create a consistent Excel filename for one experiment.

    Examples
    --------
    ``portfolio_performance_18_36.xlsx``
    ``portfolio_performance_synthetic_18_36.xlsx``

    The optional ``data_label`` is omitted for the default real-data aliases so
    existing filenames stay unchanged.
    """
    safe_base = "_".join(str(base_name).strip().split())
    label = None if data_label is None else "_".join(str(data_label).strip().split())
    if label and label.lower() not in {"all", "all_data", "real", "openap"}:
        return f"{safe_base}_{label}_{n_factors}_{train_window_months}.xlsx"
    return f"{safe_base}_{n_factors}_{train_window_months}.xlsx"
