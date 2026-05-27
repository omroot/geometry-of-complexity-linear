"""Synthetic-data generators (paper §8 DGP).

One DGP, two entry points — see :mod:`src.generate.synthetic` for details.
"""

from src.generate.synthetic import (
    generate_synthetic_panel,
    generate_workflow_panel,
)

__all__ = [
    "generate_synthetic_panel",
    "generate_workflow_panel",
]
