"""Project paths and shared defaults."""

from pathlib import Path


# Project root (parent of src/)
ROOT_DIR = Path(__file__).parent.parent

# Tracked outputs (csv, xlsx, paper figures, diagnostics)
RESULTS_DIR = ROOT_DIR / "results"

# Frozen csv inputs that drive the analytical figure pipeline
DATA_DIR = RESULTS_DIR / "data"

# Cached xlsx results from past IPCA experiments
XLSX_DIR = RESULTS_DIR / "data" / "xlsx"

# Paper figures
FIGURES_DIR = RESULTS_DIR / "figures"

# Documentation directory
DOCS_DIR = ROOT_DIR / "docs"


# Random Fourier Features defaults (consumed by src.models.rff)
DEFAULT_RFF_GAMMA = 2.0      # bandwidth parameter
DEFAULT_RFF_N_FEATURES = 6000  # number of features (P = 2 * n_features)
