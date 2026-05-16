# External input data

This directory holds raw inputs that come from outside the codebase (downloads,
parquet drops, etc.). Code-generated outputs live under `results/` instead.

## char_data.parquet

The real-data experiment needs a characteristics panel at:

    data/char_data.parquet

The original notebook hardcoded a co-author path
(`/Users/apple/Desktop/Academics/char_data.parquet`); the refactored copy reads
the path from the `CHAR_DATA_PARQUET` env var, defaulting to
`data/char_data.parquet` (relative to the project root).

Drop a real parquet at that path, then run either:

    jupyter nbconvert --execute notebooks/experiments/synthetic/experiment_ppp.ipynb
    # or, as a script:
    CHAR_DATA_PARQUET=/some/other/path python scripts/experiments/real_data/experiment_ppp_real.py

Note: with `USE_SYNTHETIC_DATA = True` (the notebook's default), the parquet is
not required — synthetic data is generated in-notebook. The parquet is only
consumed when switching to real-data runs (`USE_SYNTHETIC_DATA = False`).

The original `refactored/` checkout had a `char_data.parquet` symlink pointing
at an external `omroot_repos/TheGeometryOfVirtueOfComplexity_LinearCase/...`
path. That symlink was dropped from this project to keep it portable. Drop a
real parquet here (or point `CHAR_DATA_PARQUET` at one) before running
real-data mode.
