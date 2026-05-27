# geometry-of-complexity-linear

Codebase for the **linear** track of *[The Virtue of Complexity: A
Geometric View](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6798000)*,
by **Arsh Deep, Andrew Lesniewski, Oualid Missaoui, and Chaithanya Pakala**
(SSRN: 6798000).

## Setup

    conda create -n geometry-of-complexity-linear python=3.10
    conda activate geometry-of-complexity-linear
    pip install -r requirements.txt

WRDS credentials are needed only for the real-data ingest step (below).
Synthetic experiments run without them.

## Repository layout

    src/
      config.py        Project paths and defaults.
      diagnostics/     Principal angles, anisotropy, effective rank,
                       spectral gap, geodesic acceleration.
      generate/        Paper §8.1 synthetic-data generator + workflow adapter.
      ingest/          OpenAP / WRDS / Ken French downloads + WRDS auth.
      preprocess/      Pure DataFrame-in/DataFrame-out cleaning + merging.
      models/          Grassmannian-IPCA estimator (pymanopt), RFF,
                       rolling-window fit.
      portfolio/       Directional / long-only / quantile construction
                       and Sharpe/max-DD/hit-rate metrics.
      sweep/           Shared run_one_fit, real-panel loader, paper
                       §8.1 DGP defaults.

    scripts/
      00_build_panel.py                     Build the real-data parquet.
      experiments/synthetic/01_smoke_panel.py    Sanity-check the DGP.
      experiments/synthetic/02..05_sweep_*.py    Paper §8.2 sweeps (A-D).
      experiments/synthetic/06_short_window.py   Paper §8.5 (k0=18) sweep.
      experiments/real_data/01_load_panel.py     Sanity-check the parquet.
      experiments/real_data/02_baseline_grid.py  §9.1-§9.9 baseline grid.
      experiments/real_data/03_ridge_sweep.py    §9.5 finer-z slice.
      experiments/real_data/04_extended_grid.py  §9.10 extended backtest.
      plots/fig01_*.py ... fig20_*.py            Paper figures.
      plots/make_all.py                          Render every figure.

    data/
      raw/             Raw upstream parquets (gitignored, ~2.3 GB).
      preprocessed/    char_data.parquet (gitignored, ~114 MB).

    results/
      data/            Sweep CSVs consumed by figure scripts.
      figures/         Paper-figure PDFs/PNGs.

## Build the real-data modelling panel

    python scripts/00_build_panel.py

End-to-end download → merge → impute → prune → write. Produces
`data/preprocessed/char_data.parquet` (~114 MB; 210k rows, 119
characteristics, S&P 500 constituents, 1990-01..2024-12). First run
takes ~5 min (dominated by the OpenAP signals download). Subsequent
runs hit the per-source cache under `data/raw/` and finish in seconds.

WRDS credentials must be available via one of:

* `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables,
* a project-root `.env` file with the same keys (gitignored), or
* an existing `~/.pgpass` entry for `wrds-pgdata.wharton.upenn.edu`.

The script reads credentials from `.env` first; if not present, the
materialized runtime pgpass lands at `data/raw/.pgpass_runtime`
(mode 600). Pass `--force-refresh` to invalidate every cached download;
`--help` lists all flags.

## Synthetic experiments

The §8 broad-grid sweeps run on the paper §8.1 DGP — no real data
needed.

    # Paper §8.2 broad-grid sweeps (A-D), feeding Figs 1-7 + Table 1.
    python scripts/experiments/synthetic/02_sweep_factor_dim.py   # (k, z) at m=1024
    python scripts/experiments/synthetic/03_sweep_ridge.py        # finer-z at k=12
    python scripts/experiments/synthetic/04_sweep_rff_dim.py      # vary RFF m
    python scripts/experiments/synthetic/05_sweep_pgrid.py        # (m, k, z)

    # Paper §8.5 short-window (k0=18, rank-aligned), feeding Figs 8-11 + Table 2.
    python scripts/experiments/synthetic/06_short_window.py

Each script writes one CSV under `results/data/`. The figure scripts
read those CSVs.

## Real-data experiments

After building the parquet (above):

    # §9.1-§9.9 baseline (top-50 fixed universe, 88-config grid; Figs 12, 14-16, Tables 3-4).
    python scripts/experiments/real_data/02_baseline_grid.py

    # §9.5 ridge slice at k=12, m=1024 (3 seeds per z; Fig 13).
    python scripts/experiments/real_data/03_ridge_sweep.py

    # §9.10 extended backtest (rolling top-50 by previous-month mcap; Figs 17-20).
    python scripts/experiments/real_data/04_extended_grid.py

Each writes one CSV under `results/data/`.

## Paper figures

After the sweep CSVs exist:

    python scripts/plots/make_all.py                # render every fig*.py
    python scripts/plots/make_all.py --only fig13   # one figure
    python scripts/plots/fig13_ridge_sweep.py       # equivalent direct call

Figures land under `results/figures/` as `fig01_*.pdf` ... `fig20_*.{pdf,png}`.

## Environment overrides

| Variable | Purpose | Default |
|---|---|---|
| `CHAR_DATA_PARQUET` | Modelling-panel parquet read by experiment scripts | `data/preprocessed/char_data.parquet` |
| `GOCL_DATA_DIR` | Where figure scripts look for sweep CSVs | `results/data/` |
| `FIGURES_DIR` | Where rendered figures land | `results/figures/` |

The raw-downloads directory is configurable via `--raw-dir` on the
build script (default `data/raw/`).

## Adding a new sweep

`run_one_fit` in `src/sweep/runner.py` packages the
`rolling_fit → diagnostics → portfolio → metrics` chain into one call.
A new sweep is typically 30-50 lines: define cell-shape constants,
loop over cells, call `run_one_fit`, write CSV. The existing scripts
under `scripts/experiments/synthetic/` and `scripts/experiments/real_data/`
are the templates.
