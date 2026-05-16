# geometry-of-complexity-linear

Self-contained subproject for the **linear** track of *The Virtue of Complexity:
A Geometric View*, by **Arsh Deep, Andrew Lesniewski, Oualid Missaoui, and Chaithanya Pakala**.

See `git log` for the refactor history (baseline → artifact separation →
src consolidation → restructure by purpose → entry-script simplification).

Override default paths with env vars: `GOCL_CACHE_DIR`, `GOCL_DATA_DIR`,
`XLSX_DIR`, `FIGURES_DIR`. Experiment scripts also honor `GOCL_K` / `GOCL_T`
to override the default `RUN_CONFIGS` list with a single `(k, T)` cell.

## Setup

Recommended: create a conda environment named after this repo.

    conda create -n geometry-of-complexity-linear python=3.10
    conda activate geometry-of-complexity-linear
    pip install -r requirements.txt

## Regenerate figures from frozen artifacts

Run from the project root. Each script reads frozen csv/xlsx from `results/data/`
and writes figures to `results/figures/`:

    python scripts/plots/generate_paper_figures.py    # §8 + §9 figs from csv
    python scripts/plots/graphs_synthetic.py          # figs 8-11 from xlsx
    python scripts/plots/graphs_historical.py         # figs 18-20 + portfolio_perf

`ext_plotA.pdf` (fig 17) is already in `results/figures/` and is only regenerated
when the experiment runs end-to-end (last cell of `experiment_ppp.ipynb`).

## Run experiments

The canonical sources are the notebooks under `notebooks/experiments/`. Open the
relevant one in Jupyter and run cells, or invoke its jupytext-paired `.py`:

    # Synthetic IPCA workflow (default: 5 (k,T) configs)
    python scripts/experiments/synthetic/experiment_ppp.py

    # Single (k, T) only (e.g., k=12, T=24)
    GOCL_K=12 GOCL_T=24 python scripts/experiments/synthetic/experiment_ppp.py

    # Real-data IPCA workflow (needs data/char_data.parquet)
    python scripts/experiments/real_data/experiment_ppp_real.py

    # v3 broad-grid synthetic sweep (writes results/data/csv/; 12h+)
    python scripts/experiments/synthetic/rerun_synthetic_v3_checkpointed.py \
        --sweeps A,B,C,D --overwrite-main

The interactive RFF sweep (`notebooks/experiments/real_data/experiment_rff_sweep.ipynb`)
must be run in Jupyter — it appends one row at a time to `all_rff_results.csv`.

## Smoke-test the code

See [`tests/README.md`](tests/README.md). Quickest test:

    python tests/smoke_test_one_job.py

## Real-data Tier C

Requires `char_data.parquet` at `data/`. See [`data/README.md`](data/README.md).
Synthetic mode is the default and needs nothing extra.
