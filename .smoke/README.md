# .smoke/ — pipeline-completes harness

Tiny-config end-to-end run of the four producers and three plotters that
build `results/`. Outputs go to `smoke_results/` (gitignored), not
`results/`. Runtime target: ~15 minutes.

## Usage

From the project root:

```bash
python .smoke/run_all.py
```

Exits 0 if every step (producer or plotter) ran to completion and
produced the expected output files. Verification is structural only —
no numerical comparison against the frozen `results/`.

## What runs vs. what's skipped

| Step | Status |
|---|---|
| `rerun_synthetic_v3_checkpointed.py --smoke` (sweeps A,B,C,D) | runs |
| `experiment_ppp.py` (synthetic, k=2, T=24) | runs |
| `experiment_ppp_real.py` | runs **iff** `data/char_data.parquet` exists; otherwise skipped |
| `experiment_rff_sweep.py` | **always skipped** — 2500-line interactive notebook; orchestrator copies the frozen `results/data/csv/all_rff_results.csv` so the historical plotters have their input |
| `generate_paper_figures.py` | runs |
| `graphs_synthetic.py` | runs |
| `graphs_historical.py` | runs |

## How it scales the science down

The producers honor `GOCL_SMOKE=1` (synthetic + real `experiment_ppp.py`)
and `--smoke` (v3 sweep). Concretely, smoke mode:

- v3 sweep: 60-period × 20-asset × 10-char panel; tiny single-cell
  `SMOKE_SWEEPS` for A/B/C/D defined in the v3 script
- `experiment_ppp.py` (both variants): `SYNTHETIC_CONFIG` overridden to
  T=60 / N=20 / m=10; `FULL_SEEDS=[10, 20]`, one large seed, RFF
  components `[8, 16]` small + `[32]` large; `MIN_TRAIN_OBS=24`

The single source of truth for paths and env vars is
[`config.py`](config.py). Each subprocess is invoked from project root
with `XLSX_DIR`, `FIGURES_DIR`, `GOCL_DATA_DIR`, `GOCL_CACHE_DIR`,
`GOCL_K`, `GOCL_T`, and `GOCL_SMOKE` set.
