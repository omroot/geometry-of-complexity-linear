# Smoke tests

Three smoke paths, ordered from fastest to slowest. Goal is to verify the code
*executes* end-to-end; figure reproduction is a side effect.

All commands run from the project root.

## 1. From-cache figure rebuild (~30 s)

    python reproduce_all.py --from-cache

Exercises all 20 plot scripts against frozen artifacts (csv + xlsx + the static
`ext_plotA.pdf`). Regenerates every figure into `results/figures/`. Heavyweight
smoke for the plotting layer; does NOT rerun the heavy ML.

## 2. Tier A — single-job analytical chain (~30s–a few minutes)

    python tests/smoke_test_one_job.py

Runs one `(k=12, P=1024, z=10)` job through:
`generate_synthetic_panel` → RFF features → Grassmannian-IPCA backtest → aggregation.
Writes a one-row summary to `artifacts/smoke/tier_a/smoke_test_output.csv`
(override with `GOCL_ARTIFACTS_DIR`). Joblib checkpoints land in
`cache/smoke_test_checkpoints/` (override with `GOCL_CACHE_DIR`). Does NOT touch
the canonical `all_synthetic_datarun_results.csv`.

## 3. Tier B/C synthetic chain (tens of minutes)

    python tests/smoke_test_tier_bc.py

Runs four jobs (baseline IPCA + 2 small RFF + 1 large RFF) through
`GrassmannianIPCAWorkflow.rolling_ipca_predictions`, writes an xlsx via
`summarize_experiment`, and renders one ext_plotA chart. Outputs land in
`artifacts/smoke/tier_bc/` (override with `GOCL_ARTIFACTS_DIR`).

## Skipped: Tier B/C real-data smoke

`tests/smoke_test_tier_bc_real.py` is intentionally not in the standard
set. It needs `data/char_data.parquet` (not shipped — see `data/README.md`) and
the underlying real-data Grassmannian-IPCA fit is a known blocker. See the long
note in `../refactored/README.md` for context.
