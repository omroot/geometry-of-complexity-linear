# External input data

This directory holds raw inputs that come from outside the codebase. Code-generated
outputs live under `results/` instead.

## char_data.parquet

The real-data experiments read a characteristics panel from:

    data/char_data.parquet

(override via the `CHAR_DATA_PARQUET` env var).

### Build it from upstream sources

`src/ingest/build_panel.py` orchestrates the full download → clean → merge →
write pipeline. From the project root:

    python -m src.ingest.build_panel --start 199001 --end 202512

That runs end-to-end:

1. **OpenAP signals** — downloaded via the `openassetpricing` PyPI package
2. **CRSP S&P 500 returns** (point-in-time, no survivorship bias) — downloaded
   via the `wrds` package; requires WRDS credentials (see below)
3. **Ken French monthly risk-free rate** — fetched directly from the Tuck
   Dartmouth data library (no auth, no extra deps)
4. Drops mostly-NaN columns, merges all three sources on `(permno, yyyymm)`,
   computes `excess_ret = ret_adj - rf` and `mcap = abs(prc) * shrout`,
   imputes remaining NaNs cross-sectionally, drops low-variance and
   highly-correlated characteristics, and writes the final parquet

Intermediate downloads are cached at `cache/ingest/` so re-runs only pay for
what's changed:

    cache/ingest/openap_raw.parquet     # OpenAP signal panel
    cache/ingest/crsp_sp500_raw.parquet # CRSP S&P 500 returns + delist adj.
    cache/ingest/rf_monthly.parquet     # Ken French monthly RF

Pass `--force-refresh` to invalidate every cache and re-download. Useful
CLI flags also include `--max-nan-frac`, `--min-std`, `--max-corr`, and
`--skip-clean`; see `python -m src.ingest.build_panel --help`.

### Output schema

| Column | Type | Source |
|---|---|---|
| `permno` | int | CRSP/OpenAP |
| `yyyymm` | datetime (month-start) | CRSP/OpenAP |
| `ret_adj` | float | CRSP (return × (1 + dlret)) |
| `rf` | float | Ken French monthly T-bill |
| `excess_ret` | float | `ret_adj - rf` |
| `prc`, `shrout`, `cfacpr`, `cfacshr` | float | CRSP |
| `mcap` | float | `abs(prc) * shrout` |
| ~100-200 OpenAP characteristics | float | OpenAP (post cleaning) |

### WRDS credentials

`build_panel` resolves WRDS credentials in this order (first hit wins):

1. **`WRDS_USERNAME` and `WRDS_PASSWORD` env vars** — useful in CI or one-off shells
2. **Project-local `.env` file** at the repo root — copy `.env.example` to `.env`
   (already gitignored) and fill in your WRDS username/password. This is the
   recommended path for new contributors.
3. **`~/.pgpass`** — the wrds-package default. Set up once with
   `python -c 'import wrds; wrds.Connection()'` from an interactive shell.
4. **Interactive prompt** — last resort, only works in a foreground TTY.

When 1 or 2 supplies the credentials, a temporary pgpass is written to
`cache/ingest/.pgpass_runtime` (mode 600) and `PGPASSFILE` is set to point
at it. Both `wrds.Connection()` and OpenAP's internal WRDS calls pick it up
automatically.

### Without WRDS

The CRSP step (and OpenAP, which uses WRDS internally for some signals)
requires WRDS credentials. If you don't have an account, the build will
fail at step 1 with a setup hint. Pre-built parquets dropped manually at
`data/char_data.parquet` will continue to work — the experiments just read
the file, they don't know or care how it was produced.

### Running the real-data experiments

Once the parquet exists:

    python scripts/experiments/real_data/experiment_ppp_real.py
    # or with a custom path:
    CHAR_DATA_PARQUET=/some/other/path python scripts/experiments/real_data/experiment_ppp_real.py

The synthetic experiments do not need this parquet — they generate data in-script.
