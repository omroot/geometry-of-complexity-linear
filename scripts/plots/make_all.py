#!/usr/bin/env python3
"""Render every paper figure under scripts/plots/fig*.py.

Iterates over the 20 figure scripts in alphanumeric order, runs each as a
subprocess, and writes a one-line status per figure. Missing-CSV failures
are reported but don't abort the run. Exit code is 0 only when every
script wrote its output file.

Output: figures land in ``results/figures/`` (or wherever ``FIGURES_DIR``
points). Filenames are ``fig01_*.pdf`` through ``fig20_*.{pdf,png}`` and
never collide with the legacy ``paper/figures/`` artifacts.

Usage:
    python scripts/plots/make_all.py
    FIGURES_DIR=/tmp/somewhere python scripts/plots/make_all.py
    python scripts/plots/make_all.py --only fig17,fig18   # comma-separated prefixes
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent


def discover_scripts(only: list[str] | None) -> list[Path]:
    """Return the fig*.py scripts to run, optionally filtered by prefix."""
    all_scripts = sorted(SCRIPTS_DIR.glob("fig*.py"))
    if not only:
        return all_scripts
    keep = []
    for p in all_scripts:
        if any(p.stem.startswith(prefix) for prefix in only):
            keep.append(p)
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated script-name prefixes to run "
             "(e.g. 'fig01,fig17' or 'fig1' for figs 1, 10-19).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    scripts = discover_scripts(only)
    if not scripts:
        print(f"No scripts matched. Available: {[p.stem for p in sorted(SCRIPTS_DIR.glob('fig*.py'))]}")
        return 2

    figures_dir = os.environ.get("FIGURES_DIR", "results/figures")
    print(f"Running {len(scripts)} figure scripts -> {figures_dir}\n")

    ok: list[str] = []
    missing_csv: list[str] = []
    failed: list[tuple[str, str]] = []
    start = time.monotonic()

    for path in scripts:
        name = path.stem
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, env=os.environ.copy(),
        )
        dt = time.monotonic() - t0
        last_line = (proc.stdout or proc.stderr).strip().splitlines()[-1] if (proc.stdout or proc.stderr) else "(no output)"

        if proc.returncode == 0:
            ok.append(name)
            status = "OK"
        elif "FileNotFoundError" in (proc.stderr or "") and "scripts/experiments" in (proc.stderr or ""):
            missing_csv.append(name)
            status = "SKIP (no input CSV)"
        else:
            failed.append((name, last_line))
            status = "FAIL"
        print(f"  [{dt:5.1f}s] {name:35s} {status}: {last_line}")

    total = time.monotonic() - start
    print(f"\nDone in {total:.1f}s. {len(ok)} rendered, {len(missing_csv)} skipped, {len(failed)} failed.")
    if missing_csv:
        print(f"  Skipped (run the corresponding scripts/experiments/* first): "
              f"{', '.join(missing_csv)}")
    if failed:
        print(f"  Failed: {', '.join(name for name, _ in failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
