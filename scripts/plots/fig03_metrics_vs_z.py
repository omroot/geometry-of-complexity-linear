#!/usr/bin/env python3
"""Paper Figure 3 (§8.3.3) -- geometric metrics + Sharpe co-move with z.

Left axis: d_proj, psi_max, bar psi. Right axis: annualised Sharpe.
Sweep C: k=12, m=1024, z varying.

Output: results/figures/fig03_metrics_vs_z.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import apply_paper_style, load_sweep_csv, savefig, styled_grid


def main() -> None:
    apply_paper_style()
    df = load_sweep_csv(
        "sweep_ridge.csv", "all_synthetic_datarun_results.csv",
        "synthetic_v3_sweep_C_summary.csv",
    )
    if "sweep" in df.columns:
        df = df[df["sweep"].isin(["sweep_C_fine_z", "real_ridge_sweep"]) | (df.get("k", 12) == 12)]
    sub = df.sort_values("z").groupby("z", as_index=False).mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    l1, = ax.plot(sub["z"], sub["mean_subspace_stability"], "b-o", ms=8, lw=2,
                  label=r"$d_{\mathrm{proj}}$ (subspace stability)")
    l2, = ax.plot(sub["z"], sub["mean_max_principal_angle"], "r--s", ms=8, lw=2,
                  label=r"$\psi_{\mathrm{max}}$")
    l3, = ax.plot(sub["z"], sub["mean_mean_principal_angle"], color="purple", ls=":",
                  marker="^", ms=8, lw=2, label=r"$\bar\psi$")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(sub["z"].values)
    ax.set_xticklabels([str(int(z)) if float(z).is_integer() else f"{z:g}" for z in sub["z"].values])
    ax.set_xlabel("Ridge regularisation $z$")
    ax.set_ylabel("Geometric metrics (radians)")
    styled_grid(ax)

    ax2 = ax.twinx(); ax2.grid(False)
    l4, = ax2.plot(sub["z"], sub["sharpe"], "g-D", ms=8, lw=2, alpha=0.85,
                   label="Annualised Sharpe (right)")
    ax2.set_ylabel("Annualised Sharpe ratio", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax.legend(handles=[l1, l2, l3, l4], loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_title(r"All metrics co-move with $z$ ($k=12$, $\tilde m=1024$)")
    fig.tight_layout()
    savefig(fig, "fig03_metrics_vs_z.png", dpi=200)


if __name__ == "__main__":
    main()
