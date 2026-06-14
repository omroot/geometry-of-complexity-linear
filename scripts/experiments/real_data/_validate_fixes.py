#!/usr/bin/env python3
"""Validation run: k=12, P=64, z=20 — both fixes applied.
Checks geodesic_accel + organic 47-permno filter against VoC reference."""
import sys, time, numpy as np, pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VOC_ROOT  = REPO_ROOT.parent / "TheVirtueOfComplexity_PaperReplication_Experimentation"
sys.path.insert(0, str(REPO_ROOT))

from src.sweep import FORECAST_START, RFF_GAMMA, TRAIN_WINDOW_MONTHS, run_one_fit
from src.sweep.runner import load_real_panel, load_voc_top50_permnos

voc = pd.read_csv(VOC_ROOT / "data" / "all_rff_results.csv")
ref = voc[(voc.k==12)&(voc.P==64)&(voc.z==20.0)].iloc[0]
print(f"VoC ref (k=12, P=64, z=20, label={ref.run_label}):")
print(f"  sharpe={ref.avg_sharpe:.4f}  r2={ref.avg_r2_oos:.4f}  "
      f"erank={ref.avg_erank:.4f}  sg={ref.avg_spectral_gap:.4f}  "
      f"geo_accel={ref.avg_geodesic_accel:.4f}  "
      f"subspace_stab={ref.avg_subspace_stability:.4f}")

voc_permnos = load_voc_top50_permnos()
panel = load_real_panel(permno_list=voc_permnos)
print(f"\nPanel: {panel['permno'].nunique()} permnos (50 loaded, filter will drop 3→47)")

K, RFF_N, Z = 12, 32, 20.0   # P_old=64 → rff_n_components=32

results = []
for seed in [0, 1, 2]:
    t0 = time.time()
    m = run_one_fit(
        panel, k=K, train_window_months=TRAIN_WINDOW_MONTHS,
        forecast_start=FORECAST_START, rff_n_components=RFF_N,
        rff_gamma=RFF_GAMMA, rff_seed=seed, alpha=Z,
        cross_sectional_normalize=False, voc_rectangular=True,
        rectangular_ids=voc_permnos, cross_sectional_ddof=1,
        voc_max_nan_frac=0.3, voc_max_zero_frac=1.0, min_train_obs=50*6,
    )
    results.append(m)
    print(f"  seed={seed}: sharpe={m['sharpe']:+.4f}  r2={m['oos_r2']:.4f}  "
          f"erank={m['mean_erank']:.4f}  sg={m['mean_spectral_gap']:.4f}  "
          f"geo_accel={m['mean_geodesic_accel']:.4f}  ({time.time()-t0:.0f}s)")

keys = ["sharpe","oos_r2","mean_erank","mean_spectral_gap",
        "mean_subspace_stability","mean_max_principal_angle",
        "mean_mean_principal_angle","mean_geodesic_accel"]
avgs = {k: float(np.mean([r[k] for r in results])) for k in keys}
voc_vals = {
    "sharpe": ref.avg_sharpe, "oos_r2": ref.avg_r2_oos,
    "mean_erank": ref.avg_erank, "mean_spectral_gap": ref.avg_spectral_gap,
    "mean_subspace_stability": ref.avg_subspace_stability,
    "mean_max_principal_angle": ref.avg_max_principal_angle,
    "mean_mean_principal_angle": ref.avg_mean_principal_angle,
    "mean_geodesic_accel": ref.avg_geodesic_accel,
}

print("\n=== 3-seed average vs VoC reference ===")
for metric, nv in avgs.items():
    vv = voc_vals[metric]
    pct = 100*(nv-vv)/abs(vv) if abs(vv) > 1e-6 else float("nan")
    flag = "✅" if abs(pct) < 15 else ("⚠️ " if abs(pct) < 35 else "❌")
    print(f"  {metric:<30} new={nv:+.4f}  voc={vv:+.4f}  {pct:+.1f}% {flag}")
