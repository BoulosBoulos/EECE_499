"""Job 5: computational overhead per method from existing metrics.csv timing columns.

No new timing runs — aggregates the per-iteration timing already logged during
Tier-1 training. Statistical unit is the training run (mean over its iterations),
then mean +/- bootstrap CI across runs per method.

Primary, hardware-invariant metric:
  residual_frac = residual_compute_time_seconds / iter_time_seconds
    (within-run ratio, robust to the mixed cmu-node hardware).

Secondary, hardware-CONFOUNDED metric (labeled as such):
  wall/iter time ratio vs DRPPO baseline.

Usage:
  python scripts/aggregate_job5_overhead.py --repo . --out results/tables/computational_overhead.csv
"""

import argparse
import csv
import os

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "fusion_aux", "cbf_aux"]
METHOD_LABELS = {
    "drppo": "DRPPO", "hjb_aux": "Hard-HJB", "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux": "Eikonal", "fusion_aux": "Fusion", "cbf_aux": "CBF-PDE",
}
TIMING_COLS = ["iter_time_seconds", "env_step_time_seconds",
               "learn_step_time_seconds", "residual_compute_time_seconds"]


def _ci95(vals):
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    boots = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(2000)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _run_timing(metrics_path):
    """Mean per-iteration timing for one run. Skips warmup iteration 0."""
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return None
    if df.empty or "iter_time_seconds" not in df.columns:
        return None
    if "iteration" in df.columns and len(df) > 1:
        df = df[df["iteration"] > 0]  # drop warmup/first iter timing spike
    if df.empty:
        return None
    out = {}
    for c in TIMING_COLS:
        out[c] = float(df[c].mean()) if c in df.columns else float("nan")
    it = out.get("iter_time_seconds", float("nan"))
    rc = out.get("residual_compute_time_seconds", float("nan"))
    out["residual_frac"] = (rc / it) if (it and it == it and it > 0) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/tables/computational_overhead.csv")
    args = ap.parse_args()
    if pd is None:
        print("pandas required")
        return

    t1_root = os.path.join(args.repo, "results", "tier_1_full")
    # Collect per-run timing keyed by method
    by_method = {m: [] for m in METHODS}
    for run in sorted(os.listdir(t1_root)):
        # method is the token before _intent/_nointent
        method = None
        for m in sorted(METHODS, key=len, reverse=True):
            if f"_{m}_intent_" in run or f"_{m}_nointent_" in run:
                method = m
                break
        if method is None:
            continue
        stats = _run_timing(os.path.join(t1_root, run, "metrics.csv"))
        if stats is not None:
            by_method[method].append(stats)

    # DRPPO baseline mean iter time (hardware-confounded normalizer)
    drppo_iter = np.nanmean([r["iter_time_seconds"] for r in by_method["drppo"]]) \
        if by_method["drppo"] else float("nan")

    rows = []
    for m in METHODS:
        runs = by_method[m]
        if not runs:
            continue
        n = len(runs)
        iter_t = [r["iter_time_seconds"] for r in runs]
        learn_t = [r["learn_step_time_seconds"] for r in runs]
        res_t = [r["residual_compute_time_seconds"] for r in runs]
        res_frac = [r["residual_frac"] for r in runs]
        mean_iter = float(np.nanmean(iter_t))
        rf_lo, rf_hi = _ci95(res_frac)
        rows.append({
            "method": METHOD_LABELS.get(m, m),
            "n_runs": n,
            "mean_iter_time_s": round(mean_iter, 4),
            "mean_learn_step_time_s": round(float(np.nanmean(learn_t)), 4),
            "mean_residual_compute_time_s": round(float(np.nanmean(res_t)), 5),
            "residual_frac_of_iter": round(float(np.nanmean(res_frac)), 5),
            "residual_frac_ci_low": round(rf_lo, 5),
            "residual_frac_ci_high": round(rf_hi, 5),
            "walltime_ratio_vs_drppo_HW_CONFOUNDED": round(mean_iter / drppo_iter, 3)
            if drppo_iter and drppo_iter == drppo_iter else float("nan"),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {args.out} ({len(rows)} methods)")
    for r in rows:
        print(f"  {r['method']:10s} n={r['n_runs']:3d}  "
              f"resid_frac={r['residual_frac_of_iter']:.4f}  "
              f"iter={r['mean_iter_time_s']:.3f}s  "
              f"HWratio={r['walltime_ratio_vs_drppo_HW_CONFOUNDED']}")


if __name__ == "__main__":
    main()
