"""Compute computational overhead per method.

Reads timing information from training CSVs and reports
mean wall-time per 1000 training steps for each method.

Usage:
    python experiments/pde/analysis/compute_overhead.py \
        --results_dir results/ablation/tier1
"""

import argparse
import os
import glob
import numpy as np

METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux"]
METHOD_LABELS = {
    "drppo": "DRPPO (baseline)",
    "hjb_aux": "Hard-HJB",
    "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux": "Eikonal",
    "cbf_aux": "CBF-PDE",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out", default=None, help="Output CSV path (optional)")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: pip install pandas")
        return

    results = {}
    for method in METHODS:
        pattern = os.path.join(args.results_dir, "**", f"train_{method}_*.csv")
        csv_files = glob.glob(pattern, recursive=True)
        times = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if "train_time_per_iter" in df.columns:
                    times.extend(df["train_time_per_iter"].dropna().tolist())
            except Exception:
                continue
        if times:
            results[method] = {"mean": np.mean(times), "std": np.std(times), "n": len(times)}

    if not results:
        print(f"No timing data found in {args.results_dir}")
        return

    baseline = results.get("drppo", {}).get("mean", 1.0)

    print(f"\n{'Method':<25} {'Time/iter (s)':<20} {'Relative to DRPPO':<20} {'N':<5}")
    print("-" * 70)
    for method in METHODS:
        if method not in results:
            continue
        r = results[method]
        rel = r["mean"] / baseline if baseline > 0 else 0
        label = METHOD_LABELS.get(method, method)
        print(f"{label:<25} {r['mean']:.1f} +/- {r['std']:.1f}       {rel:.2f}x              {r['n']}")

    if args.out:
        import csv
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "mean_time_per_iter", "std_time_per_iter", "relative_to_drppo", "n_samples"])
            for method in METHODS:
                if method not in results:
                    continue
                r = results[method]
                rel = r["mean"] / baseline if baseline > 0 else 0
                w.writerow([method, f"{r['mean']:.2f}", f"{r['std']:.2f}", f"{rel:.2f}", r["n"]])
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
