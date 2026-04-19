"""Compute Area Under the Learning Curve (AULC) for training runs.

Scans for train_*.csv files, computes AULC for episode_return and
collision_rate columns, and prints per-method summary.

Usage:
    python experiments/pde/analysis/compute_aulc.py \
        --results_dir results/pde --out results/aulc_summary.csv
"""

import argparse
import os
import glob
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


def compute_aulc(df, metric_col="episode_return", step_col="step"):
    """Compute the area under the learning curve, normalized by step range.

    Parameters
    ----------
    df : pandas.DataFrame
        Training log with at least *step_col* and *metric_col*.
    metric_col : str
        Column name for the metric (e.g. "episode_return", "collision_rate").
    step_col : str
        Column name for the training step.

    Returns
    -------
    float
        AULC value (normalized by step range), or NaN if data is insufficient.
    """
    if metric_col not in df.columns or step_col not in df.columns:
        return float("nan")
    sub = df[[step_col, metric_col]].dropna().sort_values(step_col)
    if len(sub) < 2:
        return float("nan")
    steps = sub[step_col].values.astype(float)
    values = sub[metric_col].values.astype(float)
    step_range = steps[-1] - steps[0]
    if step_range <= 0:
        return float("nan")
    area = np.trapz(values, steps)
    return float(area / step_range)


def main():
    parser = argparse.ArgumentParser(
        description="Compute AULC from train_*.csv logs.")
    parser.add_argument("--results_dir", required=True,
                        help="Directory to scan for train_*.csv files")
    parser.add_argument("--out", default="results/aulc_summary.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if pd is None:
        print("pandas required: pip install pandas")
        return

    pattern = os.path.join(args.results_dir, "**", "train_*.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))
    if not csv_files:
        print(f"No train_*.csv files found in {args.results_dir}")
        return

    rows = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        basename = os.path.basename(fpath)
        # Try to extract method name from filename: train_{method}.csv or
        # train_{method}_{scenario}_{maneuver}.csv
        stem = basename.replace("train_", "").replace(".csv", "")
        method = stem.split("_")[0] if stem else basename

        aulc_return = compute_aulc(df, metric_col="episode_return", step_col="step")
        aulc_coll = compute_aulc(df, metric_col="collision_rate", step_col="step")

        rows.append({
            "file": fpath,
            "method": method,
            "aulc_return": aulc_return,
            "aulc_collision_rate": aulc_coll,
        })
        print(f"  {basename}: aulc_return={aulc_return:.4f}  "
              f"aulc_coll={aulc_coll:.4f}")

    # Write summary CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["file", "method",
                                                "aulc_return",
                                                "aulc_collision_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {args.out} ({len(rows)} entries)")

    # Per-method summary
    methods = sorted(set(r["method"] for r in rows))
    print("\n--- Per-method AULC summary ---")
    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        ret_vals = [r["aulc_return"] for r in m_rows
                    if r["aulc_return"] == r["aulc_return"]]
        coll_vals = [r["aulc_collision_rate"] for r in m_rows
                     if r["aulc_collision_rate"] == r["aulc_collision_rate"]]
        ret_str = f"{np.mean(ret_vals):.4f}" if ret_vals else "n/a"
        coll_str = f"{np.mean(coll_vals):.4f}" if coll_vals else "n/a"
        print(f"  {m}: aulc_return={ret_str}  aulc_coll={coll_str}  "
              f"(n={len(m_rows)} files)")


if __name__ == "__main__":
    main()
