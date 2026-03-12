"""Merge ablation_results.csv and ablation_train_log.csv from batch1 and batch2 into results/ablation/.

Use after running ablation in two scenario batches (batch1: 1a,1b,1c,1d; batch2: 2,3,4)
and aggregating each. Writes merged CSVs to results/ablation/ for dashboard and plotting.

Usage:
  python3 experiments/merge_ablation_batches.py
  python3 experiments/merge_ablation_batches.py --batch1 results/ablation_batch1 --batch2 results/ablation_batch2 --out_dir results/ablation
"""

from __future__ import annotations

import argparse
import csv
import os


def merge_csvs(path1: str, path2: str, out_path: str) -> int:
    """Concatenate two CSVs (same header) into out_path. Returns total rows."""
    if not os.path.isfile(path1):
        raise FileNotFoundError(path1)
    if not os.path.isfile(path2):
        raise FileNotFoundError(path2)
    rows = []
    fieldnames = None
    for path in (path1, path2):
        with open(path) as f:
            r = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = r.fieldnames
            for row in r:
                rows.append(row)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Merge two ablation batch CSVs into one out_dir")
    parser.add_argument("--batch1", default="results/ablation_batch1")
    parser.add_argument("--batch2", default="results/ablation_batch2")
    parser.add_argument("--out_dir", default="results/ablation")
    args = parser.parse_args()

    n_eval = merge_csvs(
        os.path.join(args.batch1, "ablation_results.csv"),
        os.path.join(args.batch2, "ablation_results.csv"),
        os.path.join(args.out_dir, "ablation_results.csv"),
    )
    n_train = merge_csvs(
        os.path.join(args.batch1, "ablation_train_log.csv"),
        os.path.join(args.batch2, "ablation_train_log.csv"),
        os.path.join(args.out_dir, "ablation_train_log.csv"),
    )
    print(f"Merged {n_eval} eval rows and {n_train} train rows -> {args.out_dir}/")


if __name__ == "__main__":
    main()
