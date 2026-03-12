"""Merge per-job result CSVs into single ablation_results.csv and ablation_train_log.csv.

After running parallel jobs (e.g. via launch_parallel_16gpu.sh), job outputs live in
results/ablation/jobs/eval_*.csv and results/ablation/jobs/train_*.csv. This script
concatenates them into results/ablation/ablation_results.csv and
results/ablation/ablation_train_log.csv so the existing dashboard and plotting work.

Also validates completeness: checks that every job in the manifest produced output.

Usage:
  python experiments/aggregate_results.py --out_dir results/ablation
  python experiments/aggregate_results.py --out_dir results/ablation --manifest results/ablation/job_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys


def merge_csvs(
    pattern: str,
    out_path: str,
    fieldnames: list[str] | None = None,
) -> int:
    """Glob pattern for CSVs, merge into out_path. Returns number of data rows written."""
    files = sorted(glob.glob(pattern))
    if not files:
        return 0
    rows = []
    for path in files:
        with open(path) as f:
            r = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = r.fieldnames
            for row in r:
                rows.append(row)
    if not rows:
        return 0
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _csv_data_rows(path: str) -> int:
    """Return number of data rows (excluding header) in a CSV. Returns 0 if missing or empty."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 0
    with open(path) as f:
        r = csv.DictReader(f)
        return sum(1 for _ in r)


def validate_completeness(manifest_path: str, jobs_dir: str) -> tuple[int, int, list[int]]:
    """Check which manifest jobs produced valid eval and train CSVs (present, non-empty, >=1 row).
    Returns (n_expected, n_found, missing_or_incomplete_ids)."""
    with open(manifest_path) as f:
        data = json.load(f)
    n_expected = data["n_jobs"]
    missing = []
    for job_id in range(n_expected):
        eval_csv = os.path.join(jobs_dir, f"eval_{job_id:06d}.csv")
        train_csv = os.path.join(jobs_dir, f"train_{job_id:06d}.csv")
        if _csv_data_rows(eval_csv) < 1 or _csv_data_rows(train_csv) < 1:
            missing.append(job_id)
    return n_expected, n_expected - len(missing), missing


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-job ablation CSVs into single files")
    parser.add_argument("--out_dir", default="results/ablation")
    parser.add_argument("--jobs_dir", default=None, help="Defaults to out_dir/jobs")
    parser.add_argument("--manifest", default=None,
                        help="Path to job_manifest.json for completeness validation")
    args = parser.parse_args()

    jobs_dir = args.jobs_dir or os.path.join(args.out_dir, "jobs")
    manifest = args.manifest or os.path.join(args.out_dir, "job_manifest.json")
    eval_pattern = os.path.join(jobs_dir, "eval_*.csv")
    train_pattern = os.path.join(jobs_dir, "train_*.csv")

    os.makedirs(args.out_dir, exist_ok=True)
    results_csv = os.path.join(args.out_dir, "ablation_results.csv")
    train_csv = os.path.join(args.out_dir, "ablation_train_log.csv")

    missing: list[int] = []
    if os.path.isfile(manifest):
        n_expected, n_found, missing = validate_completeness(manifest, jobs_dir)
        print(f"Manifest: {n_expected} jobs, {n_found} completed, {len(missing)} missing or incomplete")
        if missing:
            print(f"Missing/incomplete job IDs: {missing[:20]}{'...' if len(missing) > 20 else ''}", file=sys.stderr)
            print("Aggregation failed: run workers to completion or fix failed jobs.", file=sys.stderr)
    else:
        print(f"No manifest found at {manifest}; skipping completeness check.")

    eval_fields = [
        "scenario", "variant", "pinn_placement", "use_l_ego", "use_safety_filter",
        "lambda_phys", "seed", "eval_mode",
        "mean_return", "std_return", "collision_rate", "pothole_hits_mean",
        "mean_ttc", "min_ttc",
    ]
    train_fields = [
        "scenario", "variant", "lambda_phys", "seed", "step",
        "actor_loss", "vf_loss", "entropy", "total_loss",
        "l_physics", "l_actor_physics", "l_ego",
        "viol_ttc_rate", "viol_stop_rate", "viol_fric_rate",
        "viol_ttc_mag", "viol_stop_mag", "viol_fric_mag",
    ]

    n_eval = merge_csvs(eval_pattern, results_csv, fieldnames=eval_fields)
    n_train = merge_csvs(train_pattern, train_csv, fieldnames=train_fields)

    print(f"Aggregated {n_eval} eval rows -> {results_csv}")
    print(f"Aggregated {n_train} train rows -> {train_csv}")

    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
    sys.exit(0)
