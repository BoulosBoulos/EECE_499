#!/usr/bin/env python3
"""Aggregate per-machine Tier 1 results into a unified results/tier_1/.

Usage:
    python3 experiments/pde/aggregate_tier_1_results.py \\
        --source_dirs results/tier_1_machine_1/tier1 \\
                      results/tier_1_machine_2/tier1 \\
                      results/tier_1_machine_3/tier1 \\
                      results/tier_1_machine_4/tier1 \\
        --target_dir results/tier_1

Each source dir is expected to contain per-job subdirectories (one per
job_tag, written by `run_full_ablation.py` under `<output_root>/tier1/`).
For each subdirectory found, copies the entire job dir into target_dir.
Existing target subdirectories are NOT overwritten — the path is reported
as a duplicate so the user can manually reconcile.

After copy, counts how many jobs in target_dir have a `metrics.csv` for a
quick completeness signal (multi-machine launches occasionally produce
empty per-job dirs when training crashes early).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate per-machine Tier 1 results into one tree.",
    )
    ap.add_argument(
        "--source_dirs", nargs="+", required=True,
        help="One or more per-machine source directories, e.g. "
             "results/tier_1_machine_1/tier1 results/tier_1_machine_2/tier1",
    )
    ap.add_argument(
        "--target_dir", default="results/tier_1",
        help="Where to land the merged jobs (default: results/tier_1).",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be copied without performing the copy.",
    )
    args = ap.parse_args()

    target = Path(args.target_dir)
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_skipped_not_dir = 0
    duplicates: list[str] = []

    for src_dir in args.source_dirs:
        src = Path(src_dir)
        if not src.exists():
            print(f"WARN: {src} does not exist, skipping")
            continue
        if not src.is_dir():
            print(f"WARN: {src} is not a directory, skipping")
            continue
        for job_dir in sorted(src.iterdir()):
            if not job_dir.is_dir():
                total_skipped_not_dir += 1
                continue
            target_job = target / job_dir.name
            if target_job.exists():
                duplicates.append(str(job_dir))
                continue
            if args.dry_run:
                print(f"[DRY_RUN] would copy {job_dir} -> {target_job}")
            else:
                shutil.copytree(job_dir, target_job)
            total_copied += 1

    if args.dry_run:
        print(f"\n[DRY_RUN] {total_copied} job dirs would be copied to {target}")
    else:
        print(f"\nCopied {total_copied} job directories to {target}")
    if total_skipped_not_dir:
        print(f"  Skipped {total_skipped_not_dir} non-directory entries")
    if duplicates:
        print(
            f"WARN: {len(duplicates)} duplicates skipped "
            f"(job_dir name collision):"
        )
        for d in duplicates[:10]:
            print(f"  {d}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")

    # Completeness signal: count metrics.csv files under the target tree
    if not args.dry_run and target.exists():
        metrics_count = sum(1 for _ in target.glob("**/metrics.csv"))
        print(f"\nFinal: {metrics_count} jobs with metrics.csv in {target}")


if __name__ == "__main__":
    main()
