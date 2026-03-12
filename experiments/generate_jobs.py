"""Generate a job manifest for parallel ablation across multiple GPUs.

Each job is one (scenario, variant, lambda_phys, seed): train then eval.
Output: manifest JSON and optional CSV listing every job with index.

Usage:
  python experiments/generate_jobs.py --out_dir results/ablation --seeds 42 123 456 --lambda_phys 0.1 0.5 1.0
  python experiments/generate_jobs.py --out_dir results/ablation --variants nopinn pinn_critic pinn_ego

Then run workers with: run_single_job.py --manifest results/ablation/job_manifest.json --job_id <i> --gpu <gpu_id>
Or use scripts/launch_parallel_16gpu.sh to distribute jobs across 16 GPUs.
"""

from __future__ import annotations

import argparse
import json
import os

# Reuse same constants as run_ablation
from experiments.run_ablation import (
    SCENARIOS,
    VARIANTS,
    DEFAULT_SEEDS,
)


def build_jobs(
    scenarios: list[str],
    variants: list[str] | None,
    seeds: list[int],
    lambda_phys: list[float],
    use_intent: bool = False,
) -> list[dict]:
    """Build list of job specs. Each job: scenario, variant_name, pinn_placement, use_l_ego, use_safety_filter, use_ttc, use_stop, use_fric, lambda_phys, seed."""
    active = [v for v in VARIANTS if variants is None or v[0] in variants]
    jobs = []
    for scenario in scenarios:
        for (vname, placement, use_l_ego, use_sf, use_ttc, use_stop, use_fric) in active:
            lp_values = lambda_phys if placement != "none" else [0.0]
            for lp in lp_values:
                for seed in seeds:
                    jobs.append({
                        "scenario": scenario,
                        "variant": vname,
                        "pinn_placement": placement,
                        "use_l_ego": use_l_ego,
                        "use_safety_filter": use_sf,
                        "use_ttc": use_ttc,
                        "use_stop": use_stop,
                        "use_fric": use_fric,
                        "lambda_phys": lp,
                        "seed": seed,
                        "use_intent": use_intent,
                    })
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Generate job manifest for parallel ablation")
    parser.add_argument("--out_dir", default="results/ablation",
                        help="Directory where manifest will be written")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variant names (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--lambda_phys", type=float, nargs="+", default=[0.5],
                        help="lambda_physics_critic values (only for PINN variants)")
    parser.add_argument(
        "--use_intent",
        action="store_true",
        help="Enable intent LSTM for all jobs (state dim +30 where applicable)",
    )
    parser.add_argument(
        "--intent_ablation",
        action="store_true",
        help="Generate jobs for both use_intent=False and use_intent=True",
    )
    parser.add_argument("--no_csv", action="store_true", help="Do not write manifest as CSV")
    args = parser.parse_args()

    if args.intent_ablation:
        # Build two copies of the grid: one without intent features, one with.
        jobs = []
        for flag in (False, True):
            jobs.extend(
                build_jobs(
                    scenarios=args.scenarios,
                    variants=args.variants,
                    seeds=args.seeds,
                    lambda_phys=args.lambda_phys,
                    use_intent=flag,
                )
            )
    else:
        jobs = build_jobs(
            scenarios=args.scenarios,
            variants=args.variants,
            seeds=args.seeds,
            lambda_phys=args.lambda_phys,
            use_intent=args.use_intent,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "job_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"jobs": jobs, "n_jobs": len(jobs)}, f, indent=2)

    print(f"Wrote {manifest_path} with {len(jobs)} jobs.")

    if not args.no_csv:
        import csv
        csv_path = os.path.join(args.out_dir, "job_manifest.csv")
        if jobs:
            fieldnames = ["job_id"] + list(jobs[0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for i, j in enumerate(jobs):
                    w.writerow({"job_id": i, **j})
            print(f"Wrote {csv_path} with job_id 0..{len(jobs)-1}.")

    return manifest_path


if __name__ == "__main__":
    main()
