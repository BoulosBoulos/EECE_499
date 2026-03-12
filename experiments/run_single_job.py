"""Run one or more ablation jobs by index (for parallel execution across GPUs).

Reads job manifest, runs training + evaluation for the given job ID(s), writes
per-job result files under results/ablation/jobs/ so multiple processes do not
overwrite. Use with launch_parallel_16gpu.sh to distribute across 16 GPUs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/run_single_job.py --manifest results/ablation/job_manifest.json --job_id 0
  python experiments/run_single_job.py --manifest results/ablation/job_manifest.json --worker_index 0 --num_workers 16 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Set device before importing torch
def _set_gpu(gpu_id: int | None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

from experiments.run_ablation import (
    EVAL_FIELDS,
    TRAIN_LOG_FIELDS,
    _append_csv,
    _load_config,
    collect_rollouts,
    eval_one,
    train_one,
)


def load_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path) as f:
        data = json.load(f)
    return data["jobs"]


def get_job_indices(worker_index: int | None, num_workers: int | None, job_id: int | None, n_jobs: int) -> list[int]:
    if job_id is not None:
        return [job_id]
    if worker_index is not None and num_workers is not None:
        return [i for i in range(worker_index, n_jobs, num_workers)]
    return list(range(n_jobs))


def main():
    parser = argparse.ArgumentParser(description="Run ablation job(s) by index (for multi-GPU)")
    parser.add_argument("--manifest", required=True, help="Path to job_manifest.json")
    parser.add_argument("--job_id", type=int, default=None, help="Single job index to run")
    parser.add_argument("--worker_index", type=int, default=None,
                        help="Worker rank (0..num_workers-1); runs jobs worker_index, worker_index+N, ...")
    parser.add_argument("--num_workers", type=int, default=None, help="Total number of workers (e.g. 16)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index for this process (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--out_dir", default="results/ablation")
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_stochastic", action="store_true")
    parser.add_argument("--reward_config", default=None)
    args = parser.parse_args()

    _set_gpu(args.gpu)
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    jobs = load_manifest(args.manifest)
    n_jobs = len(jobs)
    indices = get_job_indices(args.worker_index, args.num_workers, args.job_id, n_jobs)
    if not indices:
        print("No jobs to run for this worker.")
        return 0

    jobs_dir = os.path.join(args.out_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    from env.sumo_env import SumoEnv

    failed_count = 0
    for job_id in indices:
        if job_id >= n_jobs:
            continue
        job = jobs[job_id]
        scenario = job["scenario"]
        vname = job["variant"]
        placement = job["pinn_placement"]
        use_l_ego = job["use_l_ego"]
        use_sf = job["use_safety_filter"]
        use_ttc = job["use_ttc"]
        use_stop = job["use_stop"]
        use_fric = job["use_fric"]
        lp = job["lambda_phys"]
        seed = job["seed"]
        use_intent = job.get("use_intent", False)
        run_tag = f"{scenario}_{vname}_lp{lp}_s{seed}"

        eval_csv = os.path.join(jobs_dir, f"eval_{job_id:06d}.csv")
        train_csv = os.path.join(jobs_dir, f"train_{job_id:06d}.csv")

        try:
            env = SumoEnv(scenario_name=scenario, use_intent=use_intent, reward_config=args.reward_config)
        except Exception as e:
            print(f"Job {job_id} ({run_tag}): env init failed: {e}", file=sys.stderr)
            failed_count += 1
            continue

        try:
            print(f"Job {job_id}: Training {run_tag}...")
            ckpt = train_one(
                env, scenario, vname,
                pinn_placement=placement, use_l_ego=use_l_ego,
                use_safety_filter=use_sf,
                use_ttc=use_ttc, use_stop=use_stop, use_fric=use_fric,
                total_steps=args.total_steps, out_dir=args.out_dir,
                device=device, seed=seed, lambda_phys=lp,
                train_csv=train_csv,
            )
        except Exception as e:
            print(f"Job {job_id} ({run_tag}): training failed: {e}", file=sys.stderr)
            env.close()
            failed_count += 1
            continue

        if not os.path.isfile(ckpt):
            print(f"Job {job_id} ({run_tag}): checkpoint not found after train.", file=sys.stderr)
            env.close()
            failed_count += 1
            continue

        eval_modes = ["deterministic"]
        if args.eval_stochastic:
            eval_modes.append("stochastic")
        for mode in eval_modes:
            det = mode == "deterministic"
            print(f"Job {job_id}: Eval {run_tag} [{mode}]...")
            try:
                m = eval_one(env, ckpt, args.eval_episodes, device, seed,
                             deterministic=det,
                             pinn_placement=placement,
                             use_safety_filter=use_sf)
            except Exception as e:
                print(f"Job {job_id} ({run_tag}) eval [{mode}] failed: {e}", file=sys.stderr)
                failed_count += 1
                continue
            row = {
                "scenario": scenario, "variant": vname,
                "pinn_placement": placement, "use_l_ego": use_l_ego,
                "use_safety_filter": use_sf,
                "lambda_phys": lp, "seed": seed,
                "eval_mode": mode, **m,
            }
            _append_csv(eval_csv, row, EVAL_FIELDS)

        env.close()
        print(f"Job {job_id} done: {run_tag}")

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
