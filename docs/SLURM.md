# Running ablation on SLURM (1 job per GPU)

This describes how to run the ablation on a SLURM cluster so that **each GPU runs exactly one ablation job** (one combination of scenario, variant, λ, seed, use_intent). No single training run is parallelized across multiple GPUs.

## Script

- **`scripts/run_ablation_slurm.sbatch`**

Defaults:

- **16 tasks** in the array, **8 running at a time** (`--array=1-16%8`).
- **1 GPU per task** (`--gres=gpu:1`).
- Each task runs `run_single_job.py --manifest <MANIFEST> --job_id <task_id-1> --gpu 0`.

So task 1 runs manifest job 0, task 2 runs job 1, … task 16 runs job 15. The ablation configuration (scenario, variant, lambda_phys, seed, use_intent) for each job comes from the manifest; **per-job ablation is defined by the manifest**, not by splitting one run across GPUs.

## Prerequisites

- One-time setup as in the main guide: venv, `PYTHONPATH`, `SUMO_HOME`.
- A job manifest (e.g. from `make jobs-manifest-batch1` or `generate_jobs.py`). For the default array 1–16, the manifest must have at least 16 jobs; tasks 1–16 run job indices 0–15.

## Basic usage

From the **project root**:

```bash
# Generate manifest (example: batch 1)
make jobs-manifest-batch1

# Tell the script where the manifest and outputs live
export MANIFEST=results/ablation_batch1/job_manifest.json
export OUT_DIR=results/ablation_batch1

# Submit (16 tasks, 8 at a time)
sbatch scripts/run_ablation_slurm.sbatch
```

After all tasks complete, aggregate and (if you use two batches) merge as usual:

```bash
python experiments/aggregate_results.py --out_dir results/ablation_batch1 --manifest results/ablation_batch1/job_manifest.json
# If you have batch2 as well:
make ablation-merge
make dashboard
```

## Optional environment overrides

Set these **before** `sbatch` if you want to change behavior for all tasks in this submission:

| Variable         | Default   | Meaning |
|------------------|-----------|--------|
| `MANIFEST`       | `results/ablation/job_manifest.json` | Path to job manifest. |
| `OUT_DIR`        | Directory of `MANIFEST`             | Where to write `jobs/eval_*.csv`, `jobs/train_*.csv`, checkpoints. |
| `TOTAL_STEPS`    | `50000`   | Training steps per job. |
| `EVAL_EPISODES`  | `50`      | Evaluation episodes per job. |
| `EVAL_STOCHASTIC`| `0`       | Set to `1` to run both deterministic and stochastic eval per job. |
| `JOB_OFFSET`     | `0`       | Add this to the job index (e.g. `16` so tasks 1–16 run manifest jobs 16–31). |
| `EXTRA_ARGS`     | (none)    | Extra args for `run_single_job.py` (e.g. `--reward_config configs/reward/custom.yaml`). |

Example:

```bash
export MANIFEST=results/ablation_batch1/job_manifest.json
export OUT_DIR=results/ablation_batch1
export TOTAL_STEPS=30000
export EVAL_STOCHASTIC=1
sbatch scripts/run_ablation_slurm.sbatch
```

## Changing how many jobs and how many in parallel

- **Only first 8 jobs, 8 at a time:**  
  `sbatch --array=1-8%8 scripts/run_ablation_slurm.sbatch`

- **16 jobs, 4 at a time:**  
  `sbatch --array=1-16%4 scripts/run_ablation_slurm.sbatch`

- **Jobs 17–32 from the same manifest (second chunk):**  
  Use an offset so task 1 runs job 16, task 2 runs job 17, … task 16 runs job 31:
  ```bash
  export JOB_OFFSET=16
  sbatch --array=1-16%8 scripts/run_ablation_slurm.sbatch
  ```
  So: first submission `JOB_OFFSET=0` runs jobs 0–15; second with `JOB_OFFSET=16` runs jobs 16–31; etc.

## Per-job hyperparameters / ablation

- **Scenario, variant, λ, seed, use_intent:** Fully determined by the **manifest**. Each `job_id` is one row in the manifest. Build the manifest with `experiments/generate_jobs.py` (and the Makefile targets like `jobs-manifest-batch1`, `jobs-manifest-batch2`; default includes both with and without LSTM) so that each row is the combo you want for that job index.
- **total_steps, eval_episodes, reward_config:** Same for all tasks in a submission; use the env vars above or `EXTRA_ARGS` to override.

## Logs

Stdout and stderr for each task go to:

- `slurm_<job_id>_<array_task_id>.out`
- `slurm_<job_id>_<array_task_id>.err`

in the directory from which you ran `sbatch`. Adjust `#SBATCH --output` and `#SBATCH --error` in the script if you want a different path (e.g. under `OUT_DIR`; the directory must exist before submit).

## Partition and time

Edit the script to set your partition and time if needed, for example:

```bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
```
