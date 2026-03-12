#!/bin/bash
#SBATCH --job-name=drppo_ablation
#SBATCH --output=slurm_%%A_%%a.out
#SBATCH --error=slurm_%%A_%%a.err
#SBATCH --array=1-16%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#
# 1 job = 1 GPU. No parallelization of a single run across GPUs.
# This submits 16 tasks; 8 run at a time. When 8 finish, the next 8 start.
# Each task runs ONE ablation combo (one row from the manifest) via --job_id.
# Per-job scenario/variant/lambda/seed/use_intent come from the manifest.
#
# Optional overrides (set before sbatch or in your env):
#   MANIFEST=path/to/job_manifest.json
#   OUT_DIR=results/ablation_slurm
#   JOB_OFFSET=0       (first task runs job_id 0; set 16 to run jobs 16..31, etc.)
#   TOTAL_STEPS=50000
#   EVAL_EPISODES=50
#   EVAL_STOCHASTIC=1   (set to 1 for deterministic + stochastic eval)
#   EXTRA_ARGS="--reward_config configs/reward/custom.yaml"
#
# Usage:
#   cd /path/to/project/root
#   make jobs-manifest-batch1   # or batch2, or intent-ablation
#   export MANIFEST=results/ablation_batch1/job_manifest.json
#   export OUT_DIR=results/ablation_batch1
#   sbatch scripts/run_ablation_slurm.sbatch
#
# To run only tasks 1-8 (first 8 jobs): sbatch --array=1-8%8 ...
# To run tasks 9-16:                    sbatch --array=9-16%8 ...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MANIFEST="${MANIFEST:-results/ablation/job_manifest.json}"
OUT_DIR="${OUT_DIR:-$(dirname "$MANIFEST")}"
OUT_DIR="$(cd "$PROJECT_ROOT" && realpath -m "$OUT_DIR")"
mkdir -p "$OUT_DIR/jobs"

# One task = one job index (0-based job_id). Use JOB_OFFSET to run next chunk (e.g. 16, 32).
JOB_OFFSET="${JOB_OFFSET:-0}"
JOB_ID=$((${JOB_OFFSET} + ${SLURM_ARRAY_TASK_ID:-1} - 1))

# Optional hyperparameter overrides (apply to all tasks in this submission)
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_STOCHASTIC="${EVAL_STOCHASTIC:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "$EVAL_STOCHASTIC" == "1" || "$EVAL_STOCHASTIC" == "true" || "$EVAL_STOCHASTIC" == "yes" ]]; then
  EVAL_FLAG="--eval_stochastic"
else
  EVAL_FLAG=""
fi

# Environment (adjust if you use conda or a different venv)
if [[ -d "$PROJECT_ROOT/.venv" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
export PYTHONPATH="${PROJECT_ROOT:?}"
export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"

# SLURM gives this task one GPU; use it as device 0
echo "Task $SLURM_ARRAY_TASK_ID (job_id=$JOB_ID) starting on $(hostname)"
python experiments/run_single_job.py \
  --manifest "$MANIFEST" \
  --job_id "$JOB_ID" \
  --gpu 0 \
  --out_dir "$OUT_DIR" \
  --total_steps "$TOTAL_STEPS" \
  --eval_episodes "$EVAL_EPISODES" \
  $EVAL_FLAG \
  $EXTRA_ARGS

echo "Task $SLURM_ARRAY_TASK_ID (job_id=$JOB_ID) finished with exit $?"
