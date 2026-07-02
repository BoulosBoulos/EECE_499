#!/bin/bash
# Tier 4 rebalance: redistribute all remaining T4 eval jobs evenly across
# 8 GPUs regardless of which source machine they came from.
#
# Use this instead of R2 when some machines finished early and their GPUs
# are idle while slow machines (cmu2/6/7/8) still have hundreds of jobs left.
#
# Each array task pools remaining jobs from ALL 8 source machines, takes an
# even slice, and runs them in parallel (--max_parallel).  Skip-if-done is
# checked at both scan time and run time, so concurrent workers won't
# double-run a job.
#
# Submit:
#   sbatch scripts/run_tier4_rebalance.sh
# (Cancel pending R2 and in-progress slow R1 tasks first to avoid races.)

#SBATCH --job-name=t4_rebal
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/t4_rebal_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/t4_rebal_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=1-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -uo pipefail

export REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"

source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0

MP=$(( $(nproc) - 2 ))

echo "=== [w${SLURM_ARRAY_TASK_ID}/8] T4 REBALANCE START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  MP=$MP  nproc=$(nproc)"
echo "    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

python scripts/tier4_pool_runner.py \
    --worker_id    "$SLURM_ARRAY_TASK_ID" \
    --n_workers    8 \
    --total_steps  400000 \
    --max_parallel "$MP" \
    --repo         "$REPO" \
    || echo "[w${SLURM_ARRAY_TASK_ID}] WARNING: runner exited non-zero"

echo "=== [w${SLURM_ARRAY_TASK_ID}/8] T4 REBALANCE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
