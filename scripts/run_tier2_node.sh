#!/bin/bash
# SLURM array job — 8 machines (cmu1–cmu8), Tier 2 ablation.
# Submit with: sbatch scripts/run_tier2_node.sh
#
# Tier 2: 1,160 jobs across 3 sub-grids, 145 per machine.
#   2a — lambda residual sweep (600 jobs): 5 methods × 2 scenarios × 2 maneuvers × 5 seeds × 6 λ
#   2b — occlusion sweep       (400 jobs): 5 methods × 4 scenarios × 2 maneuvers × 5 seeds × 2 occ
#   2c — fusion weight sweep   (160 jobs): fusion_aux × 2 scenarios × 2 maneuvers × 5 seeds × 8 w
#
# All jobs use 400k steps (calibrated). No intent dimension.
# Outputs: results/tier_2_machine_cmuN/tier2/{2a_lambda_sweep,2b_occlusion_sweep,2c_fusion_weights}/

#SBATCH --job-name=tier2
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier2_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier2_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=1-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00

set -uo pipefail

export REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"

source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0
export MACHINE_NUM=$SLURM_ARRAY_TASK_ID

MACHINE_ID="cmu${SLURM_ARRAY_TASK_ID}"
MP=$(( $(nproc) - 2 ))

echo "=== [${MACHINE_ID}] TIER 2 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  MP=$MP  nproc=$(nproc)"
echo "    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "    SUMO_HOME=$SUMO_HOME"

# ── Distributed manifest ─────────────────────────────────────────────────────
{
  flock 200
  echo "$(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$ machine_id=${MACHINE_ID}_t2 max_parallel=$MP nproc=$(nproc) [TIER2]"
} 200>>"$REPO/tier2_distributed_manifest.txt" || \
  echo "$(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$ machine_id=${MACHINE_ID}_t2 max_parallel=$MP [TIER2]" \
    >> "$REPO/tier2_distributed_manifest.txt"

# ── Compute slice boundaries ──────────────────────────────────────────────────
# 1160 jobs / 8 machines = exactly 145 each. Computed dynamically so indices
# stay correct if the manifest ever changes between runs.
SLICE=$(python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ['REPO'])
from experiments.pde.run_full_ablation import _generate_tier2_jobs, _partition_balance_jobs
machine_num = int(os.environ.get('MACHINE_NUM', '1'))
n_machines  = 8
jobs = _generate_tier2_jobs(400000, 'results')
jobs = _partition_balance_jobs(jobs)
total    = len(jobs)
per_node = total // n_machines
start    = (machine_num - 1) * per_node
end      = start + per_node if machine_num < n_machines else total
print(f"{start} {end} {total}")
PYEOF
)

START=$(echo "$SLICE" | awk '{print $1}')
END=$(echo "$SLICE"   | awk '{print $2}')
TOTAL=$(echo "$SLICE" | awk '{print $3}')

echo "    Slice: jobs [$START, $END) of $TOTAL total"

# ── Run ───────────────────────────────────────────────────────────────────────
python experiments/pde/run_full_ablation.py \
    --tier 2 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --output_root "results/tier_2_machine_${MACHINE_ID}" \
    --job_index_start "$START" \
    --job_index_end   "$END" \
    --machine_id      "${MACHINE_ID}_t2" \
    || echo "[${MACHINE_ID}] WARNING: orchestrator exited non-zero (some jobs may have failed)"

echo "=== [${MACHINE_ID}] TIER 2 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    Results: $REPO/results/tier_2_machine_${MACHINE_ID}/tier2/"
