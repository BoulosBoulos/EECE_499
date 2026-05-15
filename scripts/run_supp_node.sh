#!/bin/bash
# Supplementary tier — best method on full 42-combo grid (126 jobs).
# Submit with: sbatch scripts/run_supp_node.sh
#
# 7 scenarios × 6 maneuvers × 3 seeds = 126 jobs, ~16 per machine.
# Best method determined post-Tier-1; currently hjb_aux as placeholder.
# Skip-if-done active: safe to resubmit.

#SBATCH --job-name=tier_supp
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/supp_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/supp_%A_%a.err
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
export MACHINE_NUM=$SLURM_ARRAY_TASK_ID

MACHINE_ID="cmu${SLURM_ARRAY_TASK_ID}"
MP=$(( $(nproc) - 2 ))

echo "=== [${MACHINE_ID}] SUPPLEMENTARY START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  MP=$MP"

SLICE=$(python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ['REPO'])
from experiments.pde.run_full_ablation import generate_jobs, _partition_balance_jobs
machine_num = int(os.environ.get('MACHINE_NUM', '1'))
n_machines  = 8
jobs = generate_jobs('supp', 400000)
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
echo "    Slice: jobs [${START}, ${END}) of ${TOTAL} total"

python experiments/pde/run_full_ablation.py \
    --tier supp \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --output_root "results/tier_supp_machine_${MACHINE_ID}" \
    --job_index_start "$START" \
    --job_index_end   "$END" \
    --machine_id      "${MACHINE_ID}_supp" \
    || echo "[${MACHINE_ID}] WARNING: orchestrator exited non-zero"

echo "=== [${MACHINE_ID}] SUPPLEMENTARY DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
