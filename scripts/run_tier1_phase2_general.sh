#!/bin/bash
# Tier 1 Phase 2 ONLY — runs after Phase 1 is complete.
# Submit with: sbatch scripts/run_tier1_phase2_general.sh
#
# Phase 2: the full 1200-job Tier-1 manifest (5 methods, all 12 combos,
# 10 seeds, 2 intents) distributed across 8 machines (150 jobs each).
# drppo/cbf_aux are on the local PC — cluster never runs those.
#
# Skip-if-done: if step400k checkpoint exists, job is silently skipped.
# This makes it SAFE to resubmit if wall-time kills the job mid-run.
# 48h wall time fits ~2 batches (30 jobs × ~25h each). Resubmit as needed.

#SBATCH --job-name=t1_phase2
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/t1p2_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/t1p2_%A_%a.err
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

echo "=== [${MACHINE_ID}] PHASE 2 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  MP=$MP  nproc=$(nproc)"
echo "    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Verify intent models
for i in 0 1 2; do
    if [ ! -f "$REPO/results/intent_model_v9_member${i}.pt" ]; then
        echo "ERROR: intent_model_v9_member${i}.pt missing."
        exit 1
    fi
done

# Compute this machine's slice of the 1200-job Phase 2 manifest
P2_BOUNDS=$(python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ['REPO'])
from types import SimpleNamespace
from experiments.pde.run_full_ablation import (
    generate_jobs, _apply_filters, _partition_balance_jobs
)
machine_num = int(os.environ.get('MACHINE_NUM', '1'))
n_machines  = 8
args = SimpleNamespace(
    seeds=None, methods=None, scenarios=None, maneuvers=None,
    intents=None, include_rule_based=True,
)
args.methods = ['hjb_aux', 'soft_hjb_aux', 'eikonal_aux', 'fusion_aux', 'rule_based']
jobs = generate_jobs('1', 400000)
jobs = _apply_filters(jobs, args)
jobs = _partition_balance_jobs(jobs)
total    = len(jobs)
per_node = total // n_machines
start    = (machine_num - 1) * per_node
end      = start + per_node if machine_num < n_machines else total
print(f"{start} {end} {total}")
PYEOF
)

P2_START=$(echo "$P2_BOUNDS" | awk '{print $1}')
P2_END=$(echo "$P2_BOUNDS"   | awk '{print $2}')
P2_TOTAL=$(echo "$P2_BOUNDS" | awk '{print $3}')
echo "    Slice: jobs [${P2_START}, ${P2_END}) of ${P2_TOTAL} total"
echo "    (skip-if-done active — safe to resubmit after wall-time kill)"

python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --methods hjb_aux soft_hjb_aux eikonal_aux fusion_aux rule_based \
    --job_index_start "$P2_START" \
    --job_index_end   "$P2_END" \
    --machine_id      "${MACHINE_ID}_p2" \
    || echo "[${MACHINE_ID}] WARNING: orchestrator exited non-zero"

echo "=== [${MACHINE_ID}] PHASE 2 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    Results: $REPO/results/tier_1_machine_${MACHINE_ID}_p2/tier1/"
