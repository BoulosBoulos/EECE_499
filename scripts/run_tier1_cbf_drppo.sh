#!/bin/bash
# Tier 1 cbf_aux + drppo — cluster catch-up run.
# These two methods were omitted from the original phase2 cluster run and are
# needed before Tier 4 can evaluate HO1/HO3/HO5.
#
# Outputs land in results/tier_1_machine_${MACHINE_ID}_p2/tier1/  (same tree
# as the existing p2 results, so run_tier4_node.sh finds them automatically).
# Skip-if-done is active: safe to resubmit after wall-time kill.

#SBATCH --job-name=t1_cbfdrppo
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/t1_cbfdrppo_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/t1_cbfdrppo_%A_%a.err
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

echo "=== [${MACHINE_ID}] T1 cbf+drppo START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  MP=$MP  nproc=$(nproc)"
echo "    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Verify intent models
for i in 0 1 2; do
    if [ ! -f "$REPO/results/intent_model_v9_member${i}.pt" ]; then
        echo "ERROR: intent_model_v9_member${i}.pt missing."
        exit 1
    fi
done

# Compute this machine's slice of the 480-job cbf_aux+drppo manifest
CBF_BOUNDS=$(python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ['REPO'])
from types import SimpleNamespace
from experiments.pde.run_full_ablation import (
    generate_jobs, _apply_filters, _partition_balance_jobs
)
machine_num = int(os.environ.get('MACHINE_NUM', '1'))
n_machines  = 8
args = SimpleNamespace(
    seeds=None, methods=['cbf_aux', 'drppo'], scenarios=None, maneuvers=None,
    intents=None, include_rule_based=False,
)
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

CBF_START=$(echo "$CBF_BOUNDS" | awk '{print $1}')
CBF_END=$(echo "$CBF_BOUNDS"   | awk '{print $2}')
CBF_TOTAL=$(echo "$CBF_BOUNDS" | awk '{print $3}')
echo "    Slice: jobs [${CBF_START}, ${CBF_END}) of ${CBF_TOTAL} total"
echo "    (skip-if-done active — safe to resubmit after wall-time kill)"

python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --methods cbf_aux drppo \
    --job_index_start "$CBF_START" \
    --job_index_end   "$CBF_END" \
    --machine_id      "${MACHINE_ID}_p2" \
    || echo "[${MACHINE_ID}] WARNING: orchestrator exited non-zero"

echo "=== [${MACHINE_ID}] T1 cbf+drppo DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    Results: $REPO/results/tier_1_machine_${MACHINE_ID}_p2/tier1/"
