#!/bin/bash
# Tier-1 array job on the general partition (no preemption).
# Identical logic to run_tier1_node.sh but uses general/normal instead of
# preempt/preempt_qos.  Wall-time 36h covers Phase 1 (~85m) + Phase 2 (~7h)
# with a generous buffer.  Intent model must already be at
#   results/intent_model_v9_member{0,1,2}.pt  (placed by train_intent_v9).

#SBATCH --job-name=tier1
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier1_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier1_%A_%a.err
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

# ── Machine assignment ────────────────────────────────────────────────────────
case "$SLURM_ARRAY_TASK_ID" in
  1) METHOD=fusion_aux;    MACHINE_ID=cmu1; SEEDS="42 123 456 789 999" ;;
  2) METHOD=hjb_aux;       MACHINE_ID=cmu2; SEEDS="42 123 456 789 999" ;;
  3) METHOD=soft_hjb_aux;  MACHINE_ID=cmu3; SEEDS="42 123 456 789 999" ;;
  4) METHOD=eikonal_aux;   MACHINE_ID=cmu4; SEEDS="42 123 456 789 999" ;;
  5) METHOD=rule_based;    MACHINE_ID=cmu5; SEEDS="42 123 456 789 999" ;;
  6) METHOD=fusion_aux;    MACHINE_ID=cmu6; SEEDS="1111 2222 3333 4444 5555" ;;
  7) METHOD=hjb_aux;       MACHINE_ID=cmu7; SEEDS="1111 2222 3333 4444 5555" ;;
  8) METHOD=soft_hjb_aux;  MACHINE_ID=cmu8; SEEDS="1111 2222 3333 4444 5555" ;;
  *) echo "Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

export MACHINE_NUM=$SLURM_ARRAY_TASK_ID

if [ "$METHOD" = "rule_based" ]; then
    MP=$(nproc)
else
    MP=$(( $(nproc) - 2 ))
fi

echo "=== [$MACHINE_ID] START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  method=$METHOD  seeds='$SEEDS'  MP=$MP  nproc=$(nproc)"
echo "    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "    SUMO_HOME=$SUMO_HOME"

# Verify intent model is present
for i in 0 1 2; do
    if [ ! -f "$REPO/results/intent_model_v9_member${i}.pt" ]; then
        echo "ERROR: intent_model_v9_member${i}.pt missing. Run train_intent_v9 first."
        exit 1
    fi
done
echo "    Intent models: OK"

{
  flock 200
  echo "$(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$ method=$METHOD machine_id=$MACHINE_ID max_parallel=$MP nproc=$(nproc) [GENERAL]"
} 200>>"$REPO/tier1_distributed_manifest.txt" || \
  echo "$(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$ method=$METHOD machine_id=$MACHINE_ID max_parallel=$MP [GENERAL]" \
    >> "$REPO/tier1_distributed_manifest.txt"

# ── PHASE 1 ───────────────────────────────────────────────────────────────────
echo "--- [$MACHINE_ID] PHASE 1 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --scenarios 1a 1b 2_dense \
    --maneuvers stem_right right_left \
    --seeds $SEEDS \
    --methods "$METHOD" \
    --machine_id "$MACHINE_ID" \
    || echo "[$MACHINE_ID] WARNING: Phase 1 orchestrator exited non-zero"
echo "--- [$MACHINE_ID] PHASE 1 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"

# ── PHASE 2 ───────────────────────────────────────────────────────────────────
echo "--- [$MACHINE_ID] PHASE 2 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"

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
echo "[$MACHINE_ID] Phase 2 slice: [$P2_START, $P2_END) of $P2_TOTAL"

python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --methods hjb_aux soft_hjb_aux eikonal_aux fusion_aux rule_based \
    --job_index_start "$P2_START" \
    --job_index_end   "$P2_END" \
    --machine_id      "${MACHINE_ID}_p2" \
    || echo "[$MACHINE_ID] WARNING: Phase 2 orchestrator exited non-zero"

echo "=== [$MACHINE_ID] ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    Results: $REPO/results/tier_1_machine_${MACHINE_ID}/"
