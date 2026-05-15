#!/bin/bash
# Rerun only the intent=true jobs that failed because intent_model_v9 was missing.
# Submit AFTER train_intent_v9 finishes (model must be at results/intent_model_v9_member*.pt).
#
# Covers:
#   cmu1 Phase 1:  fusion_aux, seeds 42..999, 3 cells, intent=true  (15 jobs)
#   cmu1 Phase 2:  cmu1's 1/8 slice of full manifest, intent=true only (~75 jobs)
#   (cmu2/4-8 will run intent jobs cleanly once they start — no resubmit needed)
#
# Usage: sbatch scripts/rerun_intent_jobs.sh

#SBATCH --job-name=rerun_intent_cmu1
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/rerun_intent_cmu1_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/rerun_intent_cmu1_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

set -uo pipefail

export REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"

source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0
export MACHINE_NUM=1

MP=$(( $(nproc) - 2 ))

# Verify intent model is present before starting
for i in 0 1 2; do
    if [ ! -f "$REPO/results/intent_model_v9_member${i}.pt" ]; then
        echo "ERROR: intent_model_v9_member${i}.pt not found. Run train_intent_v9 first."
        exit 1
    fi
done
echo "Intent models verified OK."

echo "=== [cmu1-intent-rerun] START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  MP=$MP  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# ── Phase 1 intent rerun ──────────────────────────────────────────────────────
echo "--- Phase 1 intent rerun START $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --scenarios 1a 1b 2_dense \
    --maneuvers stem_right right_left \
    --seeds 42 123 456 789 999 \
    --methods fusion_aux \
    --intents true \
    --machine_id "cmu1_intent_rerun" \
    || echo "WARNING: Phase 1 intent rerun exited non-zero"
echo "--- Phase 1 intent rerun DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"

# ── Phase 2 intent rerun (cmu1's 1/8 slice, intent=true only) ────────────────
echo "--- Phase 2 intent rerun START $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"

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

echo "[cmu1-intent-rerun] Phase 2 slice: jobs [$P2_START, $P2_END) of $P2_TOTAL, intent=true only"

python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --methods hjb_aux soft_hjb_aux eikonal_aux fusion_aux rule_based \
    --intents true \
    --job_index_start "$P2_START" \
    --job_index_end   "$P2_END" \
    --machine_id      "cmu1_p2_intent_rerun" \
    || echo "WARNING: Phase 2 intent rerun exited non-zero"

echo "=== [cmu1-intent-rerun] ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
