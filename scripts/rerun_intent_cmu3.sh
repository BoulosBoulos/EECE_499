#!/bin/bash
# Rerun intent=true jobs for cmu3 that failed before the model was ready.
# cmu3 Phase 1 intent: 15 failed (model missing at Phase 1 start).
# cmu3 Phase 2 intent: first ~15 may have failed (model became available 3 min
#   after Phase 2 started); remaining Phase 2 intent should have succeeded.
# Run AFTER train_intent_v9 finishes.

#SBATCH --job-name=rerun_intent_cmu3
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/rerun_intent_cmu3_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/rerun_intent_cmu3_%j.err
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
export MACHINE_NUM=3

MP=$(( $(nproc) - 2 ))

for i in 0 1 2; do
    if [ ! -f "$REPO/results/intent_model_v9_member${i}.pt" ]; then
        echo "ERROR: intent_model_v9_member${i}.pt missing. Run train_intent_v9 first."
        exit 1
    fi
done
echo "Intent models verified OK."

echo "=== [cmu3-intent-rerun] START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  MP=$MP"

# ── Phase 1 intent rerun (15 jobs) ───────────────────────────────────────────
echo "--- Phase 1 intent rerun ---"
python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --scenarios 1a 1b 2_dense \
    --maneuvers stem_right right_left \
    --seeds 42 123 456 789 999 \
    --methods soft_hjb_aux \
    --intents true \
    --machine_id "cmu3_intent_rerun" \
    || echo "WARNING: Phase 1 intent rerun non-zero exit"

# ── Phase 2 intent rerun — full cmu3 slice, intent=true only ─────────────────
# Covers the first ~15 that may have crashed before the model file was in place,
# plus any others missed. Safe to re-run all intent=true for this slice.
echo "--- Phase 2 intent rerun ---"

P2_BOUNDS=$(python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ['REPO'])
from types import SimpleNamespace
from experiments.pde.run_full_ablation import (
    generate_jobs, _apply_filters, _partition_balance_jobs
)
machine_num = int(os.environ.get('MACHINE_NUM', '3'))
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

echo "[cmu3-intent-rerun] Phase 2 slice: [$P2_START, $P2_END), intent=true only"

python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --methods hjb_aux soft_hjb_aux eikonal_aux fusion_aux rule_based \
    --intents true \
    --job_index_start "$P2_START" \
    --job_index_end   "$P2_END" \
    --machine_id "cmu3_p2_intent_rerun" \
    || echo "WARNING: Phase 2 intent rerun non-zero exit"

echo "=== [cmu3-intent-rerun] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
