#!/bin/bash
# SLURM array job — 8 Tier-1 machines (cmu1–cmu8)
# Submit with: sbatch scripts/run_tier1_node.sh
# Each array task maps to one machine assignment from the handoff §6.1.
#
# Phase 1:  initial method-specific 30 jobs (3 cells × method × 5 seeds × 2 intents)
# Phase 2:  remaining 1/8 slice of full Tier-1 (5 non-local methods, all 12 cells, all 10 seeds)
#
# Outputs:
#   Phase 1 → results/tier_1_machine_cmuN/tier1/
#   Phase 2 → results/tier_1_machine_cmuN_p2/tier1/

#SBATCH --job-name=tier1
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier1_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier1_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=1-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

set -uo pipefail

export REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"

# ── Activate venv ────────────────────────────────────────────────────────────
source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")

# ── Machine assignment (§6.1) ────────────────────────────────────────────────
# cmu1–cmu5: initial seeds 42..999 | cmu6–cmu8: helper seeds 1111..5555
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

# ── Compute max_parallel ─────────────────────────────────────────────────────
# rule_based is eval-only (no GPU/Python training); can use all cores
if [ "$METHOD" = "rule_based" ]; then
    MP=$(nproc)
else
    MP=$(( $(nproc) - 2 ))
fi

echo "=== [$MACHINE_ID] START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  method=$METHOD  seeds='$SEEDS'  MP=$MP  nproc=$(nproc)"
echo "    SUMO_HOME=$SUMO_HOME"

# ── Distributed manifest (§13) ───────────────────────────────────────────────
{
  flock 200
  echo "$(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$ method=$METHOD machine_id=$MACHINE_ID max_parallel=$MP nproc=$(nproc)"
} 200>>"$REPO/tier1_distributed_manifest.txt" || \
  echo "$(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$ method=$METHOD machine_id=$MACHINE_ID max_parallel=$MP nproc=$(nproc)" \
    >> "$REPO/tier1_distributed_manifest.txt"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — initial 30 jobs (3 cells × method × seeds × 2 intents)
# Cells: (1a,stem_right), (1b,stem_right), (2_dense,right_left)
# ═══════════════════════════════════════════════════════════════════════════════
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
    || echo "[$MACHINE_ID] WARNING: orchestrator exited non-zero (some jobs may have failed)"

echo "--- [$MACHINE_ID] PHASE 1 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — remaining 1/8 slice of full Tier-1
# Methods: hjb_aux, soft_hjb_aux, eikonal_aux, fusion_aux, rule_based
# (excludes cbf_aux + drppo which local machine owns)
# All 12 cells, all 10 seeds, both intents → 1200 jobs total / 8 = 150 per machine
# ═══════════════════════════════════════════════════════════════════════════════
echo "--- [$MACHINE_ID] PHASE 2 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"

# Compute exact slice boundaries by running the same manifest logic
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
# Override methods to exclude local machine's cbf_aux + drppo
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

echo "[$MACHINE_ID] Phase 2 slice: jobs [$P2_START, $P2_END) of $P2_TOTAL"

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
echo "[$MACHINE_ID] Results in: $REPO/results/tier_1_machine_${MACHINE_ID}/"
echo "[$MACHINE_ID] Check next: Tier-2 (--tier 2 --subgrid 2a) per §6.3"
