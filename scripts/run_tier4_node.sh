#!/bin/bash
# Tier 4: Held-out evaluation — 8-machine SLURM array.
# Eval-only: re-evaluates trained checkpoints under held-out conditions.
#   HO1: occ→noocc     (tier1 checkpoints, no_buildings=True)
#   HO2: noocc→occ     (tier2 occOFF checkpoints, no_buildings=False)
#   HO3: full→adv      (tier1 checkpoints, style_filter=adversarial)
#   HO4: nominal→adv   (tier3_behav checkpoints, style_filter=adversarial)
#   HO5: vis→novis     (tier1 checkpoints, state_ablation=no_visibility)
#
# Each machine discovers its own checkpoints from its per-machine result dirs.
# Skip-if-done active: safe to resubmit.

#SBATCH --job-name=tier4
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier4_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier4_%A_%a.err
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

MACHINE_ID="cmu${SLURM_ARRAY_TASK_ID}"
MP=$(( $(nproc) - 2 ))

echo "=== [${MACHINE_ID}] TIER 4 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  MP=$MP  nproc=$(nproc)"
echo "    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Source dirs: each machine evaluates the checkpoints it trained
T1_SRC="$REPO/results/tier_1_machine_${MACHINE_ID}_p2/tier1"
T2_NOOCC_SRC="$REPO/results/tier_2_machine_${MACHINE_ID}/tier2/2b_occlusion_sweep"
T3_BEHAV_SRC="$REPO/results/tier_3_machine_${MACHINE_ID}/tier3_behav"

python experiments/pde/run_full_ablation.py \
    --tier 4 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --output_root "results/tier_4_machine_${MACHINE_ID}" \
    --machine_id  "${MACHINE_ID}_t4" \
    --tier4_source_dir tier1        "$T1_SRC" \
    --tier4_source_dir tier2_noocc  "$T2_NOOCC_SRC" \
    --tier4_source_dir tier3_behav  "$T3_BEHAV_SRC" \
    || echo "[${MACHINE_ID}] WARNING: orchestrator exited non-zero"

echo "=== [${MACHINE_ID}] TIER 4 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    Results: $REPO/results/tier_4_machine_${MACHINE_ID}/"
