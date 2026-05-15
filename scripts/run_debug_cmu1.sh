#!/bin/bash
# Debug-partition warm-start for cmu1 (fusion_aux, seeds 42..999).
# Starts immediately (debug partition), 12h wall time, 12-CPU cap.
# Runs as many Phase-1 jobs as finish in 12h (expect ~12–15 of 30).
# The preempt array (job 7889693) will complete the rest once it starts.

#SBATCH --job-name=tier1_cmu1_debug
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/cmu1_debug_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/cmu1_debug_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -uo pipefail

export REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"

source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0

MP=$(( $(nproc) - 2 ))
METHOD=fusion_aux
MACHINE_ID=cmu1_debug
SEEDS="42 123 456 789 999"

echo "=== [$MACHINE_ID] DEBUG START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  nproc=$(nproc)  MP=$MP  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

echo "$(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$ method=$METHOD machine_id=$MACHINE_ID max_parallel=$MP nproc=$(nproc) [DEBUG-WARMSTART]" \
    >> "$REPO/tier1_distributed_manifest.txt"

python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 400000 \
    --max_parallel "$MP" \
    --scenarios 1a 1b 2_dense \
    --maneuvers stem_right right_left \
    --seeds $SEEDS \
    --methods "$METHOD" \
    --machine_id "$MACHINE_ID" \
    || echo "[$MACHINE_ID] orchestrator exited non-zero (time limit or partial run)"

echo "=== [$MACHINE_ID] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Completed jobs in: $REPO/results/tier_1_machine_${MACHINE_ID}/"
