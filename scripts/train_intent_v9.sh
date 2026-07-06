#!/bin/bash
# Train IntentStylePredictor v9 ensemble (3 members) and place models
# at the paths expected by sumo_env.py:
#   results/intent_model_v9_member{0,1,2}.pt
#
# Run ASAP before pending tasks (cmu2, cmu4-8) start their Phase 1.
# General partition for faster scheduling.

#SBATCH --job-name=intent_v9
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/intent_v9_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/intent_v9_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -uo pipefail

export REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"

source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0

echo "=== INTENT V9 TRAINING START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Train ensemble — save to results/ so paths are easy to wire up.
# The ckpt_name is hardcoded as intent_model_member{idx}.pt in train_intent.py;
# we copy to the v9 name expected by sumo_env.py after training.
python experiments/train_intent.py \
    --out_dir "$REPO/results" \
    --scenarios 1a 1b 2_dense \
    --ego_maneuver stem_right \
    --n_episodes 200 \
    --n_epochs 100 \
    --patience 25 \
    --ensemble_size 3

echo "=== Training done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Place models at the paths sumo_env.py looks for
for i in 0 1 2; do
    src="$REPO/results/intent_model_member${i}.pt"
    dst="$REPO/results/intent_model_v9_member${i}.pt"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "  Placed: $dst ($(du -h "$dst" | cut -f1))"
    else
        echo "  ERROR: $src not found — training may have failed for member $i"
        exit 1
    fi
done

echo ""
echo "=== Intent ensemble ready ==="
echo "    Paths: $REPO/results/intent_model_v9_member{0,1,2}.pt"
echo "    Pending tasks (cmu2, cmu4-8) will now find the models when they start."
echo "=== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
