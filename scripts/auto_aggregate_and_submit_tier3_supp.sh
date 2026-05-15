#!/bin/bash
# Auto-aggregate Tier 2 and submit Tier 3 + Supplementary.
# Submit with: sbatch --dependency=afterok:<tier2_job_id> scripts/auto_aggregate_and_submit_tier3_supp.sh

#SBATCH --job-name=agg_t3_submit
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/agg_t3_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/agg_t3_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00

set -euo pipefail

REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"

echo "=== Tier 2 Aggregation START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

python experiments/pde/aggregate_tier_1_results.py \
    --source_dirs \
        results/tier_2_machine_cmu1/tier2 \
        results/tier_2_machine_cmu2/tier2 \
        results/tier_2_machine_cmu3/tier2 \
        results/tier_2_machine_cmu4/tier2 \
        results/tier_2_machine_cmu5/tier2 \
        results/tier_2_machine_cmu6/tier2 \
        results/tier_2_machine_cmu7/tier2 \
        results/tier_2_machine_cmu8/tier2 \
    --target_dir results/tier_2_full

echo "=== Tier 2 Aggregation DONE — submitting Tier 3 + Supplementary ==="

chmod +x "$REPO/scripts/run_tier3_node.sh"
chmod +x "$REPO/scripts/run_supp_node.sh"
mkdir -p "$REPO/logs"

TIER3_JOB=$(sbatch --parsable "$REPO/scripts/run_tier3_node.sh")
SUPP_JOB=$(sbatch --parsable "$REPO/scripts/run_supp_node.sh")

echo "Tier 3 array job submitted:        $TIER3_JOB"
echo "Supplementary array job submitted: $SUPP_JOB"
echo "Both run in parallel."
echo "Monitor: squeue -u \$USER"
