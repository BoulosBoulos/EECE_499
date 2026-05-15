#!/bin/bash
# Auto-aggregate Tier 1 (Phase 1 + Phase 2) and submit Tier 2.
# Submit with: sbatch --dependency=afterok:<phase2_job_id> scripts/auto_aggregate_and_submit_tier2.sh
#
# Dependency should be on the Phase 2 job (run_tier1_phase2_general.sh),
# NOT on the original Tier 1 job (which only covers Phase 1).

#SBATCH --job-name=agg_t2_submit
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/agg_t2_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/agg_t2_%j.err
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

echo "=== Tier 1 Aggregation START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Aggregate Phase 1 (paper-critical 3 scenarios, per machine method)
echo "--- Phase 1 ---"
python experiments/pde/aggregate_tier_1_results.py \
    --source_dirs \
        results/tier_1_machine_cmu1/tier1 \
        results/tier_1_machine_cmu2/tier1 \
        results/tier_1_machine_cmu3/tier1 \
        results/tier_1_machine_cmu4/tier1 \
        results/tier_1_machine_cmu5/tier1 \
        results/tier_1_machine_cmu6/tier1 \
        results/tier_1_machine_cmu7/tier1 \
        results/tier_1_machine_cmu8/tier1 \
    --target_dir results/tier_1_full

# Aggregate Phase 2 (full 12-cell manifest, 5 methods)
echo "--- Phase 2 ---"
python experiments/pde/aggregate_tier_1_results.py \
    --source_dirs \
        results/tier_1_machine_cmu1_p2/tier1 \
        results/tier_1_machine_cmu2_p2/tier1 \
        results/tier_1_machine_cmu3_p2/tier1 \
        results/tier_1_machine_cmu4_p2/tier1 \
        results/tier_1_machine_cmu5_p2/tier1 \
        results/tier_1_machine_cmu6_p2/tier1 \
        results/tier_1_machine_cmu7_p2/tier1 \
        results/tier_1_machine_cmu8_p2/tier1 \
    --target_dir results/tier_1_full

echo "=== Tier 1 Aggregation DONE — submitting Tier 2 ==="

chmod +x "$REPO/scripts/run_tier2_node.sh"
chmod +x "$REPO/scripts/auto_aggregate_and_submit_tier3_supp.sh"
mkdir -p "$REPO/logs"

TIER2_JOB=$(sbatch --parsable "$REPO/scripts/run_tier2_node.sh")
echo "Tier 2 array job submitted: $TIER2_JOB"

# Chain Tier 3 + Supp to run after Tier 2 completes
AGG3_JOB=$(sbatch --parsable --dependency=afterok:${TIER2_JOB} \
    "$REPO/scripts/auto_aggregate_and_submit_tier3_supp.sh")
echo "Tier 3/Supp aggregator queued: $AGG3_JOB (triggers after Tier 2 finishes)"
echo ""
echo "Full chain: Tier2 ($TIER2_JOB) → Tier3+Supp aggregator ($AGG3_JOB) → Tier3/Supp jobs"
echo "Monitor: squeue -u \$USER"
