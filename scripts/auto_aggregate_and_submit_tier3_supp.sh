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

# ── Completeness guard ────────────────────────────────────────────────────────
# Count Tier 2 completed jobs across all 8 machines.
# A job is done when it has eval_metrics.csv. Expected: 1160 total.
T2_DONE=$(find \
    results/tier_2_machine_cmu1/tier2 \
    results/tier_2_machine_cmu2/tier2 \
    results/tier_2_machine_cmu3/tier2 \
    results/tier_2_machine_cmu4/tier2 \
    results/tier_2_machine_cmu5/tier2 \
    results/tier_2_machine_cmu6/tier2 \
    results/tier_2_machine_cmu7/tier2 \
    results/tier_2_machine_cmu8/tier2 \
    -name "eval_metrics.csv" 2>/dev/null \
    | xargs -I{} dirname {} | sort -u | wc -l)
echo "Tier 2 completed jobs: ${T2_DONE}/1160"
if [ "$T2_DONE" -lt 1160 ]; then
    echo "ERROR: Tier 2 incomplete (${T2_DONE}/1160). Aborting — submit more Tier 2 retries and requeue this script."
    exit 1
fi
echo "Tier 2 complete. Proceeding with aggregation."
# ─────────────────────────────────────────────────────────────────────────────

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

# Submit Tier 3 + Supp each with 2 retry windows (afterany chain, skip-if-done safe).
T3_R1=$(sbatch --parsable "$REPO/scripts/run_tier3_node.sh")
T3_R2=$(sbatch --parsable --dependency=afterany:${T3_R1} "$REPO/scripts/run_tier3_node.sh")

SUPP_R1=$(sbatch --parsable "$REPO/scripts/run_supp_node.sh")
SUPP_R2=$(sbatch --parsable --dependency=afterany:${SUPP_R1} "$REPO/scripts/run_supp_node.sh")

echo "Tier 3 retry chain:        $T3_R1 → $T3_R2"
echo "Supplementary retry chain: $SUPP_R1 → $SUPP_R2"
echo "Both Tier 3 + Supp run in parallel."

# Chain Tier 4 after BOTH T3 and Supp last retries finish.
# afterany:A:B waits for both A and B to reach any terminal state.
chmod +x "$REPO/scripts/auto_aggregate_and_submit_tier4.sh"
AGG4_JOB=$(sbatch --parsable --dependency=afterany:${T3_R2}:${SUPP_R2} \
    "$REPO/scripts/auto_aggregate_and_submit_tier4.sh")
echo "Tier 4 launcher queued:    $AGG4_JOB (afterany:$T3_R2:$SUPP_R2)"
echo ""
echo "Full chain: T3($T3_R1→$T3_R2) + Supp($SUPP_R1→$SUPP_R2) → T4-submit($AGG4_JOB) → T4"
echo "Monitor: squeue -u \$USER"
