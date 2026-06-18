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

# ── Completeness guard ────────────────────────────────────────────────────────
# Count Phase 2 completed jobs across all 8 machines.
# A job is done when it has eval_metrics.csv (covers both trainable and rule_based).
# Expected: 1200 total (150 per machine × 8).
P2_DONE=$(find \
    results/tier_1_machine_cmu1_p2/tier1 \
    results/tier_1_machine_cmu2_p2/tier1 \
    results/tier_1_machine_cmu3_p2/tier1 \
    results/tier_1_machine_cmu4_p2/tier1 \
    results/tier_1_machine_cmu5_p2/tier1 \
    results/tier_1_machine_cmu6_p2/tier1 \
    results/tier_1_machine_cmu7_p2/tier1 \
    results/tier_1_machine_cmu8_p2/tier1 \
    -maxdepth 2 -name "eval_metrics.csv" 2>/dev/null \
    | xargs -I{} dirname {} | sort -u | wc -l)
echo "Phase 2 completed jobs: ${P2_DONE}/1200"
if [ "$P2_DONE" -lt 1200 ]; then
    echo "ERROR: Phase 2 incomplete (${P2_DONE}/1200). Aborting — do not aggregate yet."
    exit 1
fi
echo "Phase 2 complete. Proceeding with aggregation."
# ─────────────────────────────────────────────────────────────────────────────

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

# Submit Tier 2 with 4 retry windows chained afterany.
# Skip-if-done in run_full_ablation.py makes retries safe — already-done jobs are skipped.
# afterany (not afterok) ensures retries fire even when tasks time out.
T2_R1=$(sbatch --parsable "$REPO/scripts/run_tier2_node.sh")
T2_R2=$(sbatch --parsable --dependency=afterany:${T2_R1} "$REPO/scripts/run_tier2_node.sh")
T2_R3=$(sbatch --parsable --dependency=afterany:${T2_R2} "$REPO/scripts/run_tier2_node.sh")
T2_R4=$(sbatch --parsable --dependency=afterany:${T2_R3} "$REPO/scripts/run_tier2_node.sh")
echo "Tier 2 retry chain: $T2_R1 → $T2_R2 → $T2_R3 → $T2_R4"

# Chain Tier 3+Supp aggregator after last retry (afterany so timeout does not block).
# Completeness guard inside auto_aggregate_and_submit_tier3_supp.sh aborts if < 1160 done.
AGG3_JOB=$(sbatch --parsable --dependency=afterany:${T2_R4} \
    "$REPO/scripts/auto_aggregate_and_submit_tier3_supp.sh")
echo "Tier 3/Supp aggregator queued: $AGG3_JOB (afterany:$T2_R4, completeness-guarded)"
echo ""
echo "Chain: Tier2 ($T2_R1→$T2_R2→$T2_R3→$T2_R4) → Agg+Tier3+Supp ($AGG3_JOB)"
echo "Monitor: squeue -u \$USER"
