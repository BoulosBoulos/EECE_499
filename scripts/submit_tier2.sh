#!/bin/bash
# Submit the 8-machine Tier 2 ablation to Babel/SLURM.
# Run from EECE_499 repo root AFTER Tier 1 is fully aggregated:
#   bash scripts/submit_tier2.sh
#
# What gets submitted:
#   One SLURM array job (8 tasks), each task = one machine (cmu1–cmu8).
#   Resources per task: 1 GPU, 32 CPUs, 64 GB RAM, 7-day wall time.
#   Each machine runs 145 of 1,160 Tier 2 jobs (400k steps each).
#
# Sub-grids:
#   2a — lambda residual sweep: 5 methods × 2 scen × 2 man × 5 seeds × 6 λ  = 600 jobs
#   2b — occlusion sweep:       5 methods × 4 scen × 2 man × 5 seeds × 2 occ = 400 jobs
#   2c — fusion weight sweep:   fusion_aux × 2 scen × 2 man × 5 seeds × 8 w  = 160 jobs
#
# Expected wall time per machine: ~7–8h (145 jobs / 30 parallel × 85 min/job)

set -euo pipefail

REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
SCRIPT="$REPO/scripts/run_tier2_node.sh"
LOGS="$REPO/logs"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -d "$REPO/.venv" ]; then
    echo "ERROR: .venv not found at $REPO/.venv"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found"
    exit 1
fi

# Verify Tier 1 results exist before submitting Tier 2
T1_DIRS=$(ls -d "$REPO/results/tier_1_machine_cmu"*/tier1 2>/dev/null | wc -l)
if [ "$T1_DIRS" -lt 7 ]; then
    echo "WARNING: only $T1_DIRS Tier 1 machine directories found (expected 8)."
    echo "         Run Tier 1 aggregation first, then resubmit."
    read -p "Submit Tier 2 anyway? [y/N]: " yn
    [ "${yn,,}" = "y" ] || exit 0
fi

chmod +x "$SCRIPT"
mkdir -p "$LOGS"

# ── Submit ────────────────────────────────────────────────────────────────────
echo "Submitting Tier 2 — 8-machine SLURM array to general partition..."
echo "Script:    $SCRIPT"
echo "Resources: 1 GPU, 32 CPUs, 64 GB, 7 days per task"
echo "Jobs:      1,160 total / 145 per machine / 400k steps each"
echo ""

ARRAY_JOB=$(sbatch --parsable "$SCRIPT")
echo "Array job submitted: $ARRAY_JOB"
echo ""
echo "Tasks:"
for i in $(seq 1 8); do
    printf "  %s_%d → cmu%d (jobs [%d,%d))\n" \
        "$ARRAY_JOB" "$i" "$i" \
        "$(( (i-1)*145 ))" "$(( i < 8 ? i*145 : 1160 ))"
done
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f $LOGS/tier2_${ARRAY_JOB}_1.out"
echo ""
echo "Results tree (after jobs finish):"
echo "  $REPO/results/tier_2_machine_cmu{1..8}/tier2/"
echo ""
echo "Aggregation (after all 8 machines finish):"
cat <<'AGGEOF'
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
AGGEOF
