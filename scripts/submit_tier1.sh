#!/bin/bash
# Submit the 8-machine Tier-1 ablation to Babel/SLURM (array partition).
# Run from EECE_499 repo root:
#   bash scripts/submit_tier1.sh
#
# What gets submitted:
#   One SLURM array job (8 tasks), each task = one machine (cmu1–cmu8).
#   Resources per task: 1 GPU, 32 CPUs, 64 GB RAM, 7-day wall time.
#
# Handoff §6.1 assignments:
#   Task 1 (cmu1): fusion_aux    seeds 42..999
#   Task 2 (cmu2): hjb_aux       seeds 42..999
#   Task 3 (cmu3): soft_hjb_aux  seeds 42..999
#   Task 4 (cmu4): eikonal_aux   seeds 42..999
#   Task 5 (cmu5): rule_based    seeds 42..999  (eval-only, ~1h/job)
#   Task 6 (cmu6): fusion_aux    seeds 1111..5555  (helper — extra seeds)
#   Task 7 (cmu7): hjb_aux       seeds 1111..5555  (helper)
#   Task 8 (cmu8): soft_hjb_aux  seeds 1111..5555  (helper)
#
# Local machine owns cbf_aux + drppo — NOT submitted here.
#
# After Phase 1 finishes (~10–14h on a 32-core node), each SLURM task
# automatically continues with Phase 2: its 1/8 slice of the full Tier-1
# manifest (5 methods × 12 cells × 10 seeds × 2 intents = 1200 jobs / 8 = 150).

set -euo pipefail

REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
SCRIPT="$REPO/scripts/run_tier1_node.sh"
SMOKE_SCRIPT="$REPO/scripts/smoke_test_node.sh"
LOGS="$REPO/logs"

if [ ! -d "$REPO/.venv" ]; then
    echo "ERROR: .venv not found. Run setup first:"
    echo "  cd $REPO"
    echo "  python3.12 -m venv .venv"
    echo "  .venv/bin/pip install eclipse-sumo==1.26.0 -r requirements.txt"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found"
    exit 1
fi

chmod +x "$SCRIPT" "$SMOKE_SCRIPT"
mkdir -p "$LOGS"

# ── Optional smoke test first ────────────────────────────────────────────────
if [ "${1:-}" = "--smoke" ]; then
    echo "Submitting smoke test to debug partition (runs 1 job at 50k steps)..."
    SMOKE_JOB=$(sbatch --parsable "$SMOKE_SCRIPT")
    echo "Smoke job submitted: $SMOKE_JOB"
    echo "Watch: squeue -j $SMOKE_JOB"
    echo "Log:   tail -f $LOGS/smoke_${SMOKE_JOB}.out"
    echo ""
    echo "After smoke passes, run without --smoke to submit the full 8-task array."
    exit 0
fi

# ── Submit the 8-machine array ───────────────────────────────────────────────
echo "Submitting Tier-1 8-machine SLURM array to 'array' partition..."
echo "Script:     $SCRIPT"
echo "Resources:  1 GPU, 32 CPUs, 64 GB, 7 days per task"
echo ""

ARRAY_JOB=$(sbatch --parsable "$SCRIPT")
echo "Array job submitted: $ARRAY_JOB"
echo ""
echo "Tasks:"
printf "  %s_1 → cmu1 (fusion_aux,   seeds 42..999)\n"    "$ARRAY_JOB"
printf "  %s_2 → cmu2 (hjb_aux,      seeds 42..999)\n"    "$ARRAY_JOB"
printf "  %s_3 → cmu3 (soft_hjb_aux, seeds 42..999)\n"    "$ARRAY_JOB"
printf "  %s_4 → cmu4 (eikonal_aux,  seeds 42..999)\n"    "$ARRAY_JOB"
printf "  %s_5 → cmu5 (rule_based,   seeds 42..999)\n"    "$ARRAY_JOB"
printf "  %s_6 → cmu6 (fusion_aux,   seeds 1111..5555)\n" "$ARRAY_JOB"
printf "  %s_7 → cmu7 (hjb_aux,      seeds 1111..5555)\n" "$ARRAY_JOB"
printf "  %s_8 → cmu8 (soft_hjb_aux, seeds 1111..5555)\n" "$ARRAY_JOB"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f $LOGS/tier1_${ARRAY_JOB}_1.out   # cmu1 live log"
echo "  tail -f $LOGS/tier1_${ARRAY_JOB}_N.out   # replace N with task 1-8"
echo ""
echo "Distributed manifest (updated by each node on start):"
echo "  cat $REPO/tier1_distributed_manifest.txt"
echo ""
echo "Watchdog checks (§12 — run 1h after launch):"
echo "  squeue -j ${ARRAY_JOB} -o '%i %N %T %R'  # node names + state"
echo "  grep 'MP=' $REPO/tier1_distributed_manifest.txt  # confirm max_parallel"
echo ""
echo "Results tree (after jobs finish):"
echo "  $REPO/results/tier_1_machine_cmu{1..8}/"
echo ""
echo "Aggregation (§9 — after all machines finish):"
cat <<'AGGEOF'
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
AGGEOF
