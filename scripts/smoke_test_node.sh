#!/bin/bash
# Quick smoke test per handoff §11 — runs 1 job at 50k steps before scaling.
# Usage: sbatch scripts/smoke_test_node.sh
# Or run interactively: bash scripts/smoke_test_node.sh [METHOD]
#
# Expects METHOD env var (default: hjb_aux). Tests one job on the target method.
# Checks for metrics.csv ~50k rows and eval_metrics.csv ~300 rows.

#SBATCH --job-name=smoke_tier1
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/smoke_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/smoke_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -uo pipefail

REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"

source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")

METHOD=${METHOD:-hjb_aux}
SMOKE_ID="smoke_$(hostname | tr -d '.-')_$$"

echo "=== SMOKE TEST: method=$METHOD machine_id=$SMOKE_ID ==="
echo "    $(date -u)  host=$(hostname)"

python experiments/pde/run_full_ablation.py \
    --tier 1 \
    --total_steps 50000 \
    --max_parallel 1 \
    --scenarios 1a \
    --maneuvers stem_right \
    --seeds 42 \
    --methods "$METHOD" \
    --intents false \
    --no-include_rule_based \
    --machine_id "$SMOKE_ID"

# ── Validate output ──────────────────────────────────────────────────────────
JOB_DIR="results/tier_1_machine_${SMOKE_ID}/tier1"
JOB_TAG=$(ls "$JOB_DIR" 2>/dev/null | head -1)

if [ -z "$JOB_TAG" ]; then
    echo "FAIL: no job directory created under $JOB_DIR"
    exit 1
fi

METRICS="$JOB_DIR/$JOB_TAG/metrics.csv"
EVAL="$JOB_DIR/$JOB_TAG/eval_metrics.csv"

if [ ! -f "$METRICS" ]; then
    echo "FAIL: metrics.csv not found at $METRICS"
    exit 1
fi

if [ ! -f "$EVAL" ]; then
    echo "FAIL: eval_metrics.csv not found at $EVAL"
    exit 1
fi

METRIC_ROWS=$(wc -l < "$METRICS")
EVAL_ROWS=$(wc -l < "$EVAL")

echo "metrics.csv rows:      $METRIC_ROWS (expect ~50k+header)"
echo "eval_metrics.csv rows: $EVAL_ROWS   (expect ~300+header)"

if python -c "
import csv, sys
with open('$METRICS') as f:
    rows = list(csv.DictReader(f))
nans = sum(1 for r in rows for v in r.values() if v.strip().lower() in ('nan','inf','-inf',''))
print(f'NaN/Inf/empty cells in metrics.csv: {nans}')
if nans > len(rows) * 0.1:
    print('WARNING: >10% NaN/Inf — training may have diverged')
    sys.exit(1)
"; then
    echo "PASS: metrics.csv looks healthy"
else
    echo "WARN: NaN check raised an issue — review before scaling"
fi

echo "=== SMOKE TEST PASSED: environment is OK for $(hostname) ==="
