#!/bin/bash
# Submit Tier 4 held-out evaluation after Tier 3 + Supp both complete.
# Chained by auto_aggregate_and_submit_tier3_supp.sh via:
#   --dependency=afterany:<T3_last>:<SUPP_last>

#SBATCH --job-name=agg_t4_submit
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/agg_t4_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/agg_t4_%j.err
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

echo "=== Tier 4 Submit START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

chmod +x "$REPO/scripts/run_tier4_node.sh"
mkdir -p "$REPO/logs"

# Submit with 2 retry windows (skip-if-done makes retries safe).
T4_R1=$(sbatch --parsable "$REPO/scripts/run_tier4_node.sh")
T4_R2=$(sbatch --parsable --dependency=afterany:${T4_R1} "$REPO/scripts/run_tier4_node.sh")

echo "Tier 4 retry chain: $T4_R1 → $T4_R2"
echo "Monitor: squeue -u \$USER"
