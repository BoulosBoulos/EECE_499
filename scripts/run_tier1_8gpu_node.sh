#!/bin/bash
# Single-node 8-GPU Tier-1 job.
# Requests one whole node with 8 GPUs.  Inside the job, 8 background
# orchestrators launch — one per GPU — each driving as many SUMO processes
# as the per-process CPU share allows.
#
# Phase 1: 8 × 30 = 240 jobs (method-specific, §6.1 assignments)
# Phase 2: 8 × 150 = 1200-job Tier-1 slice, one 1/8 slice per machine
#
# Outputs: results/tier_1_machine_cmuN/tier1/  (N = 1..8)

#SBATCH --job-name=tier1_8gpu
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier1_8gpu_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/tier1_8gpu_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=1-23:00:00
#SBATCH --exclusive

set -uo pipefail

export REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"

source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")

TOTAL_CPUS=$(nproc)
GPUS=8
# Leave 2 cores for OS/orchestrator overhead; divide rest evenly across 8 GPU workers.
# Each worker gets its own SUMO-parallel budget: MP = floor((TOTAL_CPUS - 2) / GPUS)
MP=$(( (TOTAL_CPUS - 2) / GPUS ))
[ "$MP" -lt 1 ] && MP=1

echo "=== TIER-1 8-GPU NODE JOB START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  nproc=$TOTAL_CPUS  gpus=$GPUS  MP_per_gpu=$MP"
echo "    SUMO_HOME=$SUMO_HOME"
echo "    Total concurrent SUMO processes: $(( MP * GPUS ))"

# ── Distributed manifest ─────────────────────────────────────────────────────
for i in $(seq 1 $GPUS); do
  MID="cmu${i}"
  echo "$(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$ gpu=$((i-1)) machine_id=$MID max_parallel=$MP nproc=$TOTAL_CPUS" \
    >> "$REPO/tier1_distributed_manifest.txt"
done

# ── Machine assignments (§6.1) ───────────────────────────────────────────────
declare -A METHOD SEEDS
METHOD[1]=fusion_aux;    SEEDS[1]="42 123 456 789 999"
METHOD[2]=hjb_aux;       SEEDS[2]="42 123 456 789 999"
METHOD[3]=soft_hjb_aux;  SEEDS[3]="42 123 456 789 999"
METHOD[4]=eikonal_aux;   SEEDS[4]="42 123 456 789 999"
METHOD[5]=rule_based;    SEEDS[5]="42 123 456 789 999"
METHOD[6]=fusion_aux;    SEEDS[6]="1111 2222 3333 4444 5555"
METHOD[7]=hjb_aux;       SEEDS[7]="1111 2222 3333 4444 5555"
METHOD[8]=soft_hjb_aux;  SEEDS[8]="1111 2222 3333 4444 5555"

# ── Compute Phase-2 slice boundaries (done once, shared by all workers) ──────
P2_JSON=$(python - <<'PYEOF'
import os, sys, json
sys.path.insert(0, os.environ['REPO'])
from types import SimpleNamespace
from experiments.pde.run_full_ablation import (
    generate_jobs, _apply_filters, _partition_balance_jobs
)
n_machines = 8
args = SimpleNamespace(
    seeds=None, scenarios=None, maneuvers=None, intents=None,
    include_rule_based=True,
    methods=['hjb_aux', 'soft_hjb_aux', 'eikonal_aux', 'fusion_aux', 'rule_based'],
)
jobs = generate_jobs('1', 400000)
jobs = _apply_filters(jobs, args)
jobs = _partition_balance_jobs(jobs)
total = len(jobs)
per   = total // n_machines
slices = {}
for m in range(1, n_machines + 1):
    s = (m - 1) * per
    e = s + per if m < n_machines else total
    slices[str(m)] = {"start": s, "end": e}
print(json.dumps(slices))
PYEOF
)
echo "Phase 2 slices: $P2_JSON"

# ── per-GPU worker function ───────────────────────────────────────────────────
run_one_gpu() {
    local gpu_id=$1          # 0..7
    local machine_num=$((gpu_id + 1))
    local machine_id="cmu${machine_num}"
    local method="${METHOD[$machine_num]}"
    local seeds="${SEEDS[$machine_num]}"
    local log="$REPO/logs/worker_${machine_id}.log"

    # rule_based is eval-only; no GPU training → can use full MP
    local mp=$MP
    if [ "$method" = "rule_based" ]; then
        mp=$(( MP + 1 ))   # give it back the 1/8 share we withheld
    fi

    export CUDA_VISIBLE_DEVICES=$gpu_id

    {
        echo "=== [$machine_id] GPU $gpu_id START $(date -u +%Y-%m-%dT%H:%M:%SZ) method=$method seeds='$seeds' MP=$mp ==="

        # ── PHASE 1 ─────────────────────────────────────────────────────────
        echo "--- [$machine_id] PHASE 1 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
        python experiments/pde/run_full_ablation.py \
            --tier 1 --total_steps 400000 \
            --max_parallel "$mp" \
            --scenarios 1a 1b 2_dense \
            --maneuvers stem_right right_left \
            --seeds $seeds \
            --methods "$method" \
            --machine_id "$machine_id" \
            || echo "[$machine_id] WARN: Phase 1 non-zero exit"
        echo "--- [$machine_id] PHASE 1 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"

        # ── PHASE 2 ─────────────────────────────────────────────────────────
        p2_start=$(python -c "import json; d=json.loads('$P2_JSON'); print(d['$machine_num']['start'])")
        p2_end=$(  python -c "import json; d=json.loads('$P2_JSON'); print(d['$machine_num']['end'])")
        echo "--- [$machine_id] PHASE 2 START jobs[$p2_start,$p2_end) $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
        python experiments/pde/run_full_ablation.py \
            --tier 1 --total_steps 400000 \
            --max_parallel "$mp" \
            --methods hjb_aux soft_hjb_aux eikonal_aux fusion_aux rule_based \
            --job_index_start "$p2_start" \
            --job_index_end   "$p2_end" \
            --machine_id      "${machine_id}_p2" \
            || echo "[$machine_id] WARN: Phase 2 non-zero exit"
        echo "=== [$machine_id] ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    } >> "$log" 2>&1 &

    echo "Launched $machine_id on GPU $gpu_id (PID $!) → $log"
}

# ── Launch all 8 GPU workers in the background ───────────────────────────────
PIDS=()
for gpu in $(seq 0 $((GPUS - 1))); do
    run_one_gpu $gpu
    PIDS+=($!)
done

echo ""
echo "All 8 GPU workers launched. PIDs: ${PIDS[*]}"
echo "Per-worker logs: $REPO/logs/worker_cmu{1..8}.log"
echo ""
echo "Watch saturation (run from login node):"
echo "  ssh $(hostname) 'nvidia-smi; uptime; ps aux | grep sumo | wc -l'"

# ── Wait for all workers ──────────────────────────────────────────────────────
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$(( FAILED + 1 ))
done

echo ""
echo "=== ALL WORKERS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) | failed=$FAILED ==="
echo "Results: $REPO/results/tier_1_machine_cmu{1..8}/"
