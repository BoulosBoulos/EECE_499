#!/usr/bin/env bash
# Launch ablation jobs in parallel across N GPUs (default 16).
# Prerequisite: generate job manifest with experiments/generate_jobs.py
#
# Usage:
#   ./scripts/launch_parallel_16gpu.sh
#   ./scripts/launch_parallel_16gpu.sh --manifest results/ablation/job_manifest.json --num_gpus 8
#
# Each worker runs: run_single_job.py --worker_index i --num_workers N --gpu 0
# with CUDA_VISIBLE_DEVICES=i so that worker i uses GPU i.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MANIFEST="${MANIFEST:-results/ablation/job_manifest.json}"
NUM_GPUS="${NUM_GPUS:-16}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_STOCHASTIC=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --total_steps) TOTAL_STEPS="$2"; shift 2 ;;
    --eval_episodes) EVAL_EPISODES="$2"; shift 2 ;;
    --eval_stochastic) EVAL_STOCHASTIC="--eval_stochastic"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  echo "Run: python experiments/generate_jobs.py --out_dir results/ablation"
  exit 1
fi

echo "Using manifest: $MANIFEST"
echo "Spawning $NUM_GPUS workers (GPUs 0..$((NUM_GPUS-1)))"

OUT_DIR="$(dirname "$MANIFEST")"
mkdir -p "$OUT_DIR/jobs"

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python experiments/run_single_job.py \
    --manifest "$MANIFEST" \
    --worker_index $i \
    --num_workers "$NUM_GPUS" \
    --gpu 0 \
    --out_dir "$OUT_DIR" \
    --total_steps "$TOTAL_STEPS" \
    --eval_episodes "$EVAL_EPISODES" \
    $EVAL_STOCHASTIC \
    > "$OUT_DIR/jobs/worker_${i}.log" 2>&1 &
  PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} workers (PIDs: ${PIDS[*]})"
FAILED=0
for idx in "${!PIDS[@]}"; do
  pid=${PIDS[$idx]}
  if wait "$pid"; then
    echo "Worker $idx (PID $pid): OK"
  else
    echo "Worker $idx (PID $pid): FAILED (exit code $?)" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if [ $FAILED -gt 0 ]; then
  echo "WARNING: $FAILED / ${#PIDS[@]} workers failed. Check logs in $OUT_DIR/jobs/worker_*.log" >&2
else
  echo "All ${#PIDS[@]} workers finished successfully."
fi
echo "Merge results: python experiments/aggregate_results.py --out_dir $OUT_DIR --manifest $MANIFEST"

if [ $FAILED -gt 0 ]; then
  exit 1
fi
