#!/bin/bash
# Job 3: hjb_aux lambda=0.1 replication — 5 new seeds for 2_dense/right_left.
# Existing seeds {7, 42, 123, 456, 789} (n=5) already done in cmu6/cmu7.
# New seeds {1111, 2222, 3333, 4444, 5555} → cmu9 → n=10 total.
# Produces: results/tier_2_machine_cmu9/tier2/2a_lambda_sweep/T2a_2_dense_right_left_hjb_aux_lam0.1_s{seed}/
# After training: patches aggregate_t2a to scan cmu9, regenerates tier2a_lambda_sensitivity.csv.

#SBATCH --job-name=job3_hjb_rep
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job3_hjb_rep_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job3_hjb_rep_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00

set -uo pipefail

REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0
export REPO

MACHINE_DIR="$REPO/results/tier_2_machine_cmu9/tier2/2a_lambda_sweep"
mkdir -p "$MACHINE_DIR"

echo "=== JOB 3 hjb_aux λ=0.1 Replication START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  output_root=$MACHINE_DIR"

python3 - <<'PYEOF'
import os, sys, subprocess, time, threading

REPO        = os.environ["REPO"]
MACHINE_DIR = os.path.join(REPO, "results", "tier_2_machine_cmu9", "tier2", "2a_lambda_sweep")
sys.path.insert(0, REPO)

SCENARIO   = "2_dense"
MANEUVER   = "right_left"
METHOD     = "hjb_aux"
LAM        = 0.1
NEW_SEEDS  = [1111, 2222, 3333, 4444, 5555]
EVAL_SEED_OFFSETS = [1000, 2000, 3000]
N_EVAL_EPISODES   = 200   # 3 seeds x 200 = 600 episodes
TOTAL_STEPS       = 400_000

jobs = []
for seed in NEW_SEEDS:
    tag     = f"T2a_{SCENARIO}_{MANEUVER}_{METHOD}_lam{LAM}_s{seed}"
    out_dir = os.path.join(MACHINE_DIR, tag)
    ckpt    = os.path.join(out_dir, f"model_{METHOD}_{SCENARIO}_{MANEUVER}.pt")
    eval_seeds = [seed + off for off in EVAL_SEED_OFFSETS]

    train_cmd = [
        "python3", f"experiments/pde/train_{METHOD}.py",
        "--scenario",      SCENARIO,
        "--ego_maneuver",  MANEUVER,
        "--seed",          str(seed),
        "--output_dir",    out_dir,
        "--total_steps",   str(TOTAL_STEPS),
        "--lambda_residual", str(LAM),
    ]

    eval_cmd = [
        "python3", "experiments/pde/eval.py",
        "--method",        METHOD,
        "--checkpoint",    ckpt,
        "--scenario",      SCENARIO,
        "--ego_maneuver",  MANEUVER,
        "--episodes",      str(N_EVAL_EPISODES),
        "--seeds",         *[str(s) for s in eval_seeds],
        "--out_dir",       out_dir,
        "--save_failures", "--max_failures", "5",
    ]

    already_done = os.path.isfile(os.path.join(out_dir, "eval_metrics.csv"))
    jobs.append(dict(tag=tag, out_dir=out_dir, train_cmd=train_cmd,
                     eval_cmd=eval_cmd, done=already_done))

print(f"[job3] {len(jobs)} runs  ({sum(j['done'] for j in jobs)} already done)")

MAX_PARALLEL = max(1, os.cpu_count() - 1)
sem    = threading.Semaphore(MAX_PARALLEL)
errors = []
lock   = threading.Lock()

def run_one(job):
    tag     = job["tag"]
    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if job["done"]:
        print(f"[job3] SKIP {tag}")
        return

    with sem:
        t0 = time.time()
        print(f"[job3] START {tag}", flush=True)
        logfile = os.path.join(out_dir, "job3_run.log")
        with open(logfile, "w") as lf:
            r = subprocess.run(job["train_cmd"], cwd=REPO, stdout=lf, stderr=lf)
            if r.returncode != 0:
                with lock:
                    errors.append(f"TRAIN FAILED rc={r.returncode}: {tag}")
                print(f"[job3] TRAIN ERR {tag}", flush=True)
                return
            r = subprocess.run(job["eval_cmd"], cwd=REPO, stdout=lf, stderr=lf)
            if r.returncode != 0:
                with lock:
                    errors.append(f"EVAL FAILED rc={r.returncode}: {tag}")
                print(f"[job3] EVAL ERR {tag}", flush=True)
                return
        print(f"[job3] DONE {tag} ({(time.time()-t0)/60:.1f}m)", flush=True)

threads = [threading.Thread(target=run_one, args=(j,), daemon=True) for j in jobs]
for t in threads:
    t.start()
for t in threads:
    t.join()

if errors:
    print(f"\n[job3] FAILED ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
print(f"\n[job3] ALL {len(jobs)} runs complete.")
PYEOF

echo "=== JOB 3 Training+Eval DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Patch aggregate_t2a to include cmu9, then regenerate tier2a table.
# This is a one-line patch: range(1, 9) → range(1, 10)
AGG_SCRIPT="$REPO/experiments/pde/analysis/generate_tier2_tier3_tables.py"
if grep -q "range(1, 9)" "$AGG_SCRIPT"; then
    sed -i 's/for m in range(1, 9):/for m in range(1, 10):/g' "$AGG_SCRIPT"
    echo "[job3] Patched aggregate_t2a to scan cmu1..cmu9."
elif grep -q "range(1, 10)" "$AGG_SCRIPT"; then
    echo "[job3] aggregate_t2a already scans cmu1..cmu9. No patch needed."
else
    echo "[job3] WARNING: Could not find range pattern in $AGG_SCRIPT — regenerate manually."
fi

echo "--- Regenerating tier2a_lambda_sensitivity.csv (n=10 for hjb_aux 2_dense/right_left λ=0.1) ---"
python3 experiments/pde/analysis/generate_tier2_tier3_tables.py \
    --repo "$REPO" \
    --out  "$REPO/results/tables/" \
    && echo "[job3] tier2a_lambda_sensitivity.csv updated." \
    || echo "[job3] WARNING: aggregation failed — run manually."

echo "=== JOB 3 COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
