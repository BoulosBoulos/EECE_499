#!/bin/bash
# Job 2: Actor-KL ablation — 30 training runs.
# soft_hjb_aux with --lambda_actor_kl 0 (lambda_residual 0.2 unchanged)
# eikonal_aux with --beta_KL 0 (w_eik 1.0, w_fail 50.0 unchanged)
# Cells: 2_dense/right_left, 2_dense/stem_right, 1b/stem_right
# Seeds: 42 123 456 789 999 — 400k steps, no-intent, occlusion on (buildings on)
# Dir naming: T2d_{scenario}_{maneuver}_{method}_nokl_s{seed}
# Output root: results/tier_2_machine_job2/tier2/2d_actor_kl_ablation/
# Eval: 200 episodes/seed x 3 seeds = 600 total

#SBATCH --job-name=job2_actor_kl
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job2_actor_kl_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job2_actor_kl_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00

set -uo pipefail

REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0
export REPO
export OUT_ROOT="$REPO/results/tier_2_machine_job2/tier2/2d_actor_kl_ablation"

mkdir -p "$OUT_ROOT"

echo "=== JOB 2 Actor-KL Ablation START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$(hostname)  nproc=$(nproc)"
echo "    output_root=$OUT_ROOT"

python3 - <<'PYEOF'
import os, sys, subprocess, time, threading

REPO    = os.environ["REPO"]
OUT_ROOT = os.environ["OUT_ROOT"]
sys.path.insert(0, REPO)

CELLS = [
    ("2_dense", "right_left"),
    ("2_dense", "stem_right"),
    ("1b",      "stem_right"),
]
SEEDS = [42, 123, 456, 789, 999]
EVAL_SEED_OFFSETS = [1000, 2000, 3000]
N_EVAL_EPISODES = 200   # 3 seeds x 200 = 600 episodes
TOTAL_STEPS = 400_000

# Method-specific extra training args (KL disabled; everything else matches standard tier2)
METHOD_EXTRA = {
    "soft_hjb_aux": [
        "--lambda_actor_kl", "0",
        "--lambda_residual",  "0.2",
        "--tau_soft",         "1.0",
    ],
    "eikonal_aux": [
        "--beta_KL", "0",
        "--w_eik",   "1.0",
        "--w_fail",  "50.0",
    ],
}

jobs = []
for (scen, man) in CELLS:
    for seed in SEEDS:
        for method, extra in METHOD_EXTRA.items():
            tag     = f"T2d_{scen}_{man}_{method}_nokl_s{seed}"
            out_dir = os.path.join(OUT_ROOT, tag)
            ckpt    = os.path.join(out_dir, f"model_{method}_{scen}_{man}.pt")
            eval_seeds = [seed + off for off in EVAL_SEED_OFFSETS]

            train_cmd = [
                "python3", f"experiments/pde/train_{method}.py",
                "--scenario",    scen,
                "--ego_maneuver", man,
                "--seed",        str(seed),
                "--output_dir",  out_dir,
                "--total_steps", str(TOTAL_STEPS),
            ] + extra

            eval_cmd = [
                "python3", "experiments/pde/eval.py",
                "--method",      method,
                "--checkpoint",  ckpt,
                "--scenario",    scen,
                "--ego_maneuver", man,
                "--episodes",    str(N_EVAL_EPISODES),
                "--seeds",       *[str(s) for s in eval_seeds],
                "--out_dir",     out_dir,
                "--save_failures", "--max_failures", "5",
            ]

            already_done = os.path.isfile(os.path.join(out_dir, "eval_metrics.csv"))
            jobs.append(dict(tag=tag, out_dir=out_dir, train_cmd=train_cmd,
                             eval_cmd=eval_cmd, done=already_done))

print(f"[job2] {len(jobs)} jobs  ({sum(j['done'] for j in jobs)} already eval'd)")

# Parallelise up to (nproc-2) training processes concurrently
MAX_PARALLEL = max(1, os.cpu_count() - 2)
sem    = threading.Semaphore(MAX_PARALLEL)
errors = []
lock   = threading.Lock()

def run_one(job):
    tag     = job["tag"]
    out_dir = job["out_dir"]
    logfile = os.path.join(out_dir, "job2_run.log")
    os.makedirs(out_dir, exist_ok=True)

    if job["done"]:
        print(f"[job2] SKIP {tag}")
        return

    with sem:
        t0 = time.time()
        print(f"[job2] START {tag}", flush=True)
        with open(logfile, "w") as lf:
            r = subprocess.run(job["train_cmd"], cwd=REPO, stdout=lf, stderr=lf)
            if r.returncode != 0:
                with lock:
                    errors.append(f"TRAIN FAILED rc={r.returncode}: {tag}")
                print(f"[job2] TRAIN ERR {tag}", flush=True)
                return
            r = subprocess.run(job["eval_cmd"], cwd=REPO, stdout=lf, stderr=lf)
            if r.returncode != 0:
                with lock:
                    errors.append(f"EVAL FAILED rc={r.returncode}: {tag}")
                print(f"[job2] EVAL ERR {tag}", flush=True)
                return
        print(f"[job2] DONE {tag} ({(time.time()-t0)/60:.1f}m)", flush=True)

threads = [threading.Thread(target=run_one, args=(j,), daemon=True) for j in jobs]
for t in threads:
    t.start()
for t in threads:
    t.join()

if errors:
    print(f"\n[job2] FAILED ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
print(f"\n[job2] ALL {len(jobs)} jobs complete.")
PYEOF

echo "=== JOB 2 Training+Eval DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Produce the tier2d comparison CSV
echo "--- Aggregating tier2d_actor_kl_ablation.csv ---"
python3 scripts/aggregate_job2.py \
    --nokl_root  "$OUT_ROOT" \
    --kl_on_roots \
        "results/tier_2_machine_cmu8/tier2/2a_lambda_sweep" \
        "results/tier_2_machine_cmu3/tier2/2a_lambda_sweep" \
        "results/tier_2_machine_cmu8/tier2/2b_occlusion_sweep" \
        "results/tier_2_machine_cmu3/tier2/2b_occlusion_sweep" \
    --out "results/tables/tier2d_actor_kl_ablation.csv" \
    && echo "[job2] tier2d_actor_kl_ablation.csv written." \
    || echo "[job2] WARNING: aggregation failed — run manually."

echo "=== JOB 2 COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
