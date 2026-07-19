#!/bin/bash
# Job 4: three eval backfills through the OFFICIAL eval.py, one parallel pool.
#   4a  fusion_aux occOFF checkpoints evaluated under occlusion-ON  -> tier4_HO2 (completes HO2)
#   4b  rule_based on 3_dense/stem_right and 4_dense/stem_right     -> tier3_dense
#   4c  Tier-1 intent runs missing their eval (checkpoint present)  -> backfill in place
# All three are eval-only. Every job is skip-if-eval-exists so nothing is overwritten.
# Aggregation is done separately (fast, no SUMO) after this job completes.

#SBATCH --job-name=job4_evals
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job4_evals_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job4_evals_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-16:00:00

set -uo pipefail
REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export REPO

echo "############ JOB 4 EVAL BACKFILLS $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) ############"

python3 - <<'PYEOF'
import os, re, sys, glob, subprocess, threading, time
REPO = os.environ["REPO"]; sys.path.insert(0, REPO)

METHODS = ["soft_hjb_aux","hjb_aux","eikonal_aux","fusion_aux","cbf_aux","drppo","rule_based"]
SCENS = ["4_dense","3_dense","2_dense","1a","1b","1c","1d","2","3","4"]
MANS  = ["stem_right","stem_left","right_left","right_stem","left_right","left_stem"]
OFFSETS = [1000, 2000, 3000]
EPISODES = "100"

def parse_scen_man(rest):
    for s in SCENS:
        if rest.startswith(s + "_"):
            m = rest[len(s)+1:]
            if m in MANS: return s, m
    return None, None

def named_eval_exists(d):
    return any(f.startswith("eval_") and f.endswith(".csv") and f != "eval_metrics.csv"
              for f in os.listdir(d)) if os.path.isdir(d) else False

def eval_cmd(method, scen, man, ckpt, out_dir, seed, use_intent, no_buildings=False):
    seeds = [str(int(seed)+o) for o in OFFSETS]
    cmd = ["python3","experiments/pde/eval.py","--method",method,
           "--scenario",scen,"--ego_maneuver",man,"--episodes",EPISODES,
           "--seeds",*seeds,"--out_dir",out_dir,"--save_failures","--max_failures","5"]
    if method != "rule_based":
        cmd += ["--checkpoint", ckpt]
    if use_intent: cmd.append("--use_intent")
    if no_buildings: cmd.append("--no_buildings")
    return cmd

jobs = []  # (label, out_dir, cmd)

# ---- 4a: fusion occOFF -> occ-ON transfer ------------------------------------
for d in sorted(glob.glob(os.path.join(REPO,"results/tier_2_machine_cmu*/tier2/2b_occlusion_sweep/*fusion_aux*occOFF*"))):
    base = os.path.basename(d)                       # T2b_{scen}_{man}_fusion_aux_occOFF_s{seed}
    seed_m = re.search(r"_s(\d+)$", base)
    if not seed_m: continue
    seed = seed_m.group(1)
    body = re.sub(r"^T2b_","",base); body = re.sub(r"_fusion_aux_occOFF_s\d+$","",body)
    scen, man = parse_scen_man(body)
    if scen is None: continue
    ckpt = os.path.join(d, f"model_fusion_aux_{scen}_{man}.pt")
    if not os.path.isfile(ckpt): continue
    out_dir = os.path.join(REPO,"results/tier4_HO2_noocc_to_occ",
                           f"{scen}_{man}_fusion_aux_nointent_s{seed}")
    if named_eval_exists(out_dir): continue
    os.makedirs(out_dir, exist_ok=True)
    jobs.append((f"4a_fusion_{scen}_{man}_s{seed}", out_dir,
                 eval_cmd("fusion_aux",scen,man,ckpt,out_dir,seed,use_intent=False,no_buildings=False)))

# ---- 4b: rule_based dense stress ---------------------------------------------
for scen in ["3_dense","4_dense"]:
    for seed in ["42","123","456","789","999"]:
        out_dir = os.path.join(REPO,"results/tier_3_full/tier3_dense",
                               f"{scen}_stem_right_rule_based_s{seed}")
        if named_eval_exists(out_dir): continue
        os.makedirs(out_dir, exist_ok=True)
        jobs.append((f"4b_rulebased_{scen}_s{seed}", out_dir,
                     eval_cmd("rule_based",scen,"stem_right",None,out_dir,seed,use_intent=False)))

# ---- 4c: Tier-1 intent eval backfill (checkpoint present, eval missing) -------
root = os.path.join(REPO,"results/tier_1_full")
for base in sorted(os.listdir(root)):
    d = os.path.join(root, base)
    if not os.path.isdir(d) or "_intent_" not in base or "_nointent_" in base: continue
    if named_eval_exists(d): continue
    method = next((m for m in sorted(METHODS,key=len,reverse=True) if f"_{m}_" in base), None)
    if method is None or method == "rule_based": continue     # rule_based has no model
    seed_m = re.search(r"_s(\d+)$", base)
    if not seed_m: continue
    seed = seed_m.group(1)
    body = base.split(f"_{method}_")[0]
    scen, man = parse_scen_man(body)
    if scen is None: continue
    ckpt = os.path.join(d, f"model_{method}_{scen}_{man}.pt")
    if not (os.path.isfile(ckpt) and "_step" not in os.path.basename(ckpt)): continue
    jobs.append((f"4c_{method}_{scen}_{man}_s{seed}", d,
                 eval_cmd(method,scen,man,ckpt,d,seed,use_intent=True)))

print(f"[job4] total eval jobs: {len(jobs)}", flush=True)
from collections import Counter
print("[job4] by phase:", Counter(j[0].split('_')[0] for j in jobs), flush=True)

MAX_PARALLEL = max(1, (os.cpu_count() or 4) - 1)
sem = threading.Semaphore(MAX_PARALLEL); errors=[]; lock=threading.Lock(); done=[0]
def run_one(label, out_dir, cmd):
    with sem:
        t0=time.time()
        with open(os.path.join(out_dir,"job4_eval.log"),"w") as lf:
            r = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=lf)
        with lock:
            done[0]+=1
            if r.returncode!=0: errors.append(f"rc={r.returncode}: {label}")
            print(f"[job4] {'OK' if r.returncode==0 else 'FAIL'} {label} "
                  f"({(time.time()-t0)/60:.1f}m)  [{done[0]}/{len(jobs)}]", flush=True)
ths=[threading.Thread(target=run_one,args=j) for j in jobs]
for t in ths: t.start()
for t in ths: t.join()
print(f"\n[job4] complete. {len(jobs)-len(errors)}/{len(jobs)} ok.", flush=True)
if errors:
    print(f"[job4] {len(errors)} failures:")
    for e in errors[:40]: print("  ",e)
    sys.exit(1)
PYEOF

echo "############ JOB 4 EVALS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ############"
