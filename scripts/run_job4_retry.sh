#!/bin/bash
# Job 4 RETRY — array job. The first attempt (9341746) died from OOM:
# 15 parallel SUMO evals under --mem=32G (~2GB each) triggered 118 oom_kills,
# which killed the TraCI servers ("Could not connect to TraCI server") and then
# hit the 16h walltime. 47/271 finished (4a fusion + 4b rule_based); all 221
# Tier-1 intent backfills failed.
#
# Fix: 4 array tasks x 6 concurrent evals, 64G each (~10GB headroom per eval),
# 24h walltime. Work is partitioned round-robin across workers. Every job is
# skip-if-eval-exists, so the 47 already-finished evals are not redone.

#SBATCH --job-name=job4_retry
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job4_retry_%A_%a.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job4_retry_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=1-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -uo pipefail
REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export PYTHONUNBUFFERED=1
export REPO
export WORKER_ID="$SLURM_ARRAY_TASK_ID"
export N_WORKERS=4
export MAX_PARALLEL=6

echo "###### JOB4 RETRY worker=$WORKER_ID/$N_WORKERS $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) ######"

python3 - <<'PYEOF'
import os, re, sys, glob, subprocess, threading, time
REPO = os.environ["REPO"]
WID  = int(os.environ["WORKER_ID"]); NW = int(os.environ["N_WORKERS"])
MAXP = int(os.environ["MAX_PARALLEL"])

METHODS = ["soft_hjb_aux","hjb_aux","eikonal_aux","fusion_aux","cbf_aux","drppo","rule_based"]
SCENS = ["4_dense","3_dense","2_dense","1a","1b","1c","1d","2","3","4"]
MANS  = ["stem_right","stem_left","right_left","right_stem","left_right","left_stem"]
OFFSETS = [1000,2000,3000]; EPISODES = "100"

def psm(rest):
    for s in SCENS:
        if rest.startswith(s+"_"):
            m = rest[len(s)+1:]
            if m in MANS: return s,m
    return None,None

def has_eval(d):
    return any(f.startswith("eval_") and f.endswith(".csv") and f != "eval_metrics.csv"
               for f in os.listdir(d)) if os.path.isdir(d) else False

def ecmd(method, scen, man, ckpt, out_dir, seed, use_intent):
    seeds=[str(int(seed)+o) for o in OFFSETS]
    c=["python3","experiments/pde/eval.py","--method",method,"--scenario",scen,
       "--ego_maneuver",man,"--episodes",EPISODES,"--seeds",*seeds,
       "--out_dir",out_dir,"--save_failures","--max_failures","5"]
    if method!="rule_based": c+=["--checkpoint",ckpt]
    if use_intent: c.append("--use_intent")
    return c

jobs=[]
# 4a fusion occOFF -> occ-ON
for d in sorted(glob.glob(os.path.join(REPO,"results/tier_2_machine_cmu*/tier2/2b_occlusion_sweep/*fusion_aux*occOFF*"))):
    b=os.path.basename(d); sm=re.search(r"_s(\d+)$",b)
    if not sm: continue
    body=re.sub(r"^T2b_","",b); body=re.sub(r"_fusion_aux_occOFF_s\d+$","",body)
    s,m=psm(body)
    if s is None: continue
    ck=os.path.join(d,f"model_fusion_aux_{s}_{m}.pt")
    if not os.path.isfile(ck): continue
    o=os.path.join(REPO,"results/tier4_HO2_noocc_to_occ",f"{s}_{m}_fusion_aux_nointent_s{sm.group(1)}")
    if has_eval(o): continue
    os.makedirs(o,exist_ok=True)
    jobs.append((f"4a_fusion_{s}_{m}_s{sm.group(1)}",o,ecmd("fusion_aux",s,m,ck,o,sm.group(1),False)))
# 4b rule_based dense
for s in ["3_dense","4_dense"]:
    for sd in ["42","123","456","789","999"]:
        o=os.path.join(REPO,"results/tier_3_full/tier3_dense",f"{s}_stem_right_rule_based_s{sd}")
        if has_eval(o): continue
        os.makedirs(o,exist_ok=True)
        jobs.append((f"4b_rulebased_{s}_s{sd}",o,ecmd("rule_based",s,"stem_right",None,o,sd,False)))
# 4c Tier-1 intent backfill
root=os.path.join(REPO,"results/tier_1_full")
for b in sorted(os.listdir(root)):
    d=os.path.join(root,b)
    if not os.path.isdir(d) or "_intent_" not in b or "_nointent_" in b: continue
    if has_eval(d): continue
    meth=next((m for m in sorted(METHODS,key=len,reverse=True) if f"_{m}_" in b),None)
    if meth is None or meth=="rule_based": continue
    sm=re.search(r"_s(\d+)$",b)
    if not sm: continue
    s,m=psm(b.split(f"_{meth}_")[0])
    if s is None: continue
    ck=os.path.join(d,f"model_{meth}_{s}_{m}.pt")
    if not os.path.isfile(ck): continue
    jobs.append((f"4c_{meth}_{s}_{m}_s{sm.group(1)}",d,ecmd(meth,s,m,ck,d,sm.group(1),True)))

mine = jobs[WID-1::NW]
print(f"[job4r] total remaining={len(jobs)}  this worker={len(mine)}  max_parallel={MAXP}", flush=True)

sem=threading.Semaphore(MAXP); lock=threading.Lock(); errs=[]; done=[0]
def run_one(label,out_dir,cmd):
    with sem:
        t0=time.time()
        with open(os.path.join(out_dir,"job4_eval.log"),"w") as lf:
            r=subprocess.run(cmd,cwd=REPO,stdout=lf,stderr=lf)
        with lock:
            done[0]+=1
            if r.returncode!=0: errs.append(f"rc={r.returncode}: {label}")
            print(f"[job4r] {'OK' if r.returncode==0 else 'FAIL'} {label} "
                  f"({(time.time()-t0)/60:.1f}m) [{done[0]}/{len(mine)}]", flush=True)
ths=[threading.Thread(target=run_one,args=j) for j in mine]
for t in ths: t.start()
for t in ths: t.join()
print(f"\n[job4r] worker {WID} done: {len(mine)-len(errs)}/{len(mine)} ok", flush=True)
if errs:
    print(f"[job4r] {len(errs)} failures:")
    for e in errs[:30]: print("  ",e)
PYEOF

echo "###### JOB4 RETRY worker=$WORKER_ID COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ######"
