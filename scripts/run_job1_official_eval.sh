#!/bin/bash
# Job 1 (corrected): use the OFFICIAL eval.py path to adjudicate the Tier-3
# behavioral-robustness SR=0 anomaly on 1b/stem_right. The bespoke replay
# harness (job1_diagnostic.py) failed its own positive control, so it cannot
# be trusted. This reruns the SAME nominal-trained checkpoints through the
# exact eval.py that produced the tables.
#
# Conditions (drppo + hjb_aux, nominal-trained 1b/stem_right, seed 42):
#   A) --style_filter nominal   (should reproduce tier3 table's ~0.00 SR)
#   B) (no --style_filter)      (T1 setting; tier1 scored ~0.22 SR)
# Plus 1a/stem_right nominal as the positive control (table says ~0.99).
#
# Decisive logic:
#   If A≈0.00 AND B≈0.22 AND 1a-control≈0.99  -> tier3 0% is REAL (nominal harder);
#                                                 bespoke replay was simply broken.
#   If A≈0.99                                  -> tier3 table itself is wrong.
#
# All outputs go to results/diagnostics/job1_official/ (never the real run dirs).

#SBATCH --job-name=job1_offeval
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job1_offeval_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job1_offeval_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=0-03:00:00

set -uo pipefail
REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export REPO

echo "############ JOB 1 OFFICIAL-EVAL DIAGNOSTIC $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) ############"

python3 - <<'PYEOF'
import os, sys, subprocess, threading, time
REPO = os.environ["REPO"]
T3 = os.path.join(REPO, "results/tier_3_full/tier3_behav")
OUT = os.path.join(REPO, "results/diagnostics/job1_official")
os.makedirs(OUT, exist_ok=True)

# (label, method, scenario, maneuver, checkpoint, style_filter)
CKPT = lambda scen,man,meth: os.path.join(
    T3, f"{scen}_{man}_{meth}_nominal_s42", f"model_{meth}_{scen}_{man}.pt")
CONDS = [
    ("1b_drppo_nominal",   "drppo",   "1b", "stem_right", CKPT("1b","stem_right","drppo"),   "nominal"),
    ("1b_drppo_nofilter",  "drppo",   "1b", "stem_right", CKPT("1b","stem_right","drppo"),   None),
    ("1b_hjb_nominal",     "hjb_aux", "1b", "stem_right", CKPT("1b","stem_right","hjb_aux"), "nominal"),
    ("1b_hjb_nofilter",    "hjb_aux", "1b", "stem_right", CKPT("1b","stem_right","hjb_aux"), None),
    ("1a_drppo_nominal",   "drppo",   "1a", "stem_right",
        os.path.join(T3, "1a_stem_right_drppo_nominal_s123", "model_drppo_1a_stem_right.pt"), "nominal"),
]
EVAL_SEEDS = ["1042", "2042", "3042"]   # same style as tier3 (3 x 200 = 600 eps)
EPISODES = "200"

def build_cmd(label, method, scen, man, ckpt, style):
    od = os.path.join(OUT, label)
    os.makedirs(od, exist_ok=True)
    cmd = ["python3", "experiments/pde/eval.py",
           "--method", method, "--checkpoint", ckpt,
           "--scenario", scen, "--ego_maneuver", man,
           "--episodes", EPISODES, "--seeds", *EVAL_SEEDS,
           "--out_dir", od]
    if style is not None:
        cmd += ["--style_filter", style]
    return od, cmd

def run(cond):
    label, method, scen, man, ckpt, style = cond
    if not os.path.isfile(ckpt):
        print(f"[job1] MISSING CKPT {label}: {ckpt}", flush=True); return
    od, cmd = build_cmd(*cond)
    t0 = time.time()
    print(f"[job1] START {label} style={style}", flush=True)
    with open(os.path.join(od, "eval_run.log"), "w") as lf:
        r = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=lf)
    print(f"[job1] {'DONE' if r.returncode==0 else 'FAIL rc='+str(r.returncode)} "
          f"{label} ({(time.time()-t0)/60:.1f}m)", flush=True)

threads = [threading.Thread(target=run, args=(c,)) for c in CONDS]
for t in threads: t.start()
for t in threads: t.join()

# Report SR/CR per condition from the episode-level eval_metrics.csv
import csv
print("\n" + "="*70 + "\nJOB 1 OFFICIAL-EVAL VERDICT\n" + "="*70)
for label, *_ in [(c[0],) for c in CONDS]:
    p = os.path.join(OUT, label, "eval_metrics.csv")
    if not os.path.isfile(p):
        print(f"  {label:22s}: NO OUTPUT"); continue
    with open(p) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        print(f"  {label:22s}: EMPTY"); continue
    ts = [r.get("terminal_state","").lower() for r in rows]
    sr = sum(t=="success" for t in ts)/n
    cr = sum(t=="collision" for t in ts)/n
    to = sum(t=="timeout" for t in ts)/n
    print(f"  {label:22s}: SR={sr:.3f}  CR={cr:.3f}  TO={to:.3f}  (n={n})")
PYEOF

echo "############ JOB 1 OFFICIAL-EVAL COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ############"
