"""Phase 3F Step 11 — full 36-job re-calibration orchestrator.

Mirrors the parallel-launch primitive of experiments/pde/run_calibration.py but
constructs jobs explicitly with (scenario, maneuver) pairs:
  - (1a, stem_right) × 18 jobs (6 methods × 3 seeds)
  - (2_dense, right_left) × 18 jobs

Outputs go to results/step11_calibration/STEP11_<method>_<scen>_<man>_s<seed>/
to avoid colliding with Phase 3 results at results/calibration/.

Post-run: appends criterion_version="v2_sr_primary_post_stage3" to each
meta.json (Step 11 spec §2.5).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/boulosboulos/Desktop/EECE_499-main")
sys.path.insert(0, str(PROJECT_ROOT))

CRITERION_VERSION = "v2_sr_primary_post_stage3"

METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]
CELL_PAIRS = [("1a", "stem_right"), ("2_dense", "right_left")]
SEEDS = [42, 123, 456]
TOTAL_STEPS = 500_000
OUTPUT_ROOT = PROJECT_ROOT / "results/step11_calibration"
PARALLEL = 15  # matches Phase 3 (CALIBRATION_PARALLEL); within spec's 22-24-SUMO-envs target


def train_script(method: str) -> str:
    if method == "drppo":
        return str(PROJECT_ROOT / "experiments/pde/train_drppo_baseline.py")
    return str(PROJECT_ROOT / f"experiments/pde/train_{method}.py")


def make_job(method: str, scenario: str, maneuver: str, seed: int) -> dict:
    run_id = f"STEP11_{method}_{scenario}_{maneuver}_s{seed}"
    out_dir = OUTPUT_ROOT / run_id
    cmd = [
        sys.executable, train_script(method),
        "--scenario", scenario,
        "--ego_maneuver", maneuver,
        "--seed", str(seed),
        "--total_steps", str(TOTAL_STEPS),
        "--out_dir", str(out_dir),
    ]
    return {
        "run_id": run_id, "method": method,
        "scenario": scenario, "maneuver": maneuver, "seed": seed,
        "out_dir": str(out_dir), "cmd_train": cmd,
    }


def generate_all_jobs() -> list[dict]:
    jobs = []
    for method in METHODS:
        for (scen, man) in CELL_PAIRS:
            for seed in SEEDS:
                jobs.append(make_job(method, scen, man, seed))
    return jobs


def launch(jobs: list[dict], parallel: int) -> dict:
    active = []
    completed = 0
    failed = 0
    failed_runs = []
    start = time.time()
    n_total = len(jobs)
    print(f"[step11] Launching {n_total} jobs at parallel={parallel}…", flush=True)

    for job in jobs:
        while len(active) >= parallel:
            time.sleep(5)
            still_active = []
            for proc, j, log_f in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, j, log_f))
                else:
                    log_f.close()
                    elapsed_min = (time.time() - start) / 60.0
                    if ret == 0:
                        completed += 1
                        print(f"  [OK]   {j['run_id']} ({completed + failed}/{n_total}, {elapsed_min:.1f}m)", flush=True)
                    else:
                        failed += 1
                        failed_runs.append(j["run_id"])
                        print(f"  [FAIL] {j['run_id']} (exit {ret})", flush=True)
            active = still_active

        os.makedirs(job["out_dir"], exist_ok=True)
        log_path = os.path.join(job["out_dir"], "stdout.log")
        log_f = open(log_path, "w")
        proc = subprocess.Popen(job["cmd_train"], stdout=log_f, stderr=subprocess.STDOUT)
        active.append((proc, job, log_f))
        print(f"  [START] {job['run_id']} (pid {proc.pid})", flush=True)

    for proc, j, log_f in active:
        proc.wait()
        log_f.close()
        if proc.returncode == 0:
            completed += 1
            print(f"  [OK-drain] {j['run_id']}", flush=True)
        else:
            failed += 1
            failed_runs.append(j["run_id"])
            print(f"  [FAIL] {j['run_id']} (exit {proc.returncode})", flush=True)

    elapsed = time.time() - start
    print(f"\n[step11] Done: {completed} completed, {failed} failed, {elapsed/3600:.2f}h total", flush=True)
    return {"completed": completed, "failed": failed, "failed_runs": failed_runs,
            "wall_time_h": elapsed / 3600.0}


def append_criterion_version_to_metas(jobs: list[dict]) -> int:
    """Spec §2.5: meta.json must include criterion_version. Append post-run."""
    n_updated = 0
    for j in jobs:
        meta_path = Path(j["out_dir"]) / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["criterion_version"] = CRITERION_VERSION
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            n_updated += 1
        except Exception as e:
            print(f"[step11] WARN: failed to update {meta_path}: {e}")
    return n_updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=PARALLEL)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    jobs = generate_all_jobs()
    print(f"[step11] {len(jobs)} jobs, parallel={args.parallel}, output_root={OUTPUT_ROOT}", flush=True)
    for j in jobs:
        print(f"  {j['run_id']}", flush=True)
    if args.dry_run:
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = launch(jobs, args.parallel)
    n_meta = append_criterion_version_to_metas(jobs)
    print(f"[step11] Annotated criterion_version on {n_meta}/{len(jobs)} meta.json files", flush=True)

    # Write a launch summary
    with open(OUTPUT_ROOT / "step11_launch_summary.json", "w") as f:
        json.dump({
            "criterion_version": CRITERION_VERSION,
            "parallel": args.parallel,
            "n_jobs": len(jobs),
            "summary": summary,
            "n_meta_annotated": n_meta,
        }, f, indent=2)


if __name__ == "__main__":
    main()
