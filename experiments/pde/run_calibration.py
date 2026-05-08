"""Calibration orchestrator (Phase 3 / SPEC_PHASE_3_CALIBRATION).

Runs the 36-run calibration grid:
  6 methods x 2 scenarios x 3 seeds at 500,000 steps each, 15-parallel on GPU.

Writes results/calibration/CAL_<method>_<scenario>_s<seed>/{metrics.csv, meta.json, ...}.

Sub-modes:
  --stress_test     : 15-job parallelism stress test at 5,000 steps (Step 3)
  --total_steps N   : override the per-job training step count (default 500_000)
  --parallel N      : override max_parallel (default 15)
  --dry_run         : print the job grid without launching
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

CALIBRATION_METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]
CALIBRATION_SCENARIOS = ["1a", "2_dense"]
CALIBRATION_MANEUVER = "stem_right"
CALIBRATION_SEEDS = [42, 123, 456]
CALIBRATION_TOTAL_STEPS = 500_000
CALIBRATION_TOTAL_STEPS_EXTENDED = 1_000_000
CALIBRATION_PARALLEL = 15

# Stress test grid (Decision F): 15 jobs spanning methods × scenarios × seeds.
# DRPPO × {1a, 2_dense} × {42, 123, 456} = 6
# HJB × {1a, 2_dense} × {42, 123} = 4
# Soft-HJB × {1a, 2_dense} × {42, 123} = 4
# CBF × 1a × 42 = 1
# Total = 15.
STRESS_TEST_GRID: list[tuple[str, str, int]] = (
    [("drppo", scen, seed) for scen in ("1a", "2_dense") for seed in (42, 123, 456)]
    + [("hjb_aux", scen, seed) for scen in ("1a", "2_dense") for seed in (42, 123)]
    + [("soft_hjb_aux", scen, seed) for scen in ("1a", "2_dense") for seed in (42, 123)]
    + [("cbf_aux", "1a", 42)]
)


def _train_script(method: str) -> str:
    if method == "drppo":
        return "experiments/pde/train_drppo_baseline.py"
    return f"experiments/pde/train_{method}.py"


def _build_train_cmd(method: str, scenario: str, seed: int, total_steps: int, out_dir: str) -> list[str]:
    return [
        sys.executable, _train_script(method),
        "--scenario", scenario,
        "--ego_maneuver", CALIBRATION_MANEUVER,
        "--seed", str(seed),
        "--total_steps", str(total_steps),
        "--out_dir", out_dir,
    ]


def _make_job(method: str, scenario: str, seed: int, total_steps: int, output_root: str,
              run_id_prefix: str = "CAL_") -> dict:
    run_id = f"{run_id_prefix}{method}_{scenario}_s{seed}"
    out_dir = os.path.join(output_root, run_id)
    return {
        "run_id": run_id,
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "out_dir": out_dir,
        "cmd_train": _build_train_cmd(method, scenario, seed, total_steps, out_dir),
    }


def generate_jobs(total_steps: int, output_root: str) -> list[dict]:
    jobs = []
    for method in CALIBRATION_METHODS:
        for scenario in CALIBRATION_SCENARIOS:
            for seed in CALIBRATION_SEEDS:
                jobs.append(_make_job(method, scenario, seed, total_steps, output_root))
    return jobs


def generate_stress_test_jobs(total_steps: int, output_root: str) -> list[dict]:
    return [
        _make_job(method, scen, seed, total_steps, output_root, run_id_prefix="STRESS_")
        for (method, scen, seed) in STRESS_TEST_GRID
    ]


def _filter_jobs(jobs, methods, scenarios, seeds):
    out = []
    for j in jobs:
        if methods and j["method"] not in methods: continue
        if scenarios and j["scenario"] not in scenarios: continue
        if seeds and j["seed"] not in seeds: continue
        out.append(j)
    return out


def _launch(jobs: list[dict], parallel: int) -> dict:
    """Run jobs in parallel (max `parallel`); return summary dict."""
    active: list[tuple[subprocess.Popen, dict, "object"]] = []
    completed = 0
    failed = 0
    failed_runs: list[str] = []
    start = time.time()
    n_total = len(jobs)
    print(f"[calibration] Launching {n_total} jobs at parallel={parallel}…")

    for job in jobs:
        # Wait for an open slot.
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
                        print(f"  [OK]   {j['run_id']} ({completed + failed}/{n_total}, {elapsed_min:.1f}m)")
                    else:
                        failed += 1
                        failed_runs.append(j["run_id"])
                        print(f"  [FAIL] {j['run_id']} (exit {ret})")
            active = still_active

        os.makedirs(job["out_dir"], exist_ok=True)
        log_path = os.path.join(job["out_dir"], "stdout.log")
        log_f = open(log_path, "w")
        proc = subprocess.Popen(job["cmd_train"], stdout=log_f, stderr=subprocess.STDOUT)
        active.append((proc, job, log_f))
        print(f"  [START] {job['run_id']} (pid {proc.pid})")

    # Drain the rest.
    for proc, j, log_f in active:
        proc.wait()
        log_f.close()
        if proc.returncode == 0:
            completed += 1
        else:
            failed += 1
            failed_runs.append(j["run_id"])
            print(f"  [FAIL] {j['run_id']} (exit {proc.returncode})")

    elapsed = time.time() - start
    print()
    print(f"[calibration] Done: {completed} completed, {failed} failed, {elapsed/3600:.2f}h total")
    return {
        "n_total": n_total,
        "completed": completed,
        "failed": failed,
        "failed_runs": failed_runs,
        "wall_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 calibration orchestrator")
    parser.add_argument("--output_root", default="results/calibration",
                        help="Root output directory (default results/calibration)")
    parser.add_argument("--total_steps", type=int, default=CALIBRATION_TOTAL_STEPS,
                        help="Per-job step count (default 500_000)")
    parser.add_argument("--parallel", type=int, default=CALIBRATION_PARALLEL,
                        help="Max parallel jobs (default 15)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print job grid without launching")
    parser.add_argument("--stress_test", action="store_true",
                        help="Run 15-job parallelism stress test at small step count")
    parser.add_argument("--stress_steps", type=int, default=5000,
                        help="Steps per job in stress test mode (default 5000)")
    # Filter flags (subset of run_full_ablation.py's filters), useful for
    # contingency re-runs in Step 6.
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.stress_test:
        jobs = generate_stress_test_jobs(args.stress_steps, args.output_root)
        mode_label = f"STRESS-TEST ({args.stress_steps} steps/job)"
    else:
        jobs = generate_jobs(args.total_steps, args.output_root)
        mode_label = f"FULL CALIBRATION ({args.total_steps} steps/job)"

    jobs = _filter_jobs(jobs, args.methods, args.scenarios, args.seeds)

    print(f"[calibration] Mode: {mode_label}")
    print(f"[calibration] {len(jobs)} jobs, parallel={args.parallel}, output_root={args.output_root}")

    if args.dry_run:
        for j in jobs[:5]:
            print(f"  {j['run_id']}:")
            print(f"    cmd: {' '.join(j['cmd_train'])}")
        if len(jobs) > 5:
            print(f"  … and {len(jobs) - 5} more")
        return 0

    # Lock check (warn if YAML changed since lock was generated).
    try:
        from config_loader import check_config_lock
        lock_status = check_config_lock(strict=False)
        if not lock_status["matches"] and lock_status.get("warning"):
            print(lock_status["warning"])
    except Exception:
        pass

    summary = _launch(jobs, args.parallel)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
