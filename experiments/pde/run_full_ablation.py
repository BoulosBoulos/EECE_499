"""Launch the full ablation grid as parallel subprocesses.

Usage:
    python experiments/pde/run_full_ablation.py --tier 1 --max_parallel 32
    python experiments/pde/run_full_ablation.py --tier 2 --max_parallel 16
    python experiments/pde/run_full_ablation.py --tier all --dry_run
"""

from __future__ import annotations
import argparse
import subprocess
import time
import os
from itertools import product

# ── Tier 1: Main comparison ─────────────────────────────────────────────
TIER1_COMBOS = [
    ("1a", "stem_right"), ("1a", "stem_left"), ("1a", "right_stem"),
    ("1b", "stem_right"), ("1b", "stem_left"),
    ("1c", "stem_right"),
    ("2", "stem_right"), ("2", "stem_left"),
    ("4", "stem_right"), ("4", "stem_left"),
    ("3", "right_left"), ("3", "left_right"),
]
TIER1_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]
TIER1_SEEDS = [42, 123, 456, 789, 999]
TIER1_INTENTS = [False, True]

# ── Tier 2: Lambda sensitivity ──────────────────────────────────────────
TIER2_COMBOS = [("1a", "stem_right")]
TIER2_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux"]
TIER2_LAMBDAS = [0.05, 0.1, 0.2, 0.5]
TIER2_SEEDS = [42, 123, 456]

# ── Tier 3: Extended scenarios ──────────────────────────────────────────
ALL_SCENARIOS = ["1a", "1b", "1c", "1d", "2", "3", "4"]
ALL_MANEUVERS = ["stem_right", "stem_left", "right_left", "right_stem",
                 "left_right", "left_stem"]
TIER3_SEEDS = [42, 123, 456]


def _build_train_cmd(method, scenario, maneuver, seed, out_dir,
                     total_steps=50000, use_intent=False, lambda_aux=None):
    """Build the command list for a single training run."""
    if method == "drppo":
        script = "experiments/pde/train_drppo_baseline.py"
    else:
        script = f"experiments/pde/train_{method}.py"
    cmd = [
        "python3", script,
        "--scenario", scenario,
        "--ego_maneuver", maneuver,
        "--seed", str(seed),
        "--out_dir", out_dir,
        "--total_steps", str(total_steps),
    ]
    if use_intent:
        cmd.append("--use_intent")
    if lambda_aux is not None and method != "drppo":
        cmd.extend(["--lambda_aux", str(lambda_aux)])
    return cmd


def generate_jobs(tier: str, total_steps: int = 50000) -> list[dict]:
    """Generate all jobs for the given tier."""
    jobs = []
    base_dir = "results/ablation"

    if tier in ("1", "all"):
        for (scen, man), method, seed, intent in product(
            TIER1_COMBOS, TIER1_METHODS, TIER1_SEEDS, TIER1_INTENTS
        ):
            intent_tag = "intent" if intent else "nointent"
            out_dir = os.path.join(base_dir, "tier1", f"{scen}_{man}_{method}_{intent_tag}_s{seed}")
            jobs.append({
                "cmd": _build_train_cmd(method, scen, man, seed, out_dir,
                                        total_steps, use_intent=intent),
                "tag": f"T1_{scen}_{man}_{method}_{intent_tag}_s{seed}",
                "tier": 1,
            })

    if tier in ("2", "all"):
        for (scen, man), method, lam, seed in product(
            TIER2_COMBOS, TIER2_METHODS, TIER2_LAMBDAS, TIER2_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier2_lambda", f"{scen}_{man}_{method}_l{lam}_s{seed}")
            jobs.append({
                "cmd": _build_train_cmd(method, scen, man, seed, out_dir,
                                        total_steps, lambda_aux=lam),
                "tag": f"T2L_{scen}_{method}_l{lam}_s{seed}",
                "tier": 2,
            })

    if tier in ("3", "all"):
        best_method = "hjb_aux"  # update after Tier 1 results
        for scen, man, seed in product(ALL_SCENARIOS, ALL_MANEUVERS, TIER3_SEEDS):
            out_dir = os.path.join(base_dir, "tier3_full", f"{scen}_{man}_{best_method}_s{seed}")
            jobs.append({
                "cmd": _build_train_cmd(best_method, scen, man, seed, out_dir, total_steps),
                "tag": f"T3_{scen}_{man}_s{seed}",
                "tier": 3,
            })

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Launch parallel ablation jobs")
    parser.add_argument("--tier", default="1", choices=["1", "2", "3", "all"])
    parser.add_argument("--max_parallel", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--dry_run", action="store_true", help="Print jobs without running")
    args = parser.parse_args()

    jobs = generate_jobs(args.tier, args.total_steps)
    print(f"Generated {len(jobs)} jobs for tier '{args.tier}'")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['tag']}: {' '.join(j['cmd'])}")
        return

    active = []
    completed = 0
    failed = 0
    start = time.time()

    for job in jobs:
        # Wait for a slot
        while len(active) >= args.max_parallel:
            time.sleep(5)
            still_active = []
            for proc, tag in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, tag))
                elif ret == 0:
                    completed += 1
                    elapsed = time.time() - start
                    print(f"  [OK] {tag} ({completed}/{len(jobs)}, {elapsed/60:.0f}m elapsed)")
                else:
                    failed += 1
                    print(f"  [FAIL] {tag} (exit {ret})")
            active = still_active

        # Launch
        out_dir_idx = job["cmd"].index("--out_dir") + 1
        job_out_dir = job["cmd"][out_dir_idx]
        os.makedirs(job_out_dir, exist_ok=True)
        log_path = os.path.join(job_out_dir, "stdout.log")
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(job["cmd"], stdout=log_f, stderr=subprocess.STDOUT)
        active.append((proc, job["tag"]))
        print(f"  [START] {job['tag']} (pid {proc.pid})")

    # Wait for remaining
    for proc, tag in active:
        proc.wait()
        if proc.returncode == 0:
            completed += 1
        else:
            failed += 1
            print(f"  [FAIL] {tag} (exit {proc.returncode})")

    elapsed = time.time() - start
    print(f"\nDone: {completed} completed, {failed} failed, {elapsed/3600:.1f}h total")


if __name__ == "__main__":
    main()
