"""Launch the full ablation grid as parallel subprocesses.

Tiers:
    1: Main comparison (600 runs) — 12 combos x 5 methods x 5 seeds x 2 intent
    2: Lambda sensitivity (48) + Occlusion ablation (60) = 108 runs
    3: State ablation (60) + Behavioral robustness (60) + Dense scenarios (45) = 165 runs
    supp: Full 42-combo table with best method (126 runs)
    all: Tiers 1 + 2 + 3 (873 runs, excludes supp)

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

# ── Tier 2: Occlusion ablation ──────────────────────────────────────────
TIER2_OCC_COMBOS = [
    ("1a", "stem_right"), ("1b", "stem_right"),
    ("2", "stem_right"), ("4", "stem_right"),
]
TIER2_OCC_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]
TIER2_OCC_SEEDS = [42, 123, 456]

# ── Tier 3: State ablation ──────────────────────────────────────────────
TIER3_STATE_COMBOS = [
    ("1a", "stem_right"), ("1b", "stem_right"),
    ("2", "stem_right"), ("4", "stem_right"),
]
TIER3_STATE_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]
TIER3_STATE_SEEDS = [42, 123, 456]

# ── Tier 3: Behavioral robustness ───────────────────────────────────────
TIER3_BEHAV_COMBOS = [
    ("1a", "stem_right"), ("1b", "stem_right"),
    ("2", "stem_right"), ("4", "stem_right"),
]
TIER3_BEHAV_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]
TIER3_BEHAV_SEEDS = [42, 123, 456]

# ── Tier 3: Dense scenarios ─────────────────────────────────────────────
TIER3_DENSE_SCENARIOS = ["2_dense", "3_dense", "4_dense"]
TIER3_DENSE_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]
TIER3_DENSE_SEEDS = [42, 123, 456]

# ── Supplementary: Full 42-combo table ──────────────────────────────────
ALL_SCENARIOS = ["1a", "1b", "1c", "1d", "2", "3", "4"]
ALL_MANEUVERS = ["stem_right", "stem_left", "right_left", "right_stem",
                 "left_right", "left_stem"]
SUPP_SEEDS = [42, 123, 456]


def _build_train_cmd(method, scenario, maneuver, seed, out_dir,
                     total_steps=50000, use_intent=False, lambda_aux=None,
                     no_buildings=False, style_filter=None, state_ablation=None):
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
    if no_buildings:
        cmd.append("--no_buildings")
    if style_filter is not None:
        cmd.extend(["--style_filter", style_filter])
    if state_ablation is not None:
        cmd.extend(["--state_ablation", state_ablation])
    return cmd


def _build_eval_cmd(method, scenario, maneuver, train_seed, out_dir,
                    n_eval_episodes=100, no_buildings=False,
                    style_filter=None, state_ablation=None):
    """Build eval command for a trained checkpoint."""
    ckpt_path = os.path.join(out_dir, f"model_{method}_{scenario}_{maneuver}.pt")
    eval_seeds = [train_seed + 1000, train_seed + 2000, train_seed + 3000]
    cmd = [
        "python3", "experiments/pde/eval.py",
        "--method", method,
        "--checkpoint", ckpt_path,
        "--scenario", scenario,
        "--ego_maneuver", maneuver,
        "--episodes", str(n_eval_episodes),
        "--seeds", *[str(s) for s in eval_seeds],
        "--out_dir", out_dir,
        "--save_failures", "--max_failures", "5",
    ]
    if no_buildings:
        cmd.append("--no_buildings")
    if style_filter:
        cmd.extend(["--style_filter", style_filter])
    if state_ablation:
        cmd.extend(["--state_ablation", state_ablation])
    return cmd


def generate_jobs(tier: str, total_steps: int = 50000) -> list[dict]:
    """Generate all jobs for the given tier."""
    jobs = []
    base_dir = "results/ablation"

    # ── TIER 1: Main comparison ─────────────────────────────────────────
    if tier in ("1", "all"):
        for (scen, man), method, seed, intent in product(
            TIER1_COMBOS, TIER1_METHODS, TIER1_SEEDS, TIER1_INTENTS
        ):
            intent_tag = "intent" if intent else "nointent"
            out_dir = os.path.join(base_dir, "tier1",
                                   f"{scen}_{man}_{method}_{intent_tag}_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, man, seed, out_dir,
                                              total_steps, use_intent=intent),
                "cmd_eval": _build_eval_cmd(method, scen, man, seed, out_dir),
                "tag": f"T1_{scen}_{man}_{method}_{intent_tag}_s{seed}",
                "tier": 1,
            })

    # ── TIER 2: Lambda sensitivity + Occlusion ablation ─────────────────
    if tier in ("2", "all"):
        # Lambda sweep
        for (scen, man), method, lam, seed in product(
            TIER2_COMBOS, TIER2_METHODS, TIER2_LAMBDAS, TIER2_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier2_lambda",
                                   f"{scen}_{man}_{method}_l{lam}_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, man, seed, out_dir,
                                              total_steps, lambda_aux=lam),
                "cmd_eval": _build_eval_cmd(method, scen, man, seed, out_dir),
                "tag": f"T2L_{scen}_{method}_l{lam}_s{seed}",
                "tier": 2,
            })

        # Occlusion ablation (buildings removed)
        for (scen, man), method, seed in product(
            TIER2_OCC_COMBOS, TIER2_OCC_METHODS, TIER2_OCC_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier2_noocc",
                                   f"{scen}_{man}_{method}_noocc_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, man, seed, out_dir,
                                              total_steps, no_buildings=True),
                "cmd_eval": _build_eval_cmd(method, scen, man, seed, out_dir,
                                            no_buildings=True),
                "tag": f"T2O_{scen}_{method}_noocc_s{seed}",
                "tier": 2,
            })

    # ── TIER 3: State ablation, behavioral robustness, dense ────────────
    if tier in ("3", "all"):
        # State ablation
        for (scen, man), method, seed in product(
            TIER3_STATE_COMBOS, TIER3_STATE_METHODS, TIER3_STATE_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier3_state",
                                   f"{scen}_{man}_{method}_novis_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, man, seed, out_dir,
                                              total_steps, state_ablation="no_visibility"),
                "cmd_eval": _build_eval_cmd(method, scen, man, seed, out_dir,
                                            state_ablation="no_visibility"),
                "tag": f"T3S_{scen}_{method}_novis_s{seed}",
                "tier": 3,
            })

        # Behavioral robustness (train on nominal styles)
        for (scen, man), method, seed in product(
            TIER3_BEHAV_COMBOS, TIER3_BEHAV_METHODS, TIER3_BEHAV_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier3_behav",
                                   f"{scen}_{man}_{method}_nominal_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, man, seed, out_dir,
                                              total_steps, style_filter="nominal"),
                "cmd_eval": _build_eval_cmd(method, scen, man, seed, out_dir,
                                            style_filter="nominal"),
                "tag": f"T3B_{scen}_{method}_nominal_s{seed}",
                "tier": 3,
            })

        # Dense scenarios
        for scen, method, seed in product(
            TIER3_DENSE_SCENARIOS, TIER3_DENSE_METHODS, TIER3_DENSE_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier3_dense",
                                   f"{scen}_stem_right_{method}_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, "stem_right", seed,
                                              out_dir, total_steps),
                "cmd_eval": _build_eval_cmd(method, scen, "stem_right", seed, out_dir),
                "tag": f"T3D_{scen}_{method}_s{seed}",
                "tier": 3,
            })

    # ── SUPPLEMENTARY: Full 42-combo table ──────────────────────────────
    if tier == "supp":
        best_method = "hjb_aux"  # PLACEHOLDER — update based on Tier 1 results
        for scen, man, seed in product(ALL_SCENARIOS, ALL_MANEUVERS, SUPP_SEEDS):
            out_dir = os.path.join(base_dir, "supplementary",
                                   f"{scen}_{man}_{best_method}_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(best_method, scen, man, seed,
                                              out_dir, total_steps),
                "cmd_eval": _build_eval_cmd(best_method, scen, man, seed, out_dir),
                "tag": f"SUP_{scen}_{man}_s{seed}",
                "tier": "supp",
            })

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Launch parallel ablation jobs")
    parser.add_argument("--tier", default="1", choices=["1", "2", "3", "all", "supp"])
    parser.add_argument("--max_parallel", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--dry_run", action="store_true", help="Print jobs without running")
    args = parser.parse_args()

    jobs = generate_jobs(args.tier, args.total_steps)

    # Print tier breakdown
    tier_counts = {}
    for j in jobs:
        t = j["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"Generated {len(jobs)} jobs for tier '{args.tier}':")
    for t, c in sorted(tier_counts.items(), key=lambda x: str(x[0])):
        print(f"  Tier {t}: {c} jobs")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['tag']}:")
            print(f"    TRAIN: {' '.join(j['cmd_train'])}")
            print(f"    EVAL:  {' '.join(j['cmd_eval'])}")
        return

    active = []
    completed = 0
    failed = 0
    start = time.time()

    for job in jobs:
        while len(active) >= args.max_parallel:
            time.sleep(5)
            still_active = []
            for proc, tag, log_f in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, tag, log_f))
                elif ret == 0:
                    log_f.close()
                    completed += 1
                    elapsed = time.time() - start
                    print(f"  [OK] {tag} ({completed}/{len(jobs)}, {elapsed/60:.0f}m elapsed)")
                else:
                    log_f.close()
                    failed += 1
                    print(f"  [FAIL] {tag} (exit {ret})")
            active = still_active

        out_dir_idx = job["cmd_train"].index("--out_dir") + 1
        job_out_dir = job["cmd_train"][out_dir_idx]
        os.makedirs(job_out_dir, exist_ok=True)
        log_path = os.path.join(job_out_dir, "stdout.log")
        # Chain train then eval in one shell command
        train_str = " ".join(f"'{c}'" for c in job["cmd_train"])
        eval_str = " ".join(f"'{c}'" for c in job["cmd_eval"])
        shell_cmd = f"({train_str} && echo '=== EVAL ===' && {eval_str})"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(shell_cmd, stdout=log_f, stderr=subprocess.STDOUT, shell=True)
        active.append((proc, job["tag"], log_f))
        print(f"  [START] {job['tag']} (pid {proc.pid})")

    for proc, tag, log_f in active:
        proc.wait()
        log_f.close()
        if proc.returncode == 0:
            completed += 1
        else:
            failed += 1
            print(f"  [FAIL] {tag} (exit {proc.returncode})")

    elapsed = time.time() - start
    print(f"\nDone: {completed} completed, {failed} failed, {elapsed/3600:.1f}h total")


if __name__ == "__main__":
    main()
