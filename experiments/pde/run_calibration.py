"""Calibration runs to determine optimal total_steps for the full ablation.

Trains all 5 methods across multiple scenarios and seeds for an extended
number of steps. Then analyzes learning curves to find convergence points.

Usage:
    python experiments/pde/run_calibration.py --scenarios 1a 4_dense --seeds 42 123 456
    python experiments/pde/run_calibration.py --analyze_only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
import time
import glob
import numpy as np

METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]

METHOD_LABELS = {
    "hjb_aux": "Hard-HJB", "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux": "Eikonal", "cbf_aux": "CBF-PDE",
    "drppo": "DRPPO (baseline)",
}
METHOD_COLORS = {
    "hjb_aux": "#1f77b4", "soft_hjb_aux": "#ff7f0e",
    "eikonal_aux": "#2ca02c", "cbf_aux": "#d62728",
    "drppo": "#7f7f7f",
}


def run_calibration(scenarios, ego_maneuvers, seeds, total_steps,
                    out_dir, max_parallel=5):
    """Launch calibration runs: methods x scenarios x seeds."""
    os.makedirs(out_dir, exist_ok=True)
    assert len(scenarios) == len(ego_maneuvers), "scenarios and maneuvers must be parallel lists"

    all_procs = []
    for scenario, maneuver in zip(scenarios, ego_maneuvers):
        for seed in seeds:
            for method in METHODS:
                if method == "drppo":
                    script = "experiments/pde/train_drppo_baseline.py"
                else:
                    script = f"experiments/pde/train_{method}.py"

                method_dir = os.path.join(out_dir, f"{scenario}_{maneuver}_{method}_s{seed}")
                os.makedirs(method_dir, exist_ok=True)

                cmd = [
                    sys.executable, script,
                    "--scenario", scenario,
                    "--ego_maneuver", maneuver,
                    "--seed", str(seed),
                    "--total_steps", str(total_steps),
                    "--out_dir", method_dir,
                    "--log_interval_steps", "1000",
                ]

                log_path = os.path.join(method_dir, "calibration.log")
                while len([p for p, _ in all_procs if p.poll() is None]) >= max_parallel:
                    time.sleep(5)
                print(f"  [START] {scenario}/{maneuver} {method} seed={seed}")
                with open(log_path, "w") as log_f:
                    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
                all_procs.append((p, f"{scenario}_{maneuver}_{method}_s{seed}"))

    print(f"\nWaiting for {len(all_procs)} calibration runs...")
    t0 = time.time()
    n_ok = 0
    for p, tag in all_procs:
        p.wait()
        elapsed = time.time() - t0
        status = "OK" if p.returncode == 0 else f"FAIL (exit {p.returncode})"
        if p.returncode == 0:
            n_ok += 1
        print(f"  [{status}] {tag} ({elapsed / 60:.1f} min elapsed)")

    print(f"\n{n_ok}/{len(all_procs)} calibration runs completed successfully.")
    return n_ok == len(all_procs)


def analyze_calibration(out_dir, scenarios, ego_maneuvers):
    """Aggregate calibration results across seeds and plot convergence."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: pandas and matplotlib required for analysis.")
        return

    for scenario, maneuver in zip(scenarios, ego_maneuvers):
        scen_tag = f"{scenario}_{maneuver}"
        print(f"\n=== {scen_tag} ===")

        by_method = {}
        for method in METHODS:
            by_method[method] = []
            pattern = os.path.join(out_dir, f"{scenario}_{maneuver}_{method}_s*")
            for seed_dir in glob.glob(pattern):
                csv_files = glob.glob(os.path.join(seed_dir, "train_*.csv"))
                if csv_files:
                    try:
                        df = pd.read_csv(csv_files[0])
                        by_method[method].append(df)
                    except Exception as e:
                        print(f"  WARN: {csv_files[0]}: {e}")

        # Plot learning curves with mean +/- std across seeds
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, dfs in by_method.items():
            if not dfs:
                continue
            step_vals = dfs[0]["step"].values
            n_rows = min(len(df) for df in dfs)
            returns_matrix = np.array([df["episode_return"].values[:n_rows] for df in dfs])
            mean_r = returns_matrix.mean(axis=0)
            std_r = returns_matrix.std(axis=0)
            ax.plot(step_vals[:n_rows], mean_r,
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method, "black"), linewidth=2)
            ax.fill_between(step_vals[:n_rows], mean_r - std_r, mean_r + std_r, alpha=0.2,
                            color=METHOD_COLORS.get(method, "black"))
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Episode Return (mean +/- std across seeds)")
        ax.set_title(f"Calibration: {scen_tag}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f"calibration_{scen_tag}_return.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure: {scen_tag}")

        # Convergence analysis
        n_seeds = max(len(dfs) for dfs in by_method.values()) if by_method else 0
        print(f"\n  CONVERGENCE (across {n_seeds} seeds):")
        for method, dfs in by_method.items():
            if not dfs:
                continue
            conv_steps = []
            for df in dfs:
                returns = df["episode_return"].values
                steps = df["step"].values
                if len(returns) < 10:
                    continue
                final_mean = np.mean(returns[-max(5, len(returns) // 5):])
                for i in range(len(returns) - 1, 4, -1):
                    trailing_mean = np.mean(returns[max(0, i - 9):i + 1])
                    if final_mean != 0 and abs(trailing_mean - final_mean) / abs(final_mean) > 0.05:
                        conv_steps.append(steps[min(i + 1, len(steps) - 1)])
                        break
                else:
                    conv_steps.append(steps[0])
            if conv_steps:
                mean_conv = np.mean(conv_steps)
                std_conv = np.std(conv_steps)
                max_conv = np.max(conv_steps)
                print(f"    {method:20s}: conv_step mean={mean_conv:.0f} std={std_conv:.0f} max={max_conv:.0f}")

    print()
    print("RECOMMENDATION: Choose Tier 1 total_steps = max convergence step x 1.3 (30% margin).")
    print("Use the MAX across methods and scenarios as the safety bound.")


def main():
    parser = argparse.ArgumentParser(description="Calibration runs for step count determination")
    parser.add_argument("--scenarios", nargs="+", default=["1a", "4_dense"])
    parser.add_argument("--ego_maneuvers", nargs="+", default=["stem_right", "stem_right"],
                        help="Maneuver per scenario (must match length of --scenarios)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--out_dir", default="results/calibration")
    parser.add_argument("--max_parallel", type=int, default=5)
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip training, only analyze existing results")
    args = parser.parse_args()

    if not args.analyze_only:
        total_runs = len(METHODS) * len(args.scenarios) * len(args.seeds)
        print(f"Starting calibration: {len(METHODS)} methods x {len(args.scenarios)} scenarios x {len(args.seeds)} seeds x {args.steps} steps")
        print(f"Total: {total_runs} runs")
        success = run_calibration(args.scenarios, args.ego_maneuvers, args.seeds,
                                  args.steps, args.out_dir, args.max_parallel)
        if not success:
            print("\nWARNING: Some calibration runs failed. Check logs.")

    print("\nAnalyzing calibration results...")
    analyze_calibration(args.out_dir, args.scenarios, args.ego_maneuvers)


if __name__ == "__main__":
    main()
