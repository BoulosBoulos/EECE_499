"""Calibration runs to determine optimal total_steps for the full ablation.

Trains all 5 methods on one (scenario, maneuver) combo for an extended
number of steps (default 200k). Then analyzes learning curves to find
the convergence point.

Usage:
    python experiments/pde/run_calibration.py --scenario 1a --ego_maneuver stem_right --steps 200000
    python experiments/pde/run_calibration.py --scenario 1a --ego_maneuver stem_right --analyze_only
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


def run_calibration(scenario: str, maneuver: str, total_steps: int,
                    seed: int, out_dir: str, max_parallel: int = 5):
    """Launch calibration training runs for all 5 methods in parallel."""
    os.makedirs(out_dir, exist_ok=True)

    procs = []
    for method in METHODS:
        if method == "drppo":
            script = "experiments/pde/train_drppo_baseline.py"
        else:
            script = f"experiments/pde/train_{method}.py"

        method_dir = os.path.join(out_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        cmd = [
            sys.executable, script,
            "--scenario", scenario,
            "--ego_maneuver", maneuver,
            "--seed", str(seed),
            "--total_steps", str(total_steps),
            "--out_dir", method_dir,
        ]

        log_path = os.path.join(method_dir, "calibration.log")
        print(f"  [START] {method}: {' '.join(cmd)}")
        with open(log_path, "w") as log_f:
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((p, method))

    print(f"\nWaiting for {len(procs)} calibration runs...")
    t0 = time.time()
    for p, method in procs:
        p.wait()
        elapsed = time.time() - t0
        status = "OK" if p.returncode == 0 else f"FAIL (exit {p.returncode})"
        print(f"  [{status}] {method} ({elapsed / 60:.1f} min elapsed)")

    n_ok = sum(1 for p, _ in procs if p.returncode == 0)
    print(f"\n{n_ok}/{len(METHODS)} calibration runs completed successfully.")
    return n_ok == len(METHODS)


def analyze_calibration(out_dir: str, scenario: str, maneuver: str):
    """Analyze calibration results and produce convergence diagnostic plots."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: pandas and matplotlib required for analysis.")
        print("Install with: pip install pandas matplotlib")
        return

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

    method_data = {}
    for method in METHODS:
        csv_pattern = os.path.join(out_dir, method, f"train_{method}_{scenario}_{maneuver}.csv")
        csvs = glob.glob(csv_pattern)
        if not csvs:
            csvs = glob.glob(os.path.join(out_dir, method, "train_*.csv"))
        if csvs:
            try:
                df = pd.read_csv(csvs[0])
                method_data[method] = df
                print(f"  Loaded {method}: {len(df)} rows")
            except Exception as e:
                print(f"  WARN: Could not read {csvs[0]}: {e}")
        else:
            print(f"  WARN: No CSV found for {method}")

    if not method_data:
        print("ERROR: No calibration data found. Run calibration first.")
        return

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot 1: Episode Return vs Step
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, df in method_data.items():
        if "step" in df.columns and "episode_return" in df.columns:
            ax.plot(df["step"], df["episode_return"],
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method, "black"), linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Return")
    ax.set_title(f"Calibration: Return vs Steps -- {scenario} / {maneuver}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "calibration_return.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Collision Rate vs Step
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, df in method_data.items():
        if "step" in df.columns and "collision_rate" in df.columns:
            ax.plot(df["step"], df["collision_rate"],
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method, "black"), linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Collision Rate")
    ax.set_title(f"Calibration: Collision Rate -- {scenario} / {maneuver}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "calibration_collision.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: PDE Residual vs Step
    residual_cols = {
        "hjb_aux": "hjb_residual_mean", "soft_hjb_aux": "soft_residual_mean",
        "eikonal_aux": "eikonal_residual_mean", "cbf_aux": "cbf_residual_mean",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, col in residual_cols.items():
        if method in method_data and col in method_data[method].columns:
            df = method_data[method]
            ax.plot(df["step"], df[col], label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method, "black"), linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("PDE Residual (mean)")
    ax.set_title(f"Calibration: PDE Residual -- {scenario} / {maneuver}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.savefig(os.path.join(fig_dir, "calibration_pde_residual.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 4: Wall time per iteration
    fig, ax = plt.subplots(figsize=(10, 4))
    for method, df in method_data.items():
        if "train_time_per_iter" in df.columns:
            mean_time = df["train_time_per_iter"].mean()
            ax.bar(METHOD_LABELS.get(method, method), mean_time,
                   color=METHOD_COLORS.get(method, "black"), alpha=0.8)
    ax.set_ylabel("Wall Time per Iteration (s)")
    ax.set_title("Computational Overhead per Method")
    ax.grid(True, alpha=0.3, axis="y")
    plt.savefig(os.path.join(fig_dir, "calibration_overhead.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to {fig_dir}/")

    # Convergence estimation
    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)

    for method, df in method_data.items():
        if "step" not in df.columns or "episode_return" not in df.columns:
            continue
        returns = df["episode_return"].values
        steps = df["step"].values
        if len(returns) < 5:
            print(f"  {method}: too few data points ({len(returns)})")
            continue

        n = len(returns)
        last_20_pct = returns[int(0.8 * n):]
        last_40_pct = returns[int(0.6 * n):]
        mean_last_20 = np.mean(last_20_pct)
        mean_last_40 = np.mean(last_40_pct)

        if abs(mean_last_40) > 1e-6:
            improvement = abs(mean_last_20 - mean_last_40) / abs(mean_last_40) * 100
        else:
            improvement = 0.0

        final_mean = mean_last_20
        converge_step = steps[-1]
        for i in range(len(returns) - 1, -1, -1):
            window_start = max(0, i - 4)
            window_mean = np.mean(returns[window_start:i + 1])
            if final_mean != 0 and abs(window_mean - final_mean) / abs(final_mean) > 0.10:
                converge_step = steps[min(i + 5, len(steps) - 1)]
                break

        converged = improvement < 5.0
        status = "CONVERGED" if converged else "STILL IMPROVING"
        print(f"  {method:20s}: last 20% mean={mean_last_20:+.2f}, "
              f"improvement={improvement:.1f}%, "
              f"est. convergence={converge_step}, [{status}]")

    print("\n" + "-" * 60)
    print("RECOMMENDATION: Choose TOTAL_STEPS where ALL methods have plateaued.")
    print("Then: python experiments/pde/run_full_ablation.py --tier all --total_steps <VALUE>")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Calibration runs for step count determination")
    parser.add_argument("--scenario", default="1a")
    parser.add_argument("--ego_maneuver", default="stem_right")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/calibration")
    parser.add_argument("--max_parallel", type=int, default=5)
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip training, only analyze existing results")
    args = parser.parse_args()

    if not args.analyze_only:
        print(f"Starting calibration: {len(METHODS)} methods x {args.steps} steps")
        print(f"Scenario: {args.scenario}, Maneuver: {args.ego_maneuver}, Seed: {args.seed}")
        success = run_calibration(args.scenario, args.ego_maneuver, args.steps,
                                  args.seed, args.out_dir, args.max_parallel)
        if not success:
            print("\nWARNING: Some calibration runs failed. Check logs.")

    print("\nAnalyzing calibration results...")
    analyze_calibration(args.out_dir, args.scenario, args.ego_maneuver)


if __name__ == "__main__":
    main()
