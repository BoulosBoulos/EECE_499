"""Generate offline plots for the interaction benchmark results.

Reads interaction_eval_results.csv and produces:
  - Per-scenario bar charts for all metrics
  - Per-template breakdown
  - Method comparison (nopinn vs hjb_aux vs soft_hjb_aux vs rule)
  - Training curves
"""

from __future__ import annotations

import argparse
import csv
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def plot_eval_bars(results_csv: str, out_dir: str):
    """Bar charts per scenario and variant for all metrics."""
    if plt is None:
        print("matplotlib not available, skipping plots.")
        return
    if not os.path.isfile(results_csv):
        print(f"No results file: {results_csv}")
        return

    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    det_rows = [r for r in rows if r.get("eval_mode") == "deterministic"]
    if not det_rows:
        det_rows = rows

    metrics = [
        "mean_return", "success_rate", "collision_rate",
        "row_violation_rate", "conflict_intrusion_rate",
        "forced_brake_rate", "deadlock_rate", "unnecessary_wait_rate",
        "mean_steps", "mean_progress",
    ]
    available = [m for m in metrics if any(r.get(m) for r in det_rows)]

    scenarios = sorted(set(r["scenario"] for r in det_rows))
    variants = sorted(set(r["variant"] for r in det_rows))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(variants), 2)))

    for metric in available:
        fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 2), 5))
        x = np.arange(len(scenarios))
        width = 0.8 / max(len(variants), 1)

        for i, var in enumerate(variants):
            vals = []
            for sc in scenarios:
                vr = [_safe_float(r[metric]) for r in det_rows
                      if r["variant"] == var and r["scenario"] == sc]
                vals.append(np.mean(vr) if vr else 0)
            ax.bar(x + i * width, vals, width, label=var, color=colors[i])

        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels(scenarios)
        ax.set_ylabel(metric)
        ax.set_title(f"Interaction Benchmark: {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"interaction_{metric}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved interaction_{metric}.png")


def plot_training_curves(train_dir: str, out_dir: str):
    """Plot training curves from interaction training CSVs."""
    if plt is None:
        return

    import glob
    csvs = glob.glob(os.path.join(train_dir, "train_interaction_*.csv"))
    if not csvs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = ["mean_return", "collision_rate", "success_rate", "row_violation_rate"]

    for csv_path in csvs:
        label = os.path.basename(csv_path).replace("train_interaction_", "").replace(".csv", "")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        steps = [_safe_float(r.get("total_env_steps", 0)) for r in rows]
        for ax, metric in zip(axes.flat, metrics_to_plot):
            vals = [_safe_float(r.get(metric, 0)) for r in rows]
            ax.plot(steps, vals, label=label, alpha=0.8)
            ax.set_xlabel("Env Steps")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.legend(fontsize=7)

    fig.suptitle("Interaction Benchmark Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interaction_training_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved interaction_training_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Plot interaction benchmark results")
    parser.add_argument("--results_csv",
                        default="results/interaction/interaction_eval_results.csv")
    parser.add_argument("--train_dir", default="results/interaction")
    parser.add_argument("--out_dir", default="results/interaction")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Generating interaction benchmark plots...")
    plot_eval_bars(args.results_csv, args.out_dir)
    plot_training_curves(args.train_dir, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
