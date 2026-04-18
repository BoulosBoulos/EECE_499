"""Plot learning curves from training CSVs.

Reads training CSVs from the ablation output directory, groups by method,
averages across seeds, and produces a figure with shaded confidence bands.

Usage:
    python experiments/pde/analysis/plot_learning_curves.py \
        --results_dir results/ablation/tier1 \
        --scenario 1a --maneuver stem_right \
        --out results/figures/learning_curves_1a_stem_right.png
"""

import argparse
import os
import glob
import numpy as np

METHOD_LABELS = {
    "hjb_aux": "Hard-HJB",
    "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux": "Eikonal",
    "cbf_aux": "CBF-PDE",
    "drppo": "DRPPO (baseline)",
}

METHOD_COLORS = {
    "hjb_aux": "#1f77b4",
    "soft_hjb_aux": "#ff7f0e",
    "eikonal_aux": "#2ca02c",
    "cbf_aux": "#d62728",
    "drppo": "#7f7f7f",
}


def load_training_csvs(results_dir, scenario, maneuver):
    """Load and group training CSVs by method."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas required: pip install pandas")
        return {}

    method_data = {}
    pattern = os.path.join(results_dir, f"*{scenario}_{maneuver}*", "*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        pattern = os.path.join(results_dir, "**", f"train_*_{scenario}_{maneuver}.csv")
        csv_files = glob.glob(pattern, recursive=True)

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if "step" not in df.columns or "episode_return" not in df.columns:
                continue
            basename = os.path.basename(csv_path)
            method = None
            for m in METHOD_LABELS:
                if m in basename:
                    method = m
                    break
            if method is None:
                continue
            if method not in method_data:
                method_data[method] = []
            method_data[method].append(df[["step", "episode_return"]].dropna())
        except Exception:
            continue
    return method_data


def plot_learning_curves(method_data, out_path, scenario, maneuver):
    """Plot learning curves with shaded std bands."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, dfs in method_data.items():
        if not dfs:
            continue
        all_steps = sorted(set().union(*(set(df["step"]) for df in dfs)))
        values = np.full((len(dfs), len(all_steps)), np.nan)
        for i, df in enumerate(dfs):
            for j, step in enumerate(all_steps):
                mask = df["step"] == step
                if mask.any():
                    values[i, j] = df.loc[mask, "episode_return"].values[0]
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "black")
        ax.plot(all_steps, mean, color=color, label=label, linewidth=2)
        ax.fill_between(all_steps, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title(f"Learning Curves — Scenario {scenario}, {maneuver}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--scenario", default="1a")
    parser.add_argument("--maneuver", default="stem_right")
    parser.add_argument("--out", default="results/figures/learning_curves.png")
    args = parser.parse_args()
    method_data = load_training_csvs(args.results_dir, args.scenario, args.maneuver)
    if not method_data:
        print(f"No training CSVs found in {args.results_dir} for {args.scenario}_{args.maneuver}")
        return
    plot_learning_curves(method_data, args.out, args.scenario, args.maneuver)


if __name__ == "__main__":
    main()
