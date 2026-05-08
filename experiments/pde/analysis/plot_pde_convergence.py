"""Plot PDE residual convergence from training CSVs.

Shows how the physics loss decreases during training for each PDE method.

Usage:
    python experiments/pde/analysis/plot_pde_convergence.py \
        --results_dir results/ablation/tier1 \
        --scenario 1a --maneuver stem_right
"""

import argparse
import os
import glob
import numpy as np

RESIDUAL_COLUMNS = {
    "hjb_aux": "hjb_residual_mean",
    "soft_hjb_aux": "soft_residual_mean",
    "eikonal_aux": "eikonal_residual_mean",
    "cbf_aux": "cbf_residual_mean",
}

METHOD_LABELS = {
    "hjb_aux": "Hard-HJB",
    "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux": "Eikonal",
    "cbf_aux": "CBF-PDE",
}

METHOD_COLORS = {
    "hjb_aux": "#1f77b4",
    "soft_hjb_aux": "#ff7f0e",
    "eikonal_aux": "#2ca02c",
    "cbf_aux": "#d62728",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--scenario", default="1a")
    parser.add_argument("--maneuver", default="stem_right")
    parser.add_argument("--out", default="results/figures/pde_convergence.png")
    args = parser.parse_args()

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("pandas and matplotlib required")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method, res_col in RESIDUAL_COLUMNS.items():
        pattern = os.path.join(args.results_dir, "**", f"train_{method}_{args.scenario}_{args.maneuver}.csv")
        csv_files = glob.glob(pattern, recursive=True)
        if not csv_files:
            pattern = os.path.join(args.results_dir, f"*{args.scenario}_{args.maneuver}_{method}*", "*.csv")
            csv_files = glob.glob(pattern)

        all_residuals = []
        all_distill = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if "step" in df.columns and res_col in df.columns:
                    all_residuals.append(df[["step", res_col]].dropna())
                if "step" in df.columns and "distill_loss" in df.columns:
                    all_distill.append(df[["step", "distill_loss"]].dropna())
            except Exception:
                continue

        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "black")

        if all_residuals:
            steps = all_residuals[0]["step"].values
            vals = np.array([df[res_col].values[:len(steps)] for df in all_residuals])
            mean = np.nanmean(vals, axis=0)
            axes[0].plot(steps[:len(mean)], mean, color=color, label=label, linewidth=2)

        if all_distill:
            steps = all_distill[0]["step"].values
            vals = np.array([df["distill_loss"].values[:len(steps)] for df in all_distill])
            mean = np.nanmean(vals, axis=0)
            axes[1].plot(steps[:len(mean)], mean, color=color, label=label, linewidth=2)

    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("PDE Residual (mean)")
    axes[0].set_title("PDE Residual Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Distillation Loss")
    axes[1].set_title("Critic Distillation Gap")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
