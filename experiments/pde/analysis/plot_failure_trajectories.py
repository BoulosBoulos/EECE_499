"""Visualize failure episodes from recorded failure CSVs.

Produces a figure per failure showing ego trajectory with color-coded actions,
TTC timeline, and action sequence.

Usage:
    python experiments/pde/analysis/plot_failure_trajectories.py \
        --fail_dir results/pde/failures \
        --out results/figures/failures/
"""

import argparse
import os
import glob
import numpy as np


def plot_failure(csv_path, out_path):
    """Plot a single failure trajectory."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("pandas and matplotlib required")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: trajectory
    action_colors = {0: "red", 1: "orange", 2: "yellow", 3: "green", 4: "purple"}
    for i in range(len(df) - 1):
        c = action_colors.get(df["action"].iloc[i], "gray")
        axes[0].plot([df["ego_x"].iloc[i], df["ego_x"].iloc[i + 1]],
                     [df["ego_y"].iloc[i], df["ego_y"].iloc[i + 1]],
                     color=c, linewidth=2)

    # Mark collision point
    coll_rows = df[df["collision"] == 1]
    if not coll_rows.empty:
        axes[0].scatter(coll_rows["ego_x"].values, coll_rows["ego_y"].values,
                        c="red", s=100, marker="X", zorder=5, label="Collision")

    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title(os.path.basename(csv_path))
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")
    axes[0].legend()

    # Bottom: TTC timeline
    axes[1].plot(df["step"], df["ttc_min"], 'b-', linewidth=1.5)
    axes[1].axhline(y=3.0, color="r", linestyle="--", alpha=0.5, label="TTC threshold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("TTC_min (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail_dir", required=True)
    parser.add_argument("--out", default="results/figures/failures")
    parser.add_argument("--max_plots", type=int, default=20)
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.fail_dir, "*.csv")))
    if not csv_files:
        print(f"No failure CSVs found in {args.fail_dir}")
        return

    print(f"Found {len(csv_files)} failure trajectories")
    for i, csv_path in enumerate(csv_files[:args.max_plots]):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(args.out, f"{name}.png")
        plot_failure(csv_path, out_path)
        print(f"  [{i + 1}/{min(len(csv_files), args.max_plots)}] {out_path}")

    print(f"Done. Figures saved to {args.out}")


if __name__ == "__main__":
    main()
