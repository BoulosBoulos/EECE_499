"""Plot interaction/negotiation analysis from trajectory CSVs."""

from __future__ import annotations

import argparse
import os
import csv

ACTION_COLORS = {
    "STOP": "#e74c3c",
    "CREEP": "#f39c12",
    "YIELD": "#3498db",
    "GO": "#2ecc71",
    "ABORT": "#8e44ad",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Trajectory CSV from visualize_sumo.py")
    parser.add_argument("--out_dir", default="results/pde")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed")
        return

    rows = []
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Empty CSV")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.csv))[0]

    steps = [int(r["step"]) for r in rows]
    ego_x = [float(r["ego_x"]) for r in rows]
    ego_y = [float(r["ego_y"]) for r in rows]
    ego_v = [float(r["ego_v"]) for r in rows]
    actions = [r.get("action_name", "?") for r in rows]
    ttc = [float(r.get("ttc_min", 10)) for r in rows]
    nearest = [float(r.get("nearest_agent_dist", 100)) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    act_colors = [ACTION_COLORS.get(a, "gray") for a in actions]
    ax.scatter(ego_x, ego_y, c=act_colors, s=8, alpha=0.8)
    n_ag = 0
    for k in rows[0]:
        if k.startswith("ag") and k.endswith("_x"):
            n_ag += 1
    for i in range(min(n_ag, 5)):
        if f"ag{i}_x" in rows[0]:
            ax_x = [float(r.get(f"ag{i}_x", 0)) for r in rows]
            ax_y = [float(r.get(f"ag{i}_y", 0)) for r in rows]
            label = rows[0].get(f"ag{i}_type", f"agent{i}")
            ax.plot(ax_x, ax_y, "--", alpha=0.5, label=label, linewidth=1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Position Trace (color = action)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    action_indices = {"STOP": 0, "CREEP": 1, "YIELD": 2, "GO": 3, "ABORT": 4}
    act_idx = [action_indices.get(a, -1) for a in actions]
    for aname, color in ACTION_COLORS.items():
        mask = [1 if a == aname else None for a in actions]
        y_vals = [action_indices[aname] if m else None for m in mask]
        x_vals = [s for s, m in zip(steps, mask) if m]
        y_plot = [action_indices[aname]] * len(x_vals)
        ax.scatter(x_vals, y_plot, c=color, s=12, label=aname, alpha=0.8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["STOP", "CREEP", "YIELD", "GO", "ABORT"])
    ax.set_xlabel("Step")
    ax.set_title("Action Timeline")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, ego_v, "b-", label="Ego speed", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(steps, nearest, "r--", label="Nearest agent dist", alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed (m/s)", color="b")
    ax2.set_ylabel("Distance (m)", color="r")
    ax.set_title("Speed & Nearest Agent Distance")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, ttc, "g-", alpha=0.8)
    ax.axhline(y=3.0, color="r", linestyle="--", alpha=0.5, label="TTC threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("TTC (s)")
    ax.set_title("Time-to-Collision Profile")
    ax.legend()
    ax.set_ylim(0, min(max(ttc) + 1, 12))
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Interaction Analysis: {base}", fontsize=14)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"{base}_interaction.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
