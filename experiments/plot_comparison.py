"""Plot PINN vs non-PINN training comparison: return, loss, collision, TTC."""

from __future__ import annotations

import argparse
import csv
import os


def _load_csv(path: str) -> dict:
    """Load training CSV, return dict of lists keyed by column name."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return {}
    out = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k in row:
            try:
                out[k].append(float(row[k]))
            except ValueError:
                out[k].append(row[k])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results")
    parser.add_argument("--scenario", default="1a", help="Scenario suffix for train_*.csv files")
    parser.add_argument("--out", default="results/comparison.png")
    parser.add_argument("--out_loss", default="results/comparison_loss.png")
    parser.add_argument("--out_all", default="results/comparison_all.png")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    data = {}
    for name in ["pinn", "nopinn"]:
        path = os.path.join(args.dir, f"train_{name}_{args.scenario}.csv")
        d = _load_csv(path)
        if d:
            data[name] = d

    if not data:
        print("No training CSVs found")
        return

    # 1. Original: return only
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    for name, d in data.items():
        if "step" in d and "episode_return" in d:
            ax1.plot(d["step"], d["episode_return"], label=name, alpha=0.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Episode return (mean)")
    ax1.legend()
    ax1.set_title("PINN vs non-PINN DRPPO: Episode Return")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(args.out)
    print(f"Saved {args.out}")

    # 2. Losses
    fig2, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for name, d in data.items():
        if "step" not in d:
            continue
        steps = d["step"]
        if "actor_loss" in d:
            axes[0].plot(steps, d["actor_loss"], label=name, alpha=0.8)
        if "vf_loss" in d:
            axes[1].plot(steps, d["vf_loss"], label=name, alpha=0.8)
    axes[0].set_ylabel("Actor loss")
    axes[0].legend()
    axes[0].set_title("Actor loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("VF loss (critic)")
    axes[1].legend()
    axes[1].set_title("Critic (value) loss")
    axes[1].grid(True, alpha=0.3)
    fig2.suptitle("PINN vs non-PINN DRPPO: Losses", y=1.02)
    fig2.tight_layout()
    fig2.savefig(args.out_loss)
    print(f"Saved {args.out_loss}")

    # 3. All metrics
    has_coll = any("collision_rate" in d for d in data.values())
    has_ttc = any("mean_ttc" in d for d in data.values())
    nrows = 3 + (1 if has_coll else 0) + (1 if has_ttc else 0)
    fig3, axes = plt.subplots(nrows, 1, figsize=(9, 3 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    i = 0
    axes[i].set_title("Episode return")
    for name, d in data.items():
        if "step" in d and "episode_return" in d:
            axes[i].plot(d["step"], d["episode_return"], label=name, alpha=0.8)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    i += 1
    axes[i].set_title("Actor loss")
    for name, d in data.items():
        if "step" in d and "actor_loss" in d:
            axes[i].plot(d["step"], d["actor_loss"], label=name, alpha=0.8)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    i += 1
    axes[i].set_title("Critic (VF) loss")
    for name, d in data.items():
        if "step" in d and "vf_loss" in d:
            axes[i].plot(d["step"], d["vf_loss"], label=name, alpha=0.8)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    i += 1
    if has_coll:
        axes[i].set_title("Collision rate (probe rollout)")
        for name, d in data.items():
            if "step" in d and "collision_rate" in d:
                axes[i].plot(d["step"], d["collision_rate"], label=name, alpha=0.8)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        i += 1
    if has_ttc:
        axes[i].set_title("Mean TTC (s)")
        for name, d in data.items():
            if "step" in d and "mean_ttc" in d:
                axes[i].plot(d["step"], d["mean_ttc"], label=name, alpha=0.8)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel("Step")
    fig3.suptitle("PINN vs non-PINN DRPPO: All Metrics", y=1.02)
    fig3.tight_layout()
    fig3.savefig(args.out_all)
    print(f"Saved {args.out_all}")


if __name__ == "__main__":
    main()
