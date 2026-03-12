"""Plot ablation study results: mean return, collision rate, etc. across scenarios and variants."""

from __future__ import annotations

import argparse
import csv
import os

SCENARIOS = ["1a", "1b", "1c", "1d", "2", "3", "4"]
VARIANTS = ["nopinn", "pinn", "pinn_ego"]


def main():
    parser = argparse.ArgumentParser(description="Plot ablation results")
    parser.add_argument("--csv", default="results/ablation/ablation_results.csv")
    parser.add_argument("--out_dir", default="results/ablation")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed; pip install matplotlib numpy")
        return

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return

    rows = []
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV is empty")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Build data: scenario -> variant -> value
    def get_val(row, key, default=0):
        try:
            return float(row.get(key, default))
        except (ValueError, TypeError):
            return default

    scenarios = sorted(set(r["scenario"] for r in rows))
    variants = sorted(set(r["variant"] for r in rows))
    x = np.arange(len(scenarios))
    width = 0.25
    n_variants = len(variants)

    # Colors
    colors = {"nopinn": "#e74c3c", "pinn": "#3498db", "pinn_ego": "#2ecc71"}

    # 1. Mean return by scenario
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i, var in enumerate(variants):
        vals = []
        stds = []
        for sc in scenarios:
            r = next((x for x in rows if x["scenario"] == sc and x["variant"] == var), None)
            if r:
                vals.append(get_val(r, "mean_return"))
                stds.append(get_val(r, "std_return", 0))
            else:
                vals.append(0)
                stds.append(0)
        offset = (i - n_variants / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width, yerr=stds, label=var, color=colors.get(var, "gray"), capsize=2)
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Mean return")
    ax1.set_title("Ablation: Mean Return by Scenario and Variant")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    fig1.tight_layout()
    out1 = os.path.join(args.out_dir, "ablation_mean_return.png")
    fig1.savefig(out1, dpi=150)
    print(f"Saved {out1}")

    # 2. Collision rate by scenario
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, var in enumerate(variants):
        vals = []
        for sc in scenarios:
            r = next((x for x in rows if x["scenario"] == sc and x["variant"] == var), None)
            vals.append(get_val(r, "collision_rate", 0) * 100 if r else 0)
        offset = (i - n_variants / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width, label=var, color=colors.get(var, "gray"))
    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Collision rate (%)")
    ax2.set_title("Ablation: Collision Rate by Scenario and Variant")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    out2 = os.path.join(args.out_dir, "ablation_collision_rate.png")
    fig2.savefig(out2, dpi=150)
    print(f"Saved {out2}")

    # 3. Combined: return and collision (2 subplots)
    fig3, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i, var in enumerate(variants):
        vals_ret = []
        vals_coll = []
        for sc in scenarios:
            r = next((x for x in rows if x["scenario"] == sc and x["variant"] == var), None)
            vals_ret.append(get_val(r, "mean_return", 0) if r else 0)
            vals_coll.append(get_val(r, "collision_rate", 0) * 100 if r else 0)
        offset = (i - n_variants / 2 + 0.5) * width
        axes[0].bar(x + offset, vals_ret, width, label=var, color=colors.get(var, "gray"))
        axes[1].bar(x + offset, vals_coll, width, label=var, color=colors.get(var, "gray"))
    axes[0].set_ylabel("Mean return")
    axes[0].set_title("Mean Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylabel("Collision rate (%)")
    axes[1].set_title("Collision Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
    fig3.suptitle("Ablation Study: nopinn vs pinn vs pinn_ego", y=1.02)
    fig3.tight_layout()
    out3 = os.path.join(args.out_dir, "ablation_comparison.png")
    fig3.savefig(out3, dpi=150)
    print(f"Saved {out3}")

    # 4. Aggregate bar: mean across scenarios per variant
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    agg_ret = {}
    agg_coll = {}
    for var in variants:
        v_ret = [get_val(r, "mean_return") for r in rows if r["variant"] == var]
        v_coll = [get_val(r, "collision_rate") * 100 for r in rows if r["variant"] == var]
        agg_ret[var] = np.mean(v_ret) if v_ret else 0
        agg_coll[var] = np.mean(v_coll) if v_coll else 0
    x_agg = np.arange(2)  # [0]=return, [1]=collision
    w = 0.25
    for i, var in enumerate(variants):
        ax4.bar(x_agg + (i - len(variants) / 2 + 0.5) * w, [agg_ret[var], agg_coll[var]], w, label=var, color=colors.get(var, "gray"))
    ax4.set_xticks(x_agg)
    ax4.set_xticklabels(["Mean return (avg)", "Collision rate % (avg)"])
    ax4.set_ylabel("Value")
    ax4.set_title("Ablation: Aggregate Metrics Across All Scenarios")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    fig4.tight_layout()
    out4 = os.path.join(args.out_dir, "ablation_aggregate.png")
    fig4.savefig(out4, dpi=150)
    print(f"Saved {out4}")

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
