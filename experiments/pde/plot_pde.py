"""Plot PDE-family training curves, ablation results, and three-way comparison."""

from __future__ import annotations

import argparse
import os
import csv


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde_dir", default="results/pde_ablation")
    parser.add_argument("--legacy_dir", default="results/ablation")
    parser.add_argument("--out_dir", default="results/pde_ablation")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    colors_map = {
        "nopinn": "#e74c3c", "pinn_critic": "#3498db", "pinn_ego": "#2ecc71",
        "pinn_actor": "#1abc9c", "hjb_aux": "#9b59b6", "soft_hjb_aux": "#f39c12",
    }

    # ── 1. PDE Training Curves ──────────────────────────────────────────────
    train_path = os.path.join(args.pde_dir, "ablation_train_log.csv")
    if os.path.isfile(train_path):
        rows = []
        with open(train_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            variants = sorted(set(r["variant"] for r in rows))
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for var in variants:
                vr = [r for r in rows if r["variant"] == var]
                steps = [_safe_float(r.get("step", 0)) for r in vr]
                axes[0, 0].plot(steps, [_safe_float(r.get("actor_loss", 0)) for r in vr], label=var, alpha=0.7)
                res_key = "hjb_residual_mean" if "hjb" in var and "soft" not in var else "soft_residual_mean"
                axes[0, 1].plot(steps, [_safe_float(r.get(res_key, 0)) for r in vr], label=var, alpha=0.7)
                axes[1, 0].plot(steps, [_safe_float(r.get("distill_gap", 0)) for r in vr], label=var, alpha=0.7)
                axes[1, 1].plot(steps, [_safe_float(r.get("actor_align_kl", 0)) for r in vr], label=var, alpha=0.7)

            axes[0, 0].set_title("Actor Loss")
            axes[0, 1].set_title("PDE Residual")
            axes[1, 0].set_title("Distillation Gap")
            axes[1, 1].set_title("Actor Alignment KL")
            for ax in axes.flat:
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Step")
                ax.legend()
            fig.tight_layout()
            out = os.path.join(args.out_dir, "pde_training_curves.png")
            fig.savefig(out, dpi=150)
            print(f"Saved {out}")
            plt.close()

    # ── 2. PDE Ablation Bars (all metrics) ──────────────────────────────────
    eval_path = os.path.join(args.pde_dir, "ablation_results.csv")
    if os.path.isfile(eval_path):
        rows = []
        with open(eval_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            det_rows = [r for r in rows if r.get("eval_mode") == "deterministic"]
            if det_rows:
                variants = sorted(set(r["variant"] for r in det_rows))
                metrics = {
                    "mean_return": "Mean Return",
                    "collision_rate": "Collision Rate (%)",
                    "mean_ttc": "Mean TTC (s)",
                    "min_ttc": "Min TTC (s)",
                    "pothole_hits_mean": "Pothole Hits",
                }
                available = [m for m in metrics if any(r.get(m) for r in det_rows)]
                n = len(available)
                if n > 0:
                    fig, axes = plt.subplots(1, min(n, 5), figsize=(5 * min(n, 5), 5), squeeze=False)
                    axes = axes[0]
                    for idx, m in enumerate(available[:5]):
                        vals = {v: np.mean([_safe_float(r[m]) for r in det_rows if r["variant"] == v])
                                for v in variants}
                        mult = 100 if "rate" in m else 1
                        axes[idx].bar(vals.keys(), [v * mult for v in vals.values()],
                                      color=[colors_map.get(v, "gray") for v in vals.keys()])
                        axes[idx].set_title(metrics[m])
                        axes[idx].grid(True, alpha=0.3, axis="y")
                    fig.tight_layout()
                    out = os.path.join(args.out_dir, "pde_ablation_bars.png")
                    fig.savefig(out, dpi=150)
                    print(f"Saved {out}")
                    plt.close()

    # ── 3. Three-Way Comparison: legacy vs PDE ──────────────────────────────
    legacy_path = os.path.join(args.legacy_dir, "ablation_results.csv")
    if os.path.isfile(legacy_path) and os.path.isfile(eval_path):
        legacy_rows = []
        with open(legacy_path) as f:
            legacy_rows = list(csv.DictReader(f))
        pde_rows = []
        with open(eval_path) as f:
            pde_rows = list(csv.DictReader(f))

        legacy_det = [r for r in legacy_rows if r.get("eval_mode") == "deterministic"]
        pde_det = [r for r in pde_rows if r.get("eval_mode") == "deterministic"]

        if legacy_det and pde_det:
            all_data = {}
            for v in ["nopinn", "pinn_critic", "pinn_ego", "pinn_actor"]:
                vr = [r for r in legacy_det if r.get("variant") == v]
                if vr:
                    all_data[v] = vr
            for v in ["hjb_aux", "soft_hjb_aux"]:
                vr = [r for r in pde_det if r.get("variant") == v]
                if vr:
                    all_data[v] = vr

            if all_data:
                compare_metrics = ["mean_return", "collision_rate", "mean_ttc"]
                labels = {
                    "mean_return": "Mean Return",
                    "collision_rate": "Collision Rate (%)",
                    "mean_ttc": "Mean TTC (s)",
                }
                avail = [m for m in compare_metrics
                         if any(any(r.get(m) for r in vr) for vr in all_data.values())]
                n = len(avail)
                if n > 0:
                    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
                    axes = axes[0]
                    for idx, m in enumerate(avail):
                        vals = {}
                        for v, vr in all_data.items():
                            raw = [_safe_float(r.get(m, 0)) for r in vr]
                            if raw:
                                vals[v] = np.mean(raw) * (100 if "rate" in m else 1)
                        axes[idx].bar(vals.keys(), vals.values(),
                                      color=[colors_map.get(v, "gray") for v in vals.keys()])
                        axes[idx].set_title(f"Three-Way: {labels.get(m, m)}")
                        axes[idx].set_ylabel(labels.get(m, m))
                        axes[idx].grid(True, alpha=0.3, axis="y")
                        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=30, ha="right")
                    fig.tight_layout()
                    out = os.path.join(args.out_dir, "pde_vs_legacy_comparison.png")
                    fig.savefig(out, dpi=150)
                    print(f"Saved {out}")
                    plt.close()

    print(f"All plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
