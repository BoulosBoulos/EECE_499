"""Calibration diagnostic — read-only post-hoc analysis of the 36-run calibration.

Produces three artifacts (no training, no YAML/lock changes):
  1. all_runs_per_cell.{pdf,png}: per-(method,scenario) cell, all 3 seeds overlaid
     (top: smoothed mean_reward, bottom: rolling collision rate)
  2. eikonal_diagnostic.{pdf,png}: Eikonal residual + notes on what isn't logged
  3. per_run_diagnostic.json: per-run final stats + per-criterion violation magnitudes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
CAL_ROOT = REPO_ROOT / "results" / "calibration"
OUT_ROOT = CAL_ROOT / "_analysis" / "diagnostic_curves"

METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]
METHOD_LABELS = {
    "drppo": "DR-PPO",
    "hjb_aux": "HJB",
    "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux": "Eikonal",
    "cbf_aux": "CBF",
    "fusion_aux": "Fusion",
}
SCENARIOS = ["1a", "2_dense"]
SEEDS = [42, 123, 456]
SEED_COLOR = {42: "tab:blue", 123: "tab:orange", 456: "tab:green"}

CONVERGENCE_WINDOW_STEPS = 50000
REWARD_STD_REL_THRESHOLD = 0.05
COLLISION_STD_ABS_THRESHOLD = 0.02

# rolling window in steps (~20k) — at 4096 steps/iter this is ~5 iters
ROLL_STEPS = 20000


def run_id(method: str, scenario: str, seed: int) -> str:
    return f"CAL_{method}_{scenario}_s{seed}"


def load_run(method: str, scenario: str, seed: int) -> Optional[pd.DataFrame]:
    run_dir = CAL_ROOT / run_id(method, scenario, seed)
    csv = run_dir / "metrics.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if "n_episodes" in df and "n_collisions" in df:
        ep = df["n_episodes"].clip(lower=1)
        df["collision_rate"] = df["n_collisions"] / ep
    else:
        df["collision_rate"] = np.nan
    return df


def rolling_steps(steps: pd.Series, values: pd.Series, window_steps: int):
    """Rolling mean over a step-window. Each iteration is ~uniform 4096 steps,
    so use ceil(window/4096) as the iteration count."""
    if len(steps) < 2:
        return values.copy()
    iter_size = float(np.median(np.diff(steps.values)))
    if not np.isfinite(iter_size) or iter_size <= 0:
        return values.copy()
    win = max(1, int(round(window_steps / iter_size)))
    return values.rolling(win, min_periods=1).mean()


def final_window_stats(df: pd.DataFrame, window_steps: int = CONVERGENCE_WINDOW_STEPS):
    """Stats over the final `window_steps` of training."""
    if df is None or len(df) == 0:
        return None
    final_step = df["total_steps"].iloc[-1]
    mask = df["total_steps"] >= final_step - window_steps
    win = df.loc[mask]
    if len(win) < 2:
        return None
    rewards = win["mean_reward"].to_numpy()
    coll = win["collision_rate"].to_numpy()
    last5 = df.tail(5)
    return {
        "n_iters_in_window": int(len(win)),
        "first_step_in_window": int(win["total_steps"].iloc[0]),
        "last_step_in_window": int(win["total_steps"].iloc[-1]),
        "final_mean_reward_last5": float(last5["mean_reward"].mean()),
        "final_collision_rate_last5": float(last5["collision_rate"].mean()),
        "reward_window_mean": float(rewards.mean()),
        "reward_window_std": float(rewards.std(ddof=1)),
        "reward_std_rel": float(rewards.std(ddof=1) / max(abs(rewards.mean()), 1e-9)),
        "collision_window_mean": float(coll.mean()),
        "collision_window_std": float(coll.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
# Output 1 — all 36 runs, per-cell overlay
# ---------------------------------------------------------------------------
def plot_all_runs_per_cell(out_dir: Path):
    fig, axes = plt.subplots(
        len(METHODS), 4, figsize=(18, 18),
        sharex=True, squeeze=False,
        gridspec_kw={"hspace": 0.35, "wspace": 0.25},
    )
    # column layout: [scen1a reward, scen1a coll, scen2_dense reward, scen2_dense coll]
    col_titles = ["1a — reward", "1a — collision", "2_dense — reward", "2_dense — collision"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11)

    for r, method in enumerate(METHODS):
        for s_idx, scen in enumerate(SCENARIOS):
            ax_r = axes[r, 2 * s_idx]
            ax_c = axes[r, 2 * s_idx + 1]
            for seed in SEEDS:
                df = load_run(method, scen, seed)
                if df is None or len(df) == 0:
                    continue
                steps = df["total_steps"]
                r_smooth = rolling_steps(steps, df["mean_reward"], ROLL_STEPS)
                c_smooth = rolling_steps(steps, df["collision_rate"], ROLL_STEPS)
                color = SEED_COLOR[seed]
                ax_r.plot(steps, r_smooth, color=color, label=f"seed {seed}", linewidth=1.2, alpha=0.85)
                ax_c.plot(steps, c_smooth, color=color, label=f"seed {seed}", linewidth=1.2, alpha=0.85)
            ax_c.axhline(COLLISION_STD_ABS_THRESHOLD, color="grey", linestyle=":", linewidth=0.7)
            ax_r.grid(alpha=0.3)
            ax_c.grid(alpha=0.3)
            ax_c.set_ylim(bottom=0.0)
        axes[r, 0].set_ylabel(METHOD_LABELS[method], fontsize=11)

    for c in range(4):
        axes[-1, c].set_xlabel("total_steps")
    handles = [plt.Line2D([0], [0], color=SEED_COLOR[s], lw=2, label=f"seed {s}") for s in SEEDS]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=True, fontsize=10)
    fig.suptitle(
        "Calibration — all 36 runs, smoothed mean_reward and collision rate (rolling ~20k steps)",
        fontsize=13, y=0.995,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"all_runs_per_cell.{ext}"
        fig.savefig(p, dpi=180 if ext == "png" else 300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Output 2 — Eikonal residual diagnostic
# ---------------------------------------------------------------------------
def plot_eikonal_diagnostic(out_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(
        len(SCENARIOS), len(SEEDS), figsize=(15, 7),
        sharex=True, squeeze=False,
        gridspec_kw={"hspace": 0.3, "wspace": 0.25},
    )
    summary_lines = []
    for r, scen in enumerate(SCENARIOS):
        for c, seed in enumerate(SEEDS):
            ax = axes[r, c]
            df = load_run("eikonal_aux", scen, seed)
            if df is None:
                ax.set_title(f"{scen} / s{seed} — missing", fontsize=10)
                continue
            steps = df["total_steps"]
            res = df.get("L_residual_safety")
            if res is None:
                ax.text(0.5, 0.5, "L_residual_safety not logged",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            ax.plot(steps, res, color="firebrick", linewidth=1.0, alpha=0.5, label="raw")
            res_smooth = rolling_steps(steps, res, ROLL_STEPS)
            ax.plot(steps, res_smooth, color="darkred", linewidth=1.6, label="smoothed (~20k)")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{scen} / seed {seed}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"L_residual_safety  (log)\n[scenario {scen}]")
            if r == 1:
                ax.set_xlabel("total_steps")

            head = float(res.head(20).mean()) if len(res) >= 20 else float(res.mean())
            tail = float(res.tail(20).mean()) if len(res) >= 20 else float(res.mean())
            ratio = (tail / head) if head > 0 else float("nan")
            summary_lines.append(
                f"eikonal_aux/{scen}/s{seed}: head={head:.4g} tail={tail:.4g} ratio_t/h={ratio:.4g}"
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True, fontsize=9)
    note = (
        "Note: U_safety @ reference state and aux-critic gradient norm were NOT logged by the "
        "training scripts (only L_residual_safety appears in metrics.csv). Adding those traces "
        "would require modifying training code, which is out of scope per the current request."
    )
    fig.suptitle("Eikonal diagnostic — L_residual_safety per run", fontsize=13, y=0.995)
    fig.text(0.5, -0.03, note, ha="center", fontsize=9, wrap=True)
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"eikonal_diagnostic.{ext}"
        fig.savefig(p, dpi=180 if ext == "png" else 300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)

    eikonal_summary_path = out_dir / "eikonal_residual_head_tail.txt"
    eikonal_summary_path.write_text("\n".join(summary_lines) + "\n")
    paths.append(eikonal_summary_path)
    return paths


# ---------------------------------------------------------------------------
# Output 3 — per-run JSON diagnostic
# ---------------------------------------------------------------------------
def per_run_json(out_dir: Path) -> Path:
    entries = {}
    for method in METHODS:
        for scen in SCENARIOS:
            for seed in SEEDS:
                rid = run_id(method, scen, seed)
                df = load_run(method, scen, seed)
                if df is None:
                    entries[rid] = {"error": "metrics.csv missing"}
                    continue
                stats = final_window_stats(df)
                if stats is None:
                    entries[rid] = {"error": "final-window stats unavailable"}
                    continue
                reward_ok = stats["reward_std_rel"] < REWARD_STD_REL_THRESHOLD
                coll_ok = stats["collision_window_std"] < COLLISION_STD_ABS_THRESHOLD
                # Magnitude of violation (positive = how much over threshold)
                reward_violation = max(0.0, stats["reward_std_rel"] - REWARD_STD_REL_THRESHOLD)
                coll_violation = max(0.0, stats["collision_window_std"] - COLLISION_STD_ABS_THRESHOLD)
                entries[rid] = {
                    "method": method,
                    "scenario": scen,
                    "seed": seed,
                    "total_iterations": int(len(df)),
                    "total_steps_actual": int(df["total_steps"].iloc[-1]),
                    "final_mean_reward_last5": stats["final_mean_reward_last5"],
                    "final_collision_rate_last5": stats["final_collision_rate_last5"],
                    "final_50k_window": {
                        "first_step": stats["first_step_in_window"],
                        "last_step": stats["last_step_in_window"],
                        "n_iters": stats["n_iters_in_window"],
                        "reward_mean": stats["reward_window_mean"],
                        "reward_std": stats["reward_window_std"],
                        "reward_std_rel_pct": stats["reward_std_rel"] * 100.0,
                        "collision_mean": stats["collision_window_mean"],
                        "collision_std": stats["collision_window_std"],
                    },
                    "criteria": {
                        "reward_std_rel_threshold": REWARD_STD_REL_THRESHOLD,
                        "collision_std_abs_threshold": COLLISION_STD_ABS_THRESHOLD,
                        "reward_passed": bool(reward_ok),
                        "collision_passed": bool(coll_ok),
                        "all_passed": bool(reward_ok and coll_ok),
                        "reward_violation_above_thresh": reward_violation,
                        "collision_violation_above_thresh": coll_violation,
                    },
                }
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "per_run_diagnostic.json"
    p.write_text(json.dumps(entries, indent=2))
    return p


def find_top_close_to_passing(json_path: Path, top_n: int = 3):
    data = json.loads(json_path.read_text())
    rows = []
    for rid, e in data.items():
        if "criteria" not in e:
            continue
        score = e["criteria"]["reward_violation_above_thresh"] + e["criteria"]["collision_violation_above_thresh"]
        # only include cells that are FAILING (would be passing if score=0)
        if e["criteria"]["all_passed"]:
            continue
        rows.append((score, rid, e))
    rows.sort(key=lambda x: x[0])
    return rows[:top_n]


def quick_2dense_summary() -> List[str]:
    """Look at 2_dense reward trajectories — climbing, oscillating, or flat?"""
    out = []
    for method in METHODS:
        head_means = []
        tail_means = []
        for seed in SEEDS:
            df = load_run(method, "2_dense", seed)
            if df is None:
                continue
            head_means.append(float(df["mean_reward"].head(10).mean()))
            tail_means.append(float(df["mean_reward"].tail(10).mean()))
        if head_means:
            h = np.mean(head_means)
            t = np.mean(tail_means)
            out.append(
                f"{METHOD_LABELS[method]:<10s}  start={h:>9.1f}  end={t:>9.1f}  Δ={t - h:+.1f}"
            )
    return out


def quick_eikonal_residual_summary() -> List[str]:
    out = []
    for scen in SCENARIOS:
        for seed in SEEDS:
            df = load_run("eikonal_aux", scen, seed)
            if df is None or "L_residual_safety" not in df:
                continue
            res = df["L_residual_safety"]
            head = float(res.head(20).mean())
            tail = float(res.tail(20).mean())
            if head > 0:
                ratio = tail / head
                trend = "decreasing" if ratio < 0.9 else ("increasing" if ratio > 1.1 else "flat")
            else:
                ratio = float("nan")
                trend = "(head=0)"
            out.append(f"eikonal/{scen}/s{seed}: head={head:.3g} tail={tail:.3g} ratio={ratio:.3g} ({trend})")
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[diag] generating per-cell overlay…")
    p1 = plot_all_runs_per_cell(OUT_ROOT)
    print("[diag] generating Eikonal diagnostic…")
    p2 = plot_eikonal_diagnostic(OUT_ROOT)
    print("[diag] generating per-run JSON…")
    p3 = per_run_json(OUT_ROOT)

    print("\n=== 2_dense reward trajectory summary (mean across 3 seeds) ===")
    for line in quick_2dense_summary():
        print("  " + line)

    print("\n=== Eikonal residual head/tail summary ===")
    for line in quick_eikonal_residual_summary():
        print("  " + line)

    print("\n=== Top 3 cells closest to passing (smallest violation magnitude) ===")
    top = find_top_close_to_passing(p3, top_n=3)
    if not top:
        print("  (none — every failing run has a finite violation, or all runs already passed)")
    else:
        for score, rid, e in top:
            crit = e["criteria"]
            print(
                f"  {rid}: violation={score:.4f}  "
                f"reward_std_rel={e['final_50k_window']['reward_std_rel_pct']:.2f}%  "
                f"coll_std={e['final_50k_window']['collision_std']:.4f}"
            )

    print("\n=== Files ===")
    for p in p1 + p2 + [p3]:
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
