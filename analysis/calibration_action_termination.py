"""Stage 1.2 — action distribution + termination cause diagnostic for 2_dense vs 1a.

Reads existing metrics.csv for the 36 calibration runs and produces:
  - action_dist_2dense_vs_1a.{pdf,png}: action distribution + termination cause per cell
  - termination_distribution.csv: tabular termination counts per cell
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
CAL_ROOT = REPO_ROOT / "results" / "calibration"
OUT_DIR = CAL_ROOT / "_analysis" / "diagnostic_curves"

METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]
METHOD_LABELS = {
    "drppo": "DR-PPO", "hjb_aux": "HJB", "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux": "Eikonal", "cbf_aux": "CBF", "fusion_aux": "Fusion",
}
SCENARIOS = ["1a", "2_dense"]
SEEDS = [42, 123, 456]
ACTIONS = ["stop", "creep", "yield", "go", "abort"]
ACTION_COLORS = {
    "stop": "tab:red", "creep": "tab:orange", "yield": "tab:olive",
    "go": "tab:green", "abort": "tab:purple",
}
TERM_COLORS = {
    "collision": "tab:red", "timeout": "tab:gray",
    "success": "tab:green", "abort": "tab:purple",
}


def run_id(method: str, scenario: str, seed: int) -> str:
    return f"CAL_{method}_{scenario}_s{seed}"


def load_run(method: str, scenario: str, seed: int) -> Optional[pd.DataFrame]:
    csv = CAL_ROOT / run_id(method, scenario, seed) / "metrics.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def cell_action_dist(method: str, scenario: str) -> Optional[pd.DataFrame]:
    """Mean action distribution over iterations across the 3 seeds."""
    frames = []
    for seed in SEEDS:
        df = load_run(method, scenario, seed)
        if df is None:
            continue
        cols = ["total_steps"] + [f"action_dist_{a}" for a in ACTIONS]
        if not all(c in df.columns for c in cols):
            continue
        frames.append(df[cols].copy())
    if not frames:
        return None
    aligned = frames[0].copy()
    for c in [f"action_dist_{a}" for a in ACTIONS]:
        aligned[c] = np.mean([f[c].values for f in frames], axis=0)
    return aligned


def cell_termination_fractions(method: str, scenario: str) -> Optional[pd.DataFrame]:
    """Fraction of episodes ending each way, averaged over the 3 seeds and aligned by total_steps."""
    frames = []
    for seed in SEEDS:
        df = load_run(method, scenario, seed)
        if df is None:
            continue
        cols = ["total_steps", "n_episodes", "n_collisions", "n_successes", "n_timeouts", "n_aborts", "mean_episode_length"]
        if not all(c in df.columns for c in cols):
            continue
        total = df["n_episodes"].clip(lower=1)
        f = pd.DataFrame({
            "total_steps": df["total_steps"],
            "frac_collision": df["n_collisions"] / total,
            "frac_success": df["n_successes"] / total,
            "frac_timeout": df["n_timeouts"] / total,
            "frac_abort": df["n_aborts"] / total,
            "ep_len": df["mean_episode_length"],
            "n_eps": df["n_episodes"],
        })
        frames.append(f)
    if not frames:
        return None
    aligned = frames[0].copy()
    for c in ["frac_collision", "frac_success", "frac_timeout", "frac_abort", "ep_len", "n_eps"]:
        aligned[c] = np.mean([f[c].values for f in frames], axis=0)
    return aligned


def plot_action_termination(out_path_base: Path):
    fig, axes = plt.subplots(
        len(METHODS), 4, figsize=(20, 22),
        sharex=True, squeeze=False,
        gridspec_kw={"hspace": 0.32, "wspace": 0.22},
    )
    col_titles = ["1a — action dist", "1a — termination", "2_dense — action dist", "2_dense — termination"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11)

    for r, method in enumerate(METHODS):
        for s_idx, scen in enumerate(SCENARIOS):
            ax_act = axes[r, 2 * s_idx]
            ax_term = axes[r, 2 * s_idx + 1]
            ad = cell_action_dist(method, scen)
            tf = cell_termination_fractions(method, scen)
            if ad is not None:
                steps = ad["total_steps"]
                for a in ACTIONS:
                    ax_act.plot(steps, ad[f"action_dist_{a}"], label=a.upper(),
                                color=ACTION_COLORS[a], linewidth=1.4)
                ax_act.set_ylim(0.0, 1.0)
                ax_act.grid(alpha=0.3)
            if tf is not None:
                steps = tf["total_steps"]
                ax_term.plot(steps, tf["frac_collision"], label="collision", color=TERM_COLORS["collision"], linewidth=1.4)
                ax_term.plot(steps, tf["frac_timeout"], label="timeout", color=TERM_COLORS["timeout"], linewidth=1.4)
                ax_term.plot(steps, tf["frac_success"], label="success", color=TERM_COLORS["success"], linewidth=1.4)
                ax_term.plot(steps, tf["frac_abort"], label="abort_term", color=TERM_COLORS["abort"], linewidth=1.4, alpha=0.7)
                ax_term.set_ylim(0.0, 1.05)
                ax_term.grid(alpha=0.3)
        axes[r, 0].set_ylabel(METHOD_LABELS[method], fontsize=11)

    h_act, l_act = axes[0, 0].get_legend_handles_labels()
    h_term, l_term = axes[0, 1].get_legend_handles_labels()
    fig.legend(h_act, l_act, loc="upper left", bbox_to_anchor=(0.02, 0.985), ncol=5, fontsize=9, frameon=True, title="action")
    fig.legend(h_term, l_term, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=4, fontsize=9, frameon=True, title="termination")
    for c in range(4):
        axes[-1, c].set_xlabel("total_steps")
    fig.suptitle("Action distribution and termination cause — 1a vs 2_dense (mean across 3 seeds)", fontsize=13, y=0.995)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_path_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=180 if ext == "png" else 300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def termination_summary_table() -> pd.DataFrame:
    """Aggregate final-iteration termination fractions per cell."""
    rows = []
    for method in METHODS:
        for scen in SCENARIOS:
            tf = cell_termination_fractions(method, scen)
            if tf is None:
                continue
            tail = tf.tail(10)
            head = tf.head(10)
            rows.append({
                "method": method, "scenario": scen,
                "ep_len_start": float(head["ep_len"].mean()),
                "ep_len_end": float(tail["ep_len"].mean()),
                "frac_collision_start": float(head["frac_collision"].mean()),
                "frac_collision_end": float(tail["frac_collision"].mean()),
                "frac_timeout_start": float(head["frac_timeout"].mean()),
                "frac_timeout_end": float(tail["frac_timeout"].mean()),
                "frac_success_start": float(head["frac_success"].mean()),
                "frac_success_end": float(tail["frac_success"].mean()),
            })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1.2] generating action+termination plots…")
    plot_action_termination(OUT_DIR / "action_dist_2dense_vs_1a")
    print("[1.2] computing termination summary table…")
    summary = termination_summary_table()
    csv_path = OUT_DIR / "termination_summary.csv"
    summary.to_csv(csv_path, index=False)
    json_path = OUT_DIR / "termination_summary.json"
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    print(summary.to_string(index=False))
    print(f"\n[wrote] {csv_path}")
    print(f"[wrote] {json_path}")


if __name__ == "__main__":
    main()
