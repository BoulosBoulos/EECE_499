"""Phase 3 — calibration analysis.

Inputs:
  results/calibration/CAL_<method>_<scenario>_s<seed>/{metrics.csv, meta.json}

Outputs:
  results/calibration_analysis/
    convergence_per_run.csv         per-run convergence step (or null)
    convergence_per_method.csv      per (method, scenario): mean/max + noisy flag
    per_method_max_convergence.csv  per method: max across scenarios
    calibrated_total_steps.json     final calibrated step count + safety buffer math
    plots/
      convergence_<scenario>.{png,pdf}     multi-method comparison plot per scenario
      per_method_curves.{png,pdf}          supplementary per-method panels
      convergence_summary_table.{html}     tabular summary

Convergence criterion (active = v2 SR-primary, post Phase 3F Stage 3):
  A run is "converged at step t" iff trailing window [t-50_000, t] satisfies:
    1. mean(rolling-5 success_rate) >= V2_TAU_SR  (= 0.70)
    2. std(rolling-5 success_rate) <= V2_SIGMA_SR (= 0.10)
    3. mean(rolling-5 collision_rate) <= V2_TAU_COLL (= 0.05)

  Reasoning: v1 (Decision B) checked reward-signal stability
    1. std(rolling-5 mean_reward) / |mean(window)| < 0.05
    2. std(rolling-5 collision_rate) < 0.02 absolute
  Stage 3 found this false-negatives near-perfect policies because mean_reward
  fluctuates 5-30% relative even when SR is stable at ~1.0. v2 tracks SR directly.
  evaluate_convergence_v1 retained for retroactive comparison.

Per-method-scenario step: mean across 3 seeds; "noisy" if max/min > 1.5 → use max.
Per-method step: max across 2 scenarios.
Final calibrated_steps = ceil(1.1 * max_across_methods / 5000) * 5000.

meta.json schema (forward-looking, post-Stage-3):
  Future training runs should record `criterion_version` = "v2_sr_primary_post_stage3"
  in meta.json (the trainer scripts under experiments/pde/ are responsible for
  writing this field; this module exposes the canonical CRITERION_VERSION constant).
  Phase 3 meta.json files are NOT to be modified retroactively; the v2 verdict for
  those jobs is recorded in analysis output only (verification/phase3F_stage3_*).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- Constants per Decision B (v1 reward-stability criterion) ---------------
WINDOW_STEPS = 50_000
REWARD_STD_REL_THRESHOLD = 0.05
COLLISION_STD_ABS_THRESHOLD = 0.02
SAFETY_BUFFER = 1.1
ROUND_TO = 5_000
NOISY_RATIO_THRESHOLD = 1.5

# ---- Criterion version (v2 SR-primary, post-Stage-3) ------------------------
# Phase 3F Stage 3 diagnostic showed v1 false-negatives on near-perfect policies
# because mean_reward fluctuates above the 5% relative threshold from per-episode
# noise (timing, traffic) even when SR is essentially constant. v2 replaces the
# reward-stability check with a success-rate-primary plateau check.
CRITERION_VERSION = "v2_sr_primary_post_stage3"
V2_TAU_SR = 0.70           # mean SR threshold over trailing window
V2_SIGMA_SR = 0.10         # std SR stability tolerance (rolling-5 smoothed)
V2_TAU_COLL = 0.05         # mean collision-rate upper bound

CALIBRATION_METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]
CALIBRATION_SCENARIOS = ["1a", "2_dense"]


def evaluate_convergence_v1(metrics_df: pd.DataFrame) -> dict:
    """v1 reward-stability criterion (Decision B; pre-Stage-3 default).

    A run is converged at step S iff window [S, S+50_000] satisfies:
      1. std(rolling-5 mean_reward) / |mean(window)| < REWARD_STD_REL_THRESHOLD
      2. std(rolling-5 collision_rate) < COLLISION_STD_ABS_THRESHOLD

    Returns the canonical convergence-result dict shared with v2.
    """
    base = {
        "converged": False, "t_first": None, "t_last": None,
        "n_satisfied_evals": 0, "mean_SR_post_first": None,
        "criterion_version": "v1_reward_stability",
    }
    if metrics_df is None or len(metrics_df) < 5:
        return base
    df = metrics_df.sort_values("total_steps").reset_index(drop=True)
    total_steps = df["total_steps"].to_numpy(dtype=float)
    rewards = df["mean_reward"].to_numpy(dtype=float)
    n_eps = df["n_episodes"].to_numpy(dtype=float)
    n_eps_safe = np.where(n_eps > 0, n_eps, 1.0)
    collision_rates = df["n_collisions"].to_numpy(dtype=float) / n_eps_safe
    success_rates = df["n_successes"].to_numpy(dtype=float) / n_eps_safe

    smoothed_reward = pd.Series(rewards).rolling(window=5, min_periods=1).mean().to_numpy()
    smoothed_collision = pd.Series(collision_rates).rolling(window=5, min_periods=1).mean().to_numpy()

    last_step = total_steps[-1]
    sat = np.zeros(len(total_steps), dtype=bool)
    for i in range(len(total_steps)):
        S = total_steps[i]
        if S + WINDOW_STEPS > last_step:
            continue
        window_mask = (total_steps >= S) & (total_steps <= S + WINDOW_STEPS)
        if window_mask.sum() < 5:
            continue
        wr = smoothed_reward[window_mask]
        wc = smoothed_collision[window_mask]
        reward_mean_abs = abs(wr.mean()) + 1e-6
        reward_std_rel = float(wr.std(ddof=1) / reward_mean_abs)
        if reward_std_rel > REWARD_STD_REL_THRESHOLD:
            continue
        collision_std = float(wc.std(ddof=1))
        if collision_std > COLLISION_STD_ABS_THRESHOLD:
            continue
        sat[i] = True

    if not sat.any():
        return base
    first_idx = int(np.argmax(sat))
    last_idx = int(len(sat) - 1 - np.argmax(sat[::-1]))
    return {
        "converged": True,
        "t_first": int(total_steps[first_idx]),
        "t_last": int(total_steps[last_idx]),
        "n_satisfied_evals": int(sat.sum()),
        "mean_SR_post_first": float(success_rates[first_idx:].mean()) if first_idx < len(success_rates) else None,
        "criterion_version": "v1_reward_stability",
    }


def evaluate_convergence_v2(
    metrics_df: pd.DataFrame,
    tau_sr: float = V2_TAU_SR,
    sigma_sr: float = V2_SIGMA_SR,
    tau_coll: float = V2_TAU_COLL,
    window_steps: int = WINDOW_STEPS,
) -> dict:
    """v2 SR-primary criterion (Phase 3F Stage 3 redesign; current default).

    A run is converged at step t iff over the trailing window [t - window_steps, t]:
      1. mean(rolling-5 success_rate) >= tau_sr
      2. std(rolling-5 success_rate) <= sigma_sr
      3. mean(rolling-5 collision_rate) <= tau_coll

    Drops the v1 reward-stability check because mean_reward exhibits 5-30%
    relative noise even on near-perfect policies (per-episode randomness in
    SUMO traffic + episode-length variability), making v1 false-negative.
    """
    base = {
        "converged": False, "t_first": None, "t_last": None,
        "n_satisfied_evals": 0, "mean_SR_post_first": None,
        "criterion_version": CRITERION_VERSION,
    }
    if metrics_df is None or len(metrics_df) < 5:
        return base
    df = metrics_df.sort_values("total_steps").reset_index(drop=True)
    total_steps = df["total_steps"].to_numpy(dtype=float)
    n_eps = df["n_episodes"].to_numpy(dtype=float)
    n_eps_safe = np.where(n_eps > 0, n_eps, 1.0)
    success_rates = df["n_successes"].to_numpy(dtype=float) / n_eps_safe
    collision_rates = df["n_collisions"].to_numpy(dtype=float) / n_eps_safe

    smoothed_sr = pd.Series(success_rates).rolling(window=5, min_periods=1).mean().to_numpy()
    smoothed_cr = pd.Series(collision_rates).rolling(window=5, min_periods=1).mean().to_numpy()

    sat = np.zeros(len(total_steps), dtype=bool)
    for i in range(len(total_steps)):
        t = total_steps[i]
        # Require the trailing window to be fully within the trajectory.
        if t < window_steps:
            continue
        window_mask = (total_steps >= t - window_steps) & (total_steps <= t)
        if window_mask.sum() < 5:
            continue
        ws = smoothed_sr[window_mask]
        wc = smoothed_cr[window_mask]
        if float(ws.mean()) < tau_sr:
            continue
        if float(ws.std(ddof=1)) > sigma_sr:
            continue
        if float(wc.mean()) > tau_coll:
            continue
        sat[i] = True

    if not sat.any():
        return base
    first_idx = int(np.argmax(sat))
    last_idx = int(len(sat) - 1 - np.argmax(sat[::-1]))
    return {
        "converged": True,
        "t_first": int(total_steps[first_idx]),
        "t_last": int(total_steps[last_idx]),
        "n_satisfied_evals": int(sat.sum()),
        "mean_SR_post_first": float(success_rates[first_idx:].mean()) if first_idx < len(success_rates) else None,
        "criterion_version": CRITERION_VERSION,
    }


def detect_convergence_in_run(metrics_df: pd.DataFrame) -> Optional[int]:
    """Backward-compatible wrapper: returns t_first under the *active* criterion
    (v2 by default). Existing callers in this module that historically relied on
    v1 semantics now resolve through v2 unless they call evaluate_convergence_v1
    directly.
    """
    result = evaluate_convergence_v2(metrics_df)
    return result["t_first"] if result["converged"] else None


def _load_run(run_dir: Path) -> Optional[dict]:
    metrics_path = run_dir / "metrics.csv"
    meta_path = run_dir / "meta.json"
    if not (metrics_path.is_file() and meta_path.is_file()):
        return None
    try:
        df = pd.read_csv(metrics_path)
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"[cal-analysis] WARN: could not load {run_dir.name}: {e}")
        return None
    return {"run_dir": run_dir, "df": df, "meta": meta}


def _scan_calibration_runs(results_root: Path) -> list[dict]:
    runs = []
    if not results_root.is_dir():
        return runs
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("CAL_"):
            continue
        loaded = _load_run(d)
        if loaded is None:
            continue
        loaded["run_id"] = d.name
        runs.append(loaded)
    return runs


def analyze_calibration(results_root: str | Path) -> dict:
    results_root = Path(results_root)
    runs = _scan_calibration_runs(results_root)
    if not runs:
        return {"analysis_status": "no_runs", "n_runs": 0}

    # 1) Per-run convergence
    per_run_rows = []
    for r in runs:
        S = detect_convergence_in_run(r["df"])
        per_run_rows.append({
            "run_id": r["run_id"],
            "method": r["meta"].get("method"),
            "scenario": r["meta"].get("scenario"),
            "seed": r["meta"].get("seed"),
            "convergence_step": S,
            "converged": S is not None,
            "total_steps_actual": int(r["df"]["total_steps"].max()) if len(r["df"]) else 0,
        })
    per_run = pd.DataFrame(per_run_rows)

    non_converged = per_run[~per_run["converged"]].copy()

    # 2) Per (method, scenario) — average across seeds; flag "noisy"
    ms_rows = []
    for (method, scen), g in per_run.groupby(["method", "scenario"]):
        if g["converged"].all():
            steps = g["convergence_step"].astype(int).values
            S_min, S_max, S_mean = int(steps.min()), int(steps.max()), int(steps.mean())
            noisy = S_min > 0 and (S_max / S_min) > NOISY_RATIO_THRESHOLD
            S_to_use = S_max if noisy else S_mean
            ms_rows.append({
                "method": method, "scenario": scen,
                "S_min": S_min, "S_max": S_max, "S_mean": S_mean,
                "noisy": bool(noisy), "S_to_use": int(S_to_use),
                "all_seeds_converged": True,
            })
        else:
            ms_rows.append({
                "method": method, "scenario": scen,
                "S_min": None, "S_max": None, "S_mean": None,
                "noisy": False, "S_to_use": None,
                "all_seeds_converged": False,
            })
    ms = pd.DataFrame(ms_rows)

    # 3) Per-method (max across scenarios)
    if not non_converged.empty:
        analysis_status = "non_converged_cells_present"
        method_max_df = None
        max_across_methods = None
        calibrated_steps = None
    else:
        method_max_rows = []
        for method, g in ms.groupby("method"):
            m_max = int(g["S_to_use"].max())
            method_max_rows.append({"method": method, "convergence_step": m_max})
        method_max_df = pd.DataFrame(method_max_rows).sort_values("convergence_step", ascending=False)
        max_across_methods = int(method_max_df["convergence_step"].max())
        calibrated_steps = int(math.ceil(SAFETY_BUFFER * max_across_methods / ROUND_TO) * ROUND_TO)
        analysis_status = "complete"

    return {
        "analysis_status": analysis_status,
        "n_runs": len(runs),
        "per_run": per_run,
        "method_scenario": ms,
        "method_max": method_max_df,
        "max_across_methods": max_across_methods,
        "calibrated_steps": calibrated_steps,
        "non_converged": non_converged,
        "runs": runs,
    }


# ---- Plotting --------------------------------------------------------------
def _compute_per_method_seedmean(runs_for_method_scen, smooth_steps_eq_iters: int = 10):
    if not runs_for_method_scen:
        return None, None, None
    # Build a common iteration grid (use the shortest run as reference)
    n_iter = min(len(r["df"]) for r in runs_for_method_scen)
    if n_iter < 2:
        return None, None, None
    rewards_mat = np.stack([r["df"]["mean_reward"].iloc[:n_iter].to_numpy(dtype=float) for r in runs_for_method_scen])
    n_eps_mat = np.stack([np.where(r["df"]["n_episodes"].iloc[:n_iter] > 0,
                                   r["df"]["n_episodes"].iloc[:n_iter], 1.0).astype(float)
                          for r in runs_for_method_scen])
    n_coll_mat = np.stack([r["df"]["n_collisions"].iloc[:n_iter].to_numpy(dtype=float) for r in runs_for_method_scen])
    coll_mat = n_coll_mat / n_eps_mat
    steps = runs_for_method_scen[0]["df"]["total_steps"].iloc[:n_iter].to_numpy(dtype=float)

    # Light smoothing
    def smooth(M):
        return pd.DataFrame(M.T).rolling(window=5, min_periods=1).mean().T.to_numpy()
    rewards_mat = smooth(rewards_mat)
    coll_mat = smooth(coll_mat)
    return steps, rewards_mat, coll_mat


def plot_calibration_curves(analysis: dict, output_root: Path) -> list[Path]:
    runs = analysis["runs"]
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    try:
        from analysis.config import METHOD_COLORS, METHOD_LABELS, MATPLOTLIB_RC
        plt.rcParams.update(MATPLOTLIB_RC)
    except Exception:
        METHOD_COLORS = {m: c for m, c in zip(CALIBRATION_METHODS,
                                              ["#888888", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])}
        METHOD_LABELS = {m: m for m in CALIBRATION_METHODS}

    # Per-scenario multi-method comparison plot
    per_run_df = analysis["per_run"]
    for scen in CALIBRATION_SCENARIOS:
        fig, (ax_r, ax_c) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for method in CALIBRATION_METHODS:
            seed_runs = [r for r in runs
                         if r["meta"].get("method") == method and r["meta"].get("scenario") == scen]
            steps, rewards_mat, coll_mat = _compute_per_method_seedmean(seed_runs)
            if steps is None:
                continue
            mean_r = rewards_mat.mean(axis=0)
            std_r = rewards_mat.std(axis=0, ddof=1) if rewards_mat.shape[0] > 1 else np.zeros_like(mean_r)
            mean_c = coll_mat.mean(axis=0)
            std_c = coll_mat.std(axis=0, ddof=1) if coll_mat.shape[0] > 1 else np.zeros_like(mean_c)

            color = METHOD_COLORS.get(method, "black")
            label_steps_text = ""
            ms_row = analysis["method_scenario"][
                (analysis["method_scenario"]["method"] == method)
                & (analysis["method_scenario"]["scenario"] == scen)
            ]
            if not ms_row.empty and pd.notna(ms_row.iloc[0]["S_to_use"]):
                S_use = ms_row.iloc[0]["S_to_use"]
                label_steps_text = f" (S={int(S_use):,})"
                ax_r.axvline(S_use, color=color, linestyle="--", alpha=0.5)
                ax_c.axvline(S_use, color=color, linestyle="--", alpha=0.5)
            ax_r.plot(steps, mean_r, color=color, label=f"{METHOD_LABELS.get(method, method)}{label_steps_text}", linewidth=1.7)
            ax_r.fill_between(steps, mean_r - std_r, mean_r + std_r, color=color, alpha=0.15)
            ax_c.plot(steps, mean_c, color=color, linewidth=1.7)
            ax_c.fill_between(steps, mean_c - std_c, mean_c + std_c, color=color, alpha=0.15)

        ax_r.set_ylabel("mean_reward (smoothed; ±1 σ across seeds)")
        ax_r.set_title(f"Calibration — scenario {scen}")
        ax_r.grid(alpha=0.3); ax_r.legend(fontsize=9, ncol=2)
        ax_c.axhline(COLLISION_STD_ABS_THRESHOLD, color="grey", linestyle=":", linewidth=0.8, label="collision std threshold")
        ax_c.set_xlabel("training step")
        ax_c.set_ylabel("collision rate (smoothed; ±1 σ across seeds)")
        ax_c.grid(alpha=0.3)
        for ext in ("png", "pdf"):
            p = plots_dir / f"convergence_{scen}.{ext}"
            fig.savefig(p, dpi=200 if ext == "png" else 300, bbox_inches="tight")
            written.append(p)
        plt.close(fig)

    # Supplementary: per-method panels
    fig, axes = plt.subplots(len(CALIBRATION_METHODS), 1, figsize=(10, 2.5 * len(CALIBRATION_METHODS)),
                             sharex=True, squeeze=False)
    for i, method in enumerate(CALIBRATION_METHODS):
        ax = axes[i, 0]
        for scen in CALIBRATION_SCENARIOS:
            seed_runs = [r for r in runs
                         if r["meta"].get("method") == method and r["meta"].get("scenario") == scen]
            steps, rewards_mat, _ = _compute_per_method_seedmean(seed_runs)
            if steps is None: continue
            mean_r = rewards_mat.mean(axis=0)
            ax.plot(steps, mean_r, label=scen, linewidth=1.5)
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylabel("mean_reward")
    axes[-1, 0].set_xlabel("training step")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = plots_dir / f"per_method_curves.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else 300, bbox_inches="tight")
        written.append(p)
    plt.close(fig)

    return written


def write_outputs(analysis: dict, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    if analysis.get("analysis_status") == "no_runs":
        (output_root / "calibrated_total_steps.json").write_text(
            json.dumps({"analysis_status": "no_runs", "n_runs": 0}, indent=2)
        )
        return paths

    per_run = analysis["per_run"]
    p = output_root / "convergence_per_run.csv"
    per_run.to_csv(p, index=False)
    paths["convergence_per_run"] = str(p)

    ms = analysis["method_scenario"]
    p = output_root / "convergence_per_method.csv"
    ms.to_csv(p, index=False)
    paths["convergence_per_method"] = str(p)

    if analysis["method_max"] is not None:
        p = output_root / "per_method_max_convergence.csv"
        analysis["method_max"].to_csv(p, index=False)
        paths["per_method_max_convergence"] = str(p)

    summary = {
        "analysis_status": analysis["analysis_status"],
        "n_runs": analysis["n_runs"],
        "max_across_methods": analysis["max_across_methods"],
        "safety_buffer": SAFETY_BUFFER,
        "rounding": ROUND_TO,
        "calibrated_steps": analysis["calibrated_steps"],
        "convergence_window_steps": WINDOW_STEPS,
        "reward_std_rel_threshold": REWARD_STD_REL_THRESHOLD,
        "collision_std_abs_threshold": COLLISION_STD_ABS_THRESHOLD,
        "non_converged_run_ids": (
            analysis["non_converged"]["run_id"].tolist()
            if not analysis["non_converged"].empty else []
        ),
    }
    p = output_root / "calibrated_total_steps.json"
    p.write_text(json.dumps(summary, indent=2, default=str))
    paths["calibrated_total_steps"] = str(p)

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/calibration")
    parser.add_argument("--output_root",  default="results/calibration_analysis")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    analysis = analyze_calibration(args.results_root)
    output_root = Path(args.output_root)
    paths = write_outputs(analysis, output_root)

    print(f"[cal-analysis] {analysis['n_runs']} runs analyzed; status={analysis['analysis_status']}")
    if analysis["analysis_status"] == "complete":
        print(f"[cal-analysis] max_across_methods = {analysis['max_across_methods']}")
        print(f"[cal-analysis] calibrated_steps    = {analysis['calibrated_steps']}")
    elif analysis["analysis_status"] == "non_converged_cells_present":
        print("[cal-analysis] NON-CONVERGED cells present:")
        for rid in analysis["non_converged"]["run_id"]:
            print(f"   {rid}")
        print("[cal-analysis] STOP — calibration spec requires explicit human approval before 1M extension.")
    else:
        print("[cal-analysis] no runs found.")

    if not args.no_plots and analysis["analysis_status"] != "no_runs":
        plot_paths = plot_calibration_curves(analysis, output_root)
        print(f"[cal-analysis] wrote {len(plot_paths)} plot files to {output_root / 'plots'}")

    print(f"[cal-analysis] outputs:")
    for k, v in paths.items():
        print(f"   {k}: {v}")

    return 0 if analysis["analysis_status"] in ("complete", "no_runs") else 2


if __name__ == "__main__":
    sys.exit(main())
