"""Phase 3F Step 11 — post-run analysis.

Reads results/step11_calibration/STEP11_*/{metrics.csv, meta.json}, applies
v2 convergence criterion to every job, builds per-method-cell pass table,
applies §4.1 routing, runs §4.2 anti-pattern detection, compares against
Phase 3 v2-retroactive results, generates plots, writes status JSON.

Run AFTER orchestrator finishes (or on partial data with --partial).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/home/boulosboulos/Desktop/EECE_499-main")
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.calibration_analysis import (
    evaluate_convergence_v1, evaluate_convergence_v2,
    CRITERION_VERSION, V2_TAU_SR, V2_SIGMA_SR, V2_TAU_COLL, WINDOW_STEPS,
)

STEP11_ROOT = PROJECT_ROOT / "results/step11_calibration"
STATUS_OUT = PROJECT_ROOT / "verification/phase3F_step11_status.json"
PLOTS_DIR = PROJECT_ROOT / "verification/phase3F_step11_plots"
PHASE3_RETRO_PATH = PROJECT_ROOT / "verification/phase3F_stage3_criterion_update_status.json"

METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]
CELL_PAIRS = [("1a", "stem_right"), ("2_dense", "right_left")]
SEEDS = [42, 123, 456]


def job_dir(method, scenario, maneuver, seed):
    return STEP11_ROOT / f"STEP11_{method}_{scenario}_{maneuver}_s{seed}"


def load_metrics(method, scenario, maneuver, seed):
    p = job_dir(method, scenario, maneuver, seed) / "metrics.csv"
    if not p.exists(): return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def load_meta(method, scenario, maneuver, seed):
    p = job_dir(method, scenario, maneuver, seed) / "meta.json"
    if not p.exists(): return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def per_job_result(method, scenario, maneuver, seed):
    df = load_metrics(method, scenario, maneuver, seed)
    if df is None:
        return {"method": method, "scenario": scenario, "maneuver": maneuver, "seed": seed,
                "v2_converged": None, "t_first": None, "mean_SR_post_first": None,
                "n_failures_logged": 0, "error": "no_metrics_csv"}
    v2 = evaluate_convergence_v2(df)
    # Anti-pattern checks per §4.2
    anomalies = {}
    cols_to_check = ["mean_reward", "L_residual_optimality", "L_residual_safety",
                     "L_distill", "L_entropy", "L_value", "L_policy", "L_total"]
    for c in cols_to_check:
        if c in df.columns:
            v = df[c].astype(float).to_numpy()
            n_nan = int(np.isnan(v).sum())
            n_inf = int(np.isinf(v).sum())
            anomalies[c] = {"n_nan": n_nan, "n_inf": n_inf}

    # success rate trajectory + post-t_first window
    n_eps = df["n_episodes"].astype(float).to_numpy()
    n_eps_safe = np.where(n_eps > 0, n_eps, 1.0)
    sr = df["n_successes"].astype(float).to_numpy() / n_eps_safe
    cr = df["n_collisions"].astype(float).to_numpy() / n_eps_safe
    smoothed_sr = pd.Series(sr).rolling(5, min_periods=1).mean().to_numpy()
    total_steps = df["total_steps"].astype(float).to_numpy()

    sr_peak = float(smoothed_sr.max()) if len(smoothed_sr) else 0.0

    # Residual divergence (PDE methods only)
    residual_col = None
    if method in ("hjb_aux", "soft_hjb_aux", "cbf_aux", "fusion_aux"):
        residual_col = "L_residual_optimality"
    elif method == "eikonal_aux":
        # Eikonal under 7D state uses different metric layout. Try a few candidates.
        for cand in ["L_residual_optimality", "L_residual_eik", "L_eik"]:
            if cand in df.columns:
                residual_col = cand; break
    residual_diverged = False
    residual_head = residual_tail = None
    if residual_col and residual_col in df.columns:
        rv = df[residual_col].astype(float).to_numpy()
        rv = rv[np.isfinite(rv)]
        if len(rv) >= 10:
            head = float(rv[:max(1, len(rv)//10)].mean())
            tail = float(rv[-max(1, len(rv)//10):].mean())
            residual_head = head; residual_tail = tail
            if abs(head) > 1e-9 and abs(tail) > 10 * abs(head):
                residual_diverged = True

    catastrophic = False
    sr_post_min = None; sr_post_peak = None
    if v2["converged"] and v2["t_first"] is not None:
        post = df[df["total_steps"] >= v2["t_first"]]
        if len(post) >= 2:
            n_eps_post = post["n_episodes"].astype(float).to_numpy()
            n_eps_post_safe = np.where(n_eps_post > 0, n_eps_post, 1.0)
            sr_post = post["n_successes"].astype(float).to_numpy() / n_eps_post_safe
            sr_post_smoothed = pd.Series(sr_post).rolling(5, min_periods=1).mean().to_numpy()
            sr_post_min = float(np.min(sr_post_smoothed))
            sr_post_peak = float(np.max(sr_post_smoothed))
            if sr_post_peak > 0 and sr_post_min < 0.3 * sr_post_peak:
                catastrophic = True

    # σ_SR at end of training (over trailing 50k smoothed-SR window)
    sigma_sr_at_end = None
    last_step = float(total_steps[-1]) if len(total_steps) else 0.0
    if last_step >= WINDOW_STEPS:
        mask = (total_steps >= last_step - WINDOW_STEPS) & (total_steps <= last_step)
        if mask.sum() >= 5:
            ws = smoothed_sr[mask]
            sigma_sr_at_end = float(np.std(ws, ddof=1))
    high_sigma_at_end = (sigma_sr_at_end is not None and sigma_sr_at_end > 0.20)

    return {
        "method": method, "scenario": scenario, "maneuver": maneuver, "seed": seed,
        "v2_converged": v2["converged"], "t_first": v2["t_first"], "t_last": v2["t_last"],
        "n_satisfied_evals": v2["n_satisfied_evals"],
        "mean_SR_post_first": v2["mean_SR_post_first"],
        "sigma_SR_at_end": sigma_sr_at_end,
        "anti_pattern_high_sigma_at_end": high_sigma_at_end,
        "anti_pattern_catastrophic_post_t_first": catastrophic,
        "anti_pattern_residual_diverged": residual_diverged,
        "residual_col_used": residual_col, "residual_head": residual_head, "residual_tail": residual_tail,
        "anomalies": anomalies,
        "any_NaN_inf": any((info["n_nan"] > 0 or info["n_inf"] > 0) for info in anomalies.values()),
        "n_iterations": int(len(df)),
    }


def per_method_cell_pass_table(per_job):
    table = {}
    for method in METHODS:
        for (scen, man) in CELL_PAIRS:
            cell_key = f"{method}_x_{scen}_{man}"
            seeds_results = [r for r in per_job
                             if r["method"] == method and r["scenario"] == scen and r["maneuver"] == man]
            n_pass = sum(1 for r in seeds_results if r.get("v2_converged") is True)
            n_total = len(seeds_results)
            verdict = "PASS" if n_pass >= 2 else "FAIL"
            table[cell_key] = f"{n_pass}/{n_total} {verdict}"
    return table


def aggregate_outcome(table):
    # Count 1a methods passing
    n_1a_pass = sum(1 for k, v in table.items() if "_x_1a_stem_right" in k and "PASS" in v)
    # Count 2_dense methods passing
    n_2d_pass = sum(1 for k, v in table.items() if "_x_2_dense_right_left" in k and "PASS" in v)
    if n_1a_pass == 6:
        agg = "all_pass_1a"
    elif n_1a_pass == 5:
        agg = "partial_pass_1a_5_of_6"
    elif n_1a_pass <= 4:
        agg = "systemic_fail_1a"
    else:
        agg = "indeterminate"
    return {"aggregate_outcome": agg, "n_1a_methods_pass": n_1a_pass,
            "n_2dense_methods_pass": n_2d_pass}


def comparison_vs_phase3(per_job):
    if not PHASE3_RETRO_PATH.exists():
        return {"available": False, "reason": "phase3F_stage3_criterion_update_status.json not found"}
    with open(PHASE3_RETRO_PATH) as f:
        retro = json.load(f)
    phase3_rows = retro.get("phase3_retroactive_table", [])
    # Phase 3 cells were (1a, stem_right) and (2_dense, stem_right). Step 11 is (1a, stem_right) and (2_dense, right_left).
    # Comparable pairs: same (method, scenario) on 1a only.
    rows = []
    material_diffs = []
    for method in METHODS:
        for (scen, man) in CELL_PAIRS:
            step11 = [r for r in per_job if r["method"] == method and r["scenario"] == scen and r["maneuver"] == man]
            n_step11_pass = sum(1 for r in step11 if r.get("v2_converged") is True)
            n_step11_total = len(step11)
            phase3_match = [r for r in phase3_rows if r["method"] == method and r["scenario"] == scen]
            n_phase3_pass = sum(1 for r in phase3_match if r.get("v2_converged") is True)
            n_phase3_total = len(phase3_match)
            note = ""
            if scen == "2_dense":
                note = "Phase 3 used 2_dense×stem_right; Step 11 uses 2_dense×right_left — different deployment cell, comparison not directly meaningful"
            row = {
                "method": method, "cell": f"{scen}_{man}",
                "phase3_v2_pass": f"{n_phase3_pass}/{n_phase3_total}" if n_phase3_total else "n/a",
                "step11_v2_pass": f"{n_step11_pass}/{n_step11_total}",
                "comparable": (scen == "1a"),
                "note": note,
            }
            rows.append(row)
            # Material difference: phase3 v2 ≥ 2/3 but step 11 < 2/3 (or vice versa) — only meaningful for comparable cells (1a)
            if scen == "1a" and n_phase3_total >= 3 and n_step11_total >= 3:
                p3_pass2 = (n_phase3_pass >= 2)
                s11_pass2 = (n_step11_pass >= 2)
                if p3_pass2 != s11_pass2:
                    material_diffs.append({**row, "type": "phase3_pass_vs_step11_diff"})
    return {"available": True, "rows": rows, "material_differences": material_diffs}


def make_plots(per_job):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # 12 success-rate plots, 12 collision-rate, 12 residual (where applicable)
    for method in METHODS:
        for (scen, man) in CELL_PAIRS:
            fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
            for seed in SEEDS:
                df = load_metrics(method, scen, man, seed)
                if df is None: continue
                ts = df["total_steps"].astype(float).to_numpy()
                n_eps = df["n_episodes"].astype(float).to_numpy()
                n_eps_safe = np.where(n_eps > 0, n_eps, 1.0)
                sr = df["n_successes"].astype(float).to_numpy() / n_eps_safe
                cr = df["n_collisions"].astype(float).to_numpy() / n_eps_safe
                axes[0].plot(ts, sr, alpha=0.7, label=f"s{seed}")
                axes[1].plot(ts, cr, alpha=0.7, label=f"s{seed}")
                # residual on axes[2]
                col = "L_residual_optimality" if method != "drppo" else None
                if col and col in df.columns:
                    rv = df[col].astype(float).to_numpy()
                    axes[2].plot(ts, rv, alpha=0.7, label=f"s{seed}")
            axes[0].axhline(V2_TAU_SR, color='g', linestyle='--', alpha=0.4, label=f'τ_SR={V2_TAU_SR}')
            axes[0].set_title(f"Success rate — {method} {scen}×{man}")
            axes[0].set_xlabel("step"); axes[0].set_ylabel("SR"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
            axes[1].axhline(V2_TAU_COLL, color='r', linestyle='--', alpha=0.4, label=f'τ_coll={V2_TAU_COLL}')
            axes[1].set_title(f"Collision rate — {method} {scen}×{man}")
            axes[1].set_xlabel("step"); axes[1].set_ylabel("CR"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
            axes[2].set_title(f"PDE residual — {method} {scen}×{man}")
            axes[2].set_xlabel("step"); axes[2].set_ylabel("L_residual_optimality"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"per_cell_{method}_{scen}_{man}.png", dpi=110)
            plt.close(fig)

    # Aggregate t_first bar chart
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = []
    means = []; mins = []; maxs = []
    for method in METHODS:
        for (scen, man) in CELL_PAIRS:
            tfs = [r["t_first"] for r in per_job
                   if r["method"] == method and r["scenario"] == scen and r["maneuver"] == man and r["t_first"] is not None]
            label = f"{method[:6]}\n{scen}×{man[:5]}"
            if tfs:
                labels.append(label); means.append(np.mean(tfs)); mins.append(min(tfs)); maxs.append(max(tfs))
            else:
                labels.append(label); means.append(0); mins.append(0); maxs.append(0)
    xs = np.arange(len(labels))
    err = [[m - lo for m, lo in zip(means, mins)], [hi - m for hi, m in zip(maxs, means)]]
    ax.bar(xs, means, color='C0')
    ax.errorbar(xs, means, yerr=err, fmt='none', color='k', capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("v2 t_first (steps)")
    ax.set_title("Step 11 — t_first per method × cell (3 seeds, error bars)")
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "agg_t_first.png", dpi=110); plt.close(fig)

    # σ_SR-at-end bar chart
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = []; sigs = []
    for method in METHODS:
        for (scen, man) in CELL_PAIRS:
            for seed in SEEDS:
                rs = [r for r in per_job if r["method"] == method and r["scenario"] == scen and r["maneuver"] == man and r["seed"] == seed]
                if not rs: continue
                r = rs[0]
                if r.get("sigma_SR_at_end") is not None:
                    labels.append(f"{method[:5]}\n{scen[:4]}_{seed}")
                    sigs.append(r["sigma_SR_at_end"])
    xs = np.arange(len(labels))
    cols = ['C3' if s > 0.20 else 'C0' for s in sigs]
    ax.bar(xs, sigs, color=cols)
    ax.axhline(0.20, color='r', linestyle='--', alpha=0.6, label='σ_SR > 0.20 anti-pattern')
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("σ_SR over trailing 50k at end of training")
    ax.set_title("Step 11 — end-of-training σ_SR per job (red = anti-pattern)")
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "agg_sigma_SR_end.png", dpi=110); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true",
                        help="Allow analysis on incomplete runs (some metrics.csv may be missing or short)")
    args = parser.parse_args()

    per_job = []
    for method in METHODS:
        for (scen, man) in CELL_PAIRS:
            for seed in SEEDS:
                per_job.append(per_job_result(method, scen, man, seed))

    table = per_method_cell_pass_table(per_job)
    agg = aggregate_outcome(table)
    comparison = comparison_vs_phase3(per_job)

    # Anti-patterns aggregated
    anti_patterns = []
    for r in per_job:
        if r.get("any_NaN_inf"):
            anti_patterns.append({"job": f"{r['method']} {r['scenario']}×{r['maneuver']} s{r['seed']}", "type": "NaN_or_inf"})
        if r.get("anti_pattern_catastrophic_post_t_first"):
            anti_patterns.append({"job": f"{r['method']} {r['scenario']}×{r['maneuver']} s{r['seed']}", "type": "catastrophic_post_t_first"})
        if r.get("anti_pattern_residual_diverged"):
            anti_patterns.append({"job": f"{r['method']} {r['scenario']}×{r['maneuver']} s{r['seed']}",
                                  "type": "residual_diverged",
                                  "head": r.get("residual_head"), "tail": r.get("residual_tail"),
                                  "col": r.get("residual_col_used")})
        if r.get("anti_pattern_high_sigma_at_end"):
            anti_patterns.append({"job": f"{r['method']} {r['scenario']}×{r['maneuver']} s{r['seed']}",
                                  "type": "sigma_SR_at_end>0.20",
                                  "value": r.get("sigma_SR_at_end")})

    # Wall time
    summary_path = STEP11_ROOT / "step11_launch_summary.json"
    wall_h = None; jobs_completed = 0; jobs_failed = 0
    if summary_path.exists():
        try:
            launch_sum = json.loads(summary_path.read_text())
            s = launch_sum.get("summary", {})
            wall_h = s.get("wall_time_h"); jobs_completed = s.get("completed", 0); jobs_failed = s.get("failed", 0)
        except Exception: pass

    # Recommended next-step per §4.1 routing
    if agg["aggregate_outcome"] == "all_pass_1a":
        next_step = "step_12_and_phase_4"
    elif agg["aggregate_outcome"] == "partial_pass_1a_5_of_6":
        next_step = "investigate_5_of_6_method_failure"
    elif agg["aggregate_outcome"] == "systemic_fail_1a":
        next_step = "halt_systemic_escalate"
    else:
        next_step = "requires_user_direction"

    out = {
        "phase": "3F-Step11", "name": "Full 36-job re-calibration at 500k",
        "status": "complete" if (jobs_completed + jobs_failed) >= 30 else "partial_or_running",
        "criterion_version": CRITERION_VERSION,
        "criterion_thresholds": {"tau_sr": V2_TAU_SR, "sigma_sr": V2_SIGMA_SR, "tau_coll": V2_TAU_COLL, "window_steps": WINDOW_STEPS},
        "wall_time_hours": wall_h,
        "jobs_completed": jobs_completed,
        "jobs_failed": jobs_failed,
        "per_job_results": per_job,
        "method_cell_pass_table": table,
        **agg,
        "anti_patterns_detected": anti_patterns,
        "n_anti_patterns": len(anti_patterns),
        "comparisons_vs_phase3_v2_retroactive": comparison,
        "recommended_next_step": next_step,
    }
    STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[step11_analysis] wrote {STATUS_OUT}")
    make_plots(per_job)
    print(f"[step11_analysis] plots in {PLOTS_DIR}")
    print(f"[step11_analysis] aggregate outcome: {agg['aggregate_outcome']}")
    print(f"[step11_analysis] recommended next step: {next_step}")
    return out


if __name__ == "__main__":
    main()
