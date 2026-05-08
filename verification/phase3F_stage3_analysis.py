"""Phase 3F Stage 3 — Soft-HJB drift diagnostic driver.

Reads the 6 Soft-HJB Phase 3 calibration jobs, applies the Phase 3 multi-metric
plateau convergence criterion (defined in analysis/calibration_analysis.py), and
produces:
  - per-seed convergence trajectory table (§3.1)
  - per-metric trajectory analysis (§3.2)
  - mechanism cross-correlation (§3.3)
  - bug/phenomenon/inconclusive verdict (§3.4)

Writes:
  - verification/phase3F_stage3_status.json
  - verification/phase3F_stage3_phenomenon_artifact.md (if phenomenon)
  - verification/phase3F_stage3_bug_artifact.md (if bug)
  - both, if inconclusive
  - verification/phase3F_stage3_plots/*.png
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/home/boulosboulos/Desktop/EECE_499-main")
RESULTS_ROOT = PROJECT_ROOT / "results/calibration"
PLOTS_DIR = PROJECT_ROOT / "verification/phase3F_stage3_plots"
STATUS_OUT = PROJECT_ROOT / "verification/phase3F_stage3_status.json"

# Reconstructed Phase 3 multi-metric plateau criterion (decision B in analysis/calibration_analysis.py)
WINDOW_STEPS = 50_000
REWARD_STD_REL_THRESHOLD = 0.05
COLLISION_STD_ABS_THRESHOLD = 0.02

CELLS = [
    {"name": "1a_s42",        "scenario": "1a",      "seed": 42,  "maneuver": "stem_right"},
    {"name": "1a_s123",       "scenario": "1a",      "seed": 123, "maneuver": "stem_right"},
    {"name": "1a_s456",       "scenario": "1a",      "seed": 456, "maneuver": "stem_right"},
    {"name": "2_dense_s42",   "scenario": "2_dense", "seed": 42,  "maneuver": "stem_right"},
    {"name": "2_dense_s123",  "scenario": "2_dense", "seed": 123, "maneuver": "stem_right"},
    {"name": "2_dense_s456",  "scenario": "2_dense", "seed": 456, "maneuver": "stem_right"},
]


def cell_dir(cell):
    return RESULTS_ROOT / f"CAL_soft_hjb_aux_{cell['scenario']}_s{cell['seed']}"


def load_metrics_csv(cell):
    p = cell_dir(cell) / "metrics.csv"
    if not p.exists(): return None
    df = pd.read_csv(p).sort_values("total_steps").reset_index(drop=True)
    n_eps_safe = df["n_episodes"].where(df["n_episodes"] > 0, 1.0)
    df["success_rate"] = df["n_successes"].astype(float) / n_eps_safe
    df["collision_rate"] = df["n_collisions"].astype(float) / n_eps_safe
    return df


def load_train_csv(cell):
    p = cell_dir(cell) / f"train_soft_hjb_aux_{cell['scenario']}_{cell['maneuver']}.csv"
    if not p.exists(): return None
    return pd.read_csv(p).sort_values("step").reset_index(drop=True)


def detect_convergence_at(df, S_idx):
    """Check if window starting at row S_idx satisfies the multi-metric plateau criterion."""
    total_steps = df["total_steps"].to_numpy(dtype=float)
    rewards = df["mean_reward"].to_numpy(dtype=float)
    n_eps = df["n_episodes"].to_numpy(dtype=float)
    n_eps_safe = np.where(n_eps > 0, n_eps, 1.0)
    collisions = df["n_collisions"].to_numpy(dtype=float) / n_eps_safe

    smoothed_reward = pd.Series(rewards).rolling(window=5, min_periods=1).mean().to_numpy()
    smoothed_collision = pd.Series(collisions).rolling(window=5, min_periods=1).mean().to_numpy()

    last_step = total_steps[-1]
    S = total_steps[S_idx]
    if S + WINDOW_STEPS > last_step:
        return False, "insufficient_lookahead"
    mask = (total_steps >= S) & (total_steps <= S + WINDOW_STEPS)
    if mask.sum() < 5:
        return False, "insufficient_samples"
    wr = smoothed_reward[mask]; wc = smoothed_collision[mask]
    rmean_abs = abs(wr.mean()) + 1e-6
    rstd_rel = float(wr.std(ddof=1) / rmean_abs)
    cstd = float(wc.std(ddof=1))
    if rstd_rel > REWARD_STD_REL_THRESHOLD:
        return False, f"reward_std_rel={rstd_rel:.4f}>0.05"
    if cstd > COLLISION_STD_ABS_THRESHOLD:
        return False, f"coll_std={cstd:.4f}>0.02"
    return True, "ok"


def per_seed_trajectory(cell):
    df = load_metrics_csv(cell)
    if df is None:
        return {"cell": cell["name"], "error": "no_metrics_csv"}
    n = len(df)
    satisfied_mask = []
    reasons = []
    for i in range(n):
        sat, reason = detect_convergence_at(df, i)
        satisfied_mask.append(sat); reasons.append(reason)
    sat = np.array(satisfied_mask, dtype=bool)

    if not sat.any():
        return {
            "cell": cell["name"], "scenario": cell["scenario"], "seed": cell["seed"],
            "n_iterations": n, "t_first": None, "t_last": None,
            "n_satisfied": 0, "n_after_first": 0,
            "drift_count": 0, "drift_fraction": None,
            "first_satisfaction_reason": "no_window_ever_satisfies",
        }
    first_idx = int(np.argmax(sat))  # first True
    last_idx = int(len(sat) - 1 - np.argmax(sat[::-1]))
    after_first = sat[first_idx:]
    drift_count = int((~after_first).sum())
    n_after = int(len(after_first))
    drift_frac = drift_count / max(n_after, 1)

    return {
        "cell": cell["name"], "scenario": cell["scenario"], "seed": cell["seed"],
        "n_iterations": n,
        "t_first": int(df.loc[first_idx, "total_steps"]),
        "t_last":  int(df.loc[last_idx,  "total_steps"]),
        "n_satisfied": int(sat.sum()),
        "n_after_first": n_after,
        "drift_count": drift_count,
        "drift_fraction": drift_frac,
    }


def per_metric_summary(cell):
    df = load_metrics_csv(cell)
    if df is None: return {"cell": cell["name"], "error": "no_metrics_csv"}
    train_df = load_train_csv(cell)

    summary = {"cell": cell["name"]}

    # Direction-aware peak/trough/final/gap.
    # higher_is_better: success_rate, mean_reward, soft_policy_entropy
    # lower_is_better: collision_rate, HJB residual, KL alignment, distillation gap
    metrics_higher = {
        "success_rate":   ("success_rate", "higher"),
        "mean_return":    ("mean_reward",  "higher"),
        "soft_entropy":   ("L_entropy",    "higher"),
    }
    metrics_lower = {
        "collision_rate": ("collision_rate", "lower"),
        "hjb_residual":   ("L_residual_optimality", "lower"),
        "distill_gap":    ("L_distill", "lower"),
    }
    out_metrics = {}
    for label, (col, direction) in {**metrics_higher, **metrics_lower}.items():
        if col not in df.columns: continue
        vals = df[col].astype(float).to_numpy()
        steps = df["total_steps"].astype(int).to_numpy()
        if direction == "higher":
            peak_idx = int(np.argmax(vals)); peak_val = float(vals[peak_idx])
            final_val = float(vals[-1])
            gap_signed = peak_val - final_val  # positive => degraded
        else:
            peak_idx = int(np.argmin(vals)); peak_val = float(vals[peak_idx])  # peak = best = min
            final_val = float(vals[-1])
            gap_signed = final_val - peak_val  # positive => degraded back upward
        # peak-to-trough amplitude: best - worst (always positive)
        amp = float(np.max(vals) - np.min(vals))
        out_metrics[label] = {
            "metric_column": col, "direction_better": direction,
            "peak_step": int(steps[peak_idx]), "peak_value": peak_val,
            "final_step": int(steps[-1]), "final_value": final_val,
            "peak_to_final_gap_signed": gap_signed,  # positive = degraded
            "min_value": float(np.min(vals)), "max_value": float(np.max(vals)),
            "amplitude": amp,
        }
    # actor_align_kl from train CSV (sparser eval logging)
    if train_df is not None and "actor_align_kl" in train_df.columns:
        kls = train_df["actor_align_kl"].astype(float).to_numpy()
        steps_t = train_df["step"].astype(int).to_numpy()
        if len(kls) >= 2:
            best_idx = int(np.argmin(kls))
            out_metrics["actor_align_kl"] = {
                "metric_column": "actor_align_kl", "direction_better": "lower",
                "peak_step": int(steps_t[best_idx]), "peak_value": float(kls[best_idx]),
                "final_step": int(steps_t[-1]), "final_value": float(kls[-1]),
                "peak_to_final_gap_signed": float(kls[-1] - kls[best_idx]),
                "min_value": float(np.min(kls)), "max_value": float(np.max(kls)),
                "amplitude": float(np.max(kls) - np.min(kls)),
                "n_eval_points": int(len(kls)),
            }
    summary["metrics"] = out_metrics
    return summary


def mechanism_correlations(cell):
    """Compute Pearson correlations between success-rate change and mechanism signals
    on the post-t_first window."""
    df = load_metrics_csv(cell)
    if df is None: return {"cell": cell["name"], "error": "no_metrics_csv"}

    traj = per_seed_trajectory(cell)
    t_first = traj.get("t_first")
    if t_first is None:
        return {"cell": cell["name"], "error": "no_t_first", "drift_fraction": traj.get("drift_fraction")}
    post = df[df["total_steps"] >= t_first].reset_index(drop=True)
    if len(post) < 3:
        return {"cell": cell["name"], "n_points_post_first": len(post),
                "error": "too_few_post_first_points"}

    # Compute deltas (changes between consecutive eval points) so correlations target *changes*.
    sr = post["success_rate"].astype(float).to_numpy()
    hjb = post["L_residual_optimality"].astype(float).to_numpy()
    kl_loss = post["L_distill"].astype(float).to_numpy()
    ent = post["L_entropy"].astype(float).to_numpy()

    d_sr = np.diff(sr)
    d_hjb = np.diff(hjb)
    d_kl = np.diff(kl_loss)
    d_ent = np.diff(ent)

    def safe_corr(a, b):
        if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    # Per spec hypothesis pattern:
    # - HJB residual continues *decreasing* post-t_first
    # - KL alignment loss *spikes/oscillates*
    # - Soft policy entropy *decreases* (commitment)
    # - Success rate *drops*
    return {
        "cell": cell["name"],
        "n_points_post_first": int(len(post)),
        "n_diff_pairs": int(len(d_sr)),
        # Sign convention: corr(Δsuccess, Δhjb_residual). If hypothesis holds, both decrease together → positive corr.
        # corr(Δsuccess, Δkl_loss): KL spikes while success drops → negative corr.
        # corr(Δsuccess, Δentropy): entropy drops while success drops → positive corr.
        "corr_dSR_dHJB":     safe_corr(d_sr, d_hjb),
        "corr_dSR_dL_distill": safe_corr(d_sr, d_kl),
        "corr_dSR_dEntropy": safe_corr(d_sr, d_ent),
        "post_first_HJB_trend":      "decreasing" if (hjb[-1] < hjb[0]) else "increasing",
        "post_first_L_distill_trend":"decreasing" if (kl_loss[-1] < kl_loss[0]) else "increasing",
        "post_first_entropy_trend":  "decreasing" if (ent[-1] < ent[0]) else "increasing",
        "post_first_SR_trend":       "decreasing" if (sr[-1] < sr[0]) else "increasing",
    }


def check_numerical_anomalies(cell):
    df = load_metrics_csv(cell)
    if df is None: return {"cell": cell["name"], "error": "no_metrics_csv"}
    cols_to_check = ["mean_reward", "L_residual_optimality", "L_residual_safety",
                     "L_distill", "L_entropy", "L_value", "L_policy", "L_total"]
    anomalies = {}
    for c in cols_to_check:
        if c not in df.columns: continue
        v = df[c].astype(float).to_numpy()
        anomalies[c] = {
            "n_nan": int(np.isnan(v).sum()),
            "n_inf": int(np.isinf(v).sum()),
            "max_abs": float(np.nanmax(np.abs(v))) if np.isfinite(np.nanmax(np.abs(v))) else float("inf"),
        }
    return {"cell": cell["name"], "anomalies": anomalies}


def post_first_amplitude(cell, t_first):
    df = load_metrics_csv(cell)
    if df is None or t_first is None: return None
    post = df[df["total_steps"] >= t_first]
    if len(post) < 2: return None
    sr = post["success_rate"].astype(float).to_numpy()
    return {"min": float(np.min(sr)), "max": float(np.max(sr)),
            "amplitude": float(np.max(sr) - np.min(sr)), "n": int(len(sr)),
            "post_first_final": float(sr[-1])}


def evaluate_indicators(per_seed, mechanism, anomalies, per_metric):
    bug = []
    phen = []
    diagnostics = {}

    # Per-seed t_first map for downstream filters
    seed_t_first = {s["cell"]: s.get("t_first") for s in per_seed}
    converged_cells = [c for c, tf in seed_t_first.items() if tf is not None]
    never_converged = [c for c, tf in seed_t_first.items() if tf is None]
    diagnostics["converged_cells"] = converged_cells
    diagnostics["never_converged_cells"] = never_converged

    # 1) Drift fraction (only meaningful on converged seeds)
    drift_fracs = [(s["cell"], s["drift_fraction"]) for s in per_seed if s.get("drift_fraction") is not None]
    high_drift = [(c, df) for (c, df) in drift_fracs if df > 0.5]
    low_drift = [(c, df) for (c, df) in drift_fracs if df <= 0.3]
    if high_drift:
        bug.append(f"drift_fraction>0.5 on {len(high_drift)}/{len(drift_fracs)} converged seeds: " + str(high_drift))
    if drift_fracs and len(low_drift) == len(drift_fracs):
        phen.append(f"drift_fraction<=0.3 on all {len(drift_fracs)} converged seeds")

    # 2) Catastrophic regression — restrict to cells with post-t_first window data
    cat_regs = []
    moderate_regressions = []
    for pm in per_metric:
        sr = pm.get("metrics", {}).get("success_rate")
        if not sr: continue
        cell_name = pm["cell"]
        t_first_cell = seed_t_first.get(cell_name)
        if t_first_cell is None:
            # never-converged: peak/final inappropriate for "regression" framing; flag separately
            continue
        # Use post-t_first peak/final for regression check
        post_amp = post_first_amplitude(next(c for c in CELLS if c["name"] == cell_name), t_first_cell)
        if post_amp is None: continue
        peak = post_amp["max"]; final = post_amp["post_first_final"]
        if peak > 0:
            ratio_final_to_peak = final / peak
            if ratio_final_to_peak < 0.5:
                cat_regs.append((cell_name, peak, final))
            if ratio_final_to_peak < 0.7:
                moderate_regressions.append((cell_name, peak, final))
    if cat_regs:
        bug.append(f"catastrophic_regression_post_t_first(SR<0.5*peak) on cells: {cat_regs}")
    if drift_fracs and not cat_regs and not moderate_regressions:
        phen.append("no_regression_below_0.7*peak on any converged cell (post-t_first)")

    # 3) Numerical anomalies (full trajectory)
    has_anomaly = False
    anom_details = []
    for an in anomalies:
        for c, info in an.get("anomalies", {}).items():
            if info["n_nan"] > 0 or info["n_inf"] > 0:
                has_anomaly = True
                anom_details.append((an["cell"], c, info))
    if has_anomaly:
        bug.append(f"NaN_or_inf_detected: {anom_details}")
    else:
        phen.append("no_NaN_or_inf_in_any_logged_metric")

    # 4) Correlation strength on seeds showing drift
    seeds_with_drift_set = {c for (c, df) in drift_fracs if df > 0.0}
    weak_corr_drifting = []
    strong_corr_drifting = []
    for m in mechanism:
        if m.get("error"): continue
        if m["cell"] not in seeds_with_drift_set: continue
        corrs = [abs(m.get(k) or 0) for k in ("corr_dSR_dHJB", "corr_dSR_dL_distill", "corr_dSR_dEntropy")]
        if max(corrs) < 0.3:
            weak_corr_drifting.append((m["cell"], [m.get(k) for k in ("corr_dSR_dHJB", "corr_dSR_dL_distill", "corr_dSR_dEntropy")]))
        if max(corrs) >= 0.4:
            strong_corr_drifting.append((m["cell"], [m.get(k) for k in ("corr_dSR_dHJB", "corr_dSR_dL_distill", "corr_dSR_dEntropy")]))
    if weak_corr_drifting:
        bug.append(f"drift_uncorrelated_with_mechanism (|rho|<0.3 on all 3 signals) on drifting seeds: {weak_corr_drifting}")
    if strong_corr_drifting and len(strong_corr_drifting) > len(seeds_with_drift_set) / 2:
        phen.append(f"drift_correlates_with_mechanism (|rho|>=0.4) on {len(strong_corr_drifting)}/{len(seeds_with_drift_set)} drifting seeds")

    # 5) Drift amplitude — POST-t_first window only
    high_amp_cells = []
    bounded_amp_cells = []
    for cell_name, t_first_cell in seed_t_first.items():
        if t_first_cell is None: continue
        cell_obj = next(c for c in CELLS if c["name"] == cell_name)
        post = post_first_amplitude(cell_obj, t_first_cell)
        if post is None: continue
        amp = post["amplitude"]; mx = post["max"]
        if mx > 0:
            rel_amp = amp / mx
            if rel_amp > 0.3:
                high_amp_cells.append((cell_name, rel_amp))
            if rel_amp < 0.2:
                bounded_amp_cells.append((cell_name, rel_amp))
        else:
            bounded_amp_cells.append((cell_name, 0.0))
    if high_amp_cells:
        bug.append(f"high_amplitude_oscillation_post_t_first(>30%) on cells: {high_amp_cells}")
    if bounded_amp_cells and not high_amp_cells:
        phen.append(f"bounded_amplitude_post_t_first(<20%) on {len(bounded_amp_cells)} cells")

    diagnostics["weak_corr_drifting_detail"] = weak_corr_drifting
    diagnostics["strong_corr_drifting_detail"] = strong_corr_drifting
    diagnostics["catastrophic_regression_detail"] = cat_regs
    diagnostics["high_amplitude_post_t_first_detail"] = high_amp_cells
    diagnostics["bounded_amplitude_post_t_first_detail"] = bounded_amp_cells

    return {"bug_indicators_fired": bug, "phenomenon_indicators_fired": phen,
            "indicator_diagnostics": diagnostics}


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Data completeness audit
    completeness = {}
    for c in CELLS:
        d = cell_dir(c)
        completeness[c["name"]] = {
            "metrics_csv": (d / "metrics.csv").exists(),
            "eval_metrics_csv": (d / "eval_metrics.csv").exists(),
            "failures_csv": (d / "failures.csv").exists(),
            "timings_csv": (d / "timings.csv").exists(),
            "meta_json": (d / "meta.json").exists(),
            "train_soft_hjb_csv": (d / f"train_soft_hjb_aux_{c['scenario']}_{c['maneuver']}.csv").exists(),
        }

    per_seed = [per_seed_trajectory(c) for c in CELLS]
    per_metric = [per_metric_summary(c) for c in CELLS]
    mechanism = [mechanism_correlations(c) for c in CELLS]
    anomalies = [check_numerical_anomalies(c) for c in CELLS]

    indicators = evaluate_indicators(per_seed, mechanism, anomalies, per_metric)
    n_bug = len(indicators["bug_indicators_fired"])
    n_phen = len(indicators["phenomenon_indicators_fired"])
    # Verdict logic per spec §3.4 + user directive "do not auto-classify edge cases":
    #   - "any bug → bug" applies cleanly only when phenomenon indicators are essentially absent;
    #     when phenomenon indicators *also* fire substantively, this is a mixed-signal edge case
    #     and the user has explicitly directed inconclusive verdict for that situation.
    if n_bug >= 1 and n_phen <= 1:
        verdict = "bug"
    elif n_phen >= 4 and n_bug == 0:
        verdict = "phenomenon"
    elif n_bug == 0 and n_phen == 0:
        verdict = "inconclusive"
    elif n_bug >= 1 and n_phen >= 2:
        # Mixed: bug AND phenomenon indicators both fire → edge case → inconclusive (per user directive)
        verdict = "inconclusive"
    else:
        verdict = "inconclusive"

    # Step 11 implication
    if verdict == "phenomenon":
        # Compute W = smallest W s.t. W consecutive evals satisfying => 80% of remaining satisfy
        W_proposal = compute_W_proposal(per_seed)
        step11 = f"first_satisfying_window_W={W_proposal}"
    elif verdict == "bug":
        step11 = "apply_fix_and_reverify"
    else:
        step11 = "requires_user_direction"

    out = {
        "stage": "3F-Stage3",
        "name": "Soft-HJB drift investigation",
        "status": "complete" if verdict != "inconclusive" else "inconclusive",
        "verdict": verdict,
        "data_sources_used": [str(cell_dir(c)) for c in CELLS],
        "data_completeness": completeness,
        "convergence_criterion_phase3": {
            "source": "analysis/calibration_analysis.py:detect_convergence_in_run (Decision B)",
            "window_steps": WINDOW_STEPS,
            "rolling_window": 5,
            "reward_std_rel_threshold": REWARD_STD_REL_THRESHOLD,
            "collision_std_abs_threshold": COLLISION_STD_ABS_THRESHOLD,
            "metric_columns": ["mean_reward", "n_collisions/n_episodes"],
        },
        "per_seed_table": per_seed,
        "metric_trajectory_summary": per_metric,
        "mechanism_correlations": mechanism,
        "numerical_anomalies": anomalies,
        "bug_indicators_fired": indicators["bug_indicators_fired"],
        "phenomenon_indicators_fired": indicators["phenomenon_indicators_fired"],
        "branch_artifact_path": branch_artifact_path(verdict),
        "step_11_implication": step11,
        "confidence_caveats": [
            "n=6 Soft-HJB jobs is statistically limited; cannot rule out subtle bugs that manifest only on unobserved seeds or cells",
            "single 500k training run per job (no within-run replication)",
            "eval_metrics.csv, failures.csv, timings.csv MISSING for all 6 jobs; metrics.csv (per-iteration training) and train_soft_hjb_aux_*.csv (per-eval) provide the trajectory data; KL alignment is only available at eval points (~13/job) not per-iteration",
            "Step 11's 36-job × 6-seed run will provide post-hoc validation",
        ],
        "references": {
            "convergence_criterion_source": "analysis/calibration_analysis.py:18-24, 56-96",
            "soft_hjb_config": "configs/pde/soft_hjb_aux.yaml",
            "soft_hjb_agent":  "models/pde/soft_hjb_aux_agent.py",
        },
    }

    STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[stage3] wrote {STATUS_OUT}")

    make_plots(per_seed, verdict)
    print(f"[stage3] plots in {PLOTS_DIR}")
    print(f"[stage3] verdict: {verdict}")
    print(f"[stage3] step_11_implication: {step11}")
    return out


def compute_W_proposal(per_seed):
    """Smallest W such that 'criterion holds for W consecutive evals' implies
    'criterion holds for >= 80% of remaining evals' across all 6 jobs."""
    candidates = []
    for s in per_seed:
        if s.get("t_first") is None:
            candidates.append(None); continue
        df_metrics = load_metrics_csv({"scenario": s["scenario"], "seed": s["seed"], "name": s["cell"], "maneuver": "stem_right"})
        if df_metrics is None:
            candidates.append(None); continue
        sat = []
        for i in range(len(df_metrics)):
            ok, _ = detect_convergence_at(df_metrics, i)
            sat.append(ok)
        sat = np.array(sat, dtype=bool)
        n = len(sat)
        # For each W in [3, 10], find first index of W consecutive Trues, then check what
        # fraction of subsequent evals are satisfied.
        per_W = {}
        for W in range(3, 11):
            window_first = None
            for i in range(0, n - W + 1):
                if sat[i:i+W].all():
                    window_first = i; break
            if window_first is None:
                per_W[W] = None
                continue
            after = sat[window_first + W:]
            frac = float(after.mean()) if len(after) > 0 else 1.0
            per_W[W] = frac
        candidates.append(per_W)
    # Cross-job: smallest W such that frac >= 0.8 on every job that converged
    valid = [c for c in candidates if c is not None]
    if not valid:
        return 3
    for W in range(3, 11):
        fracs = [c.get(W) for c in valid]
        if all((f is not None) and f >= 0.8 for f in fracs):
            return W
    return 10  # fallback


def branch_artifact_path(verdict):
    if verdict == "phenomenon":
        return "verification/phase3F_stage3_phenomenon_artifact.md"
    if verdict == "bug":
        return "verification/phase3F_stage3_bug_artifact.md"
    return "both verification/phase3F_stage3_phenomenon_artifact.md and verification/phase3F_stage3_bug_artifact.md"


def make_plots(per_seed, verdict):
    # Per-seed 6-panel: success_rate, collision_rate, mean_return, HJB residual, L_distill, L_entropy
    for c in CELLS:
        df = load_metrics_csv(c)
        if df is None: continue
        traj = next((s for s in per_seed if s.get("cell") == c["name"]), None)
        t_first = traj.get("t_first") if traj else None

        fig, axes = plt.subplots(3, 2, figsize=(11, 9))
        steps = df["total_steps"].astype(float).to_numpy()
        plots = [
            ("success_rate",          "Success rate"),
            ("collision_rate",        "Collision rate"),
            ("mean_reward",           "Mean return"),
            ("L_residual_optimality", "HJB residual (L_residual_optimality)"),
            ("L_distill",             "Distillation loss (L_distill = MSE V_PPO vs U_aux)"),
            ("L_entropy",             "Soft policy entropy"),
        ]
        for ax, (col, label) in zip(axes.ravel(), plots):
            if col not in df.columns:
                ax.text(0.5, 0.5, f"{col} missing", ha='center', va='center'); ax.set_axis_off(); continue
            ax.plot(steps, df[col].astype(float).to_numpy(), lw=1.0)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("training step")
            ax.grid(alpha=0.3)
            if t_first is not None:
                ax.axvline(t_first, color='r', linestyle='--', alpha=0.6, label=f't_first={t_first}')
                ax.legend(fontsize=8, loc='best')
        fig.suptitle(f"Soft-HJB Phase 3 — {c['name']}", fontsize=12)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"per_seed_{c['name']}.png", dpi=110)
        plt.close(fig)

    # t_first vs seed (1a and 2_dense subplots)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, scen in zip(axes, ["1a", "2_dense"]):
        seeds = []; tfs = []
        for s in per_seed:
            if s.get("scenario") == scen:
                seeds.append(s.get("seed"))
                tfs.append(s.get("t_first") or 0)
        ax.bar([str(x) for x in seeds], tfs, color='C0')
        ax.set_title(f"t_first — scenario {scen}")
        ax.set_xlabel("seed"); ax.set_ylabel("t_first (steps)")
        ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "agg_t_first_per_seed.png", dpi=110); plt.close(fig)

    # Drift fraction per seed
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [s["cell"] for s in per_seed]
    fracs = [s.get("drift_fraction") if s.get("drift_fraction") is not None else 0 for s in per_seed]
    ax.bar(names, fracs, color='C3')
    ax.axhline(0.3, color='g', linestyle='--', alpha=0.6, label='phenomenon threshold (≤0.3)')
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.6, label='bug threshold (>0.5)')
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel("drift fraction"); ax.set_title("Drift fraction per seed")
    ax.legend(loc='best'); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "agg_drift_fraction.png", dpi=110); plt.close(fig)

    # Peak-to-final gap distribution across all 6 (success_rate)
    gaps = []
    for c in CELLS:
        df = load_metrics_csv(c)
        if df is None: continue
        sr = df["success_rate"].astype(float).to_numpy()
        if len(sr) < 2: continue
        gaps.append((c["name"], float(np.max(sr) - sr[-1])))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([g[0] for g in gaps], [g[1] for g in gaps], color='C4')
    ax.set_xticklabels([g[0] for g in gaps], rotation=30, ha='right')
    ax.set_ylabel("peak − final success rate")
    ax.set_title("Peak-to-final success-rate gap (positive = degradation)")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "agg_peak_to_final_gap.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
