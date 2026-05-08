"""Phase 3F Stage 3 — criterion-v2 retroactive analysis driver.

Calibrates v2 thresholds against the 1a Soft-HJB seeds and re-analyzes all
36 Phase 3 calibration jobs under both v1 and v2.

Writes:
  - verification/phase3F_stage3_criterion_update_status.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/boulosboulos/Desktop/EECE_499-main")
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.calibration_analysis import (
    evaluate_convergence_v1,
    evaluate_convergence_v2,
    CRITERION_VERSION,
    V2_TAU_SR, V2_SIGMA_SR, V2_TAU_COLL, WINDOW_STEPS,
    CALIBRATION_METHODS, CALIBRATION_SCENARIOS,
)

RESULTS_ROOT = PROJECT_ROOT / "results/calibration"
STATUS_OUT = PROJECT_ROOT / "verification/phase3F_stage3_criterion_update_status.json"

SEEDS = [42, 123, 456]


def load_metrics(method, scenario, seed):
    p = RESULTS_ROOT / f"CAL_{method}_{scenario}_s{seed}" / "metrics.csv"
    if not p.exists(): return None
    return pd.read_csv(p)


def calibration_diagnostic(tau_sr: float, sigma_sr: float, tau_coll: float):
    """Apply the proposed v2 thresholds to 1a Soft-HJB seeds; return per-seed result."""
    out = []
    for seed in SEEDS:
        df = load_metrics("soft_hjb_aux", "1a", seed)
        if df is None:
            out.append({"seed": seed, "error": "no_metrics"}); continue
        r = evaluate_convergence_v2(df, tau_sr=tau_sr, sigma_sr=sigma_sr, tau_coll=tau_coll)
        # Also collect raw SR / collision-rate trajectory diagnostics over the trailing 50k window
        # at the latest available step, to inform calibration if v2 returns False.
        df_sorted = df.sort_values("total_steps").reset_index(drop=True)
        ts = df_sorted["total_steps"].to_numpy(dtype=float)
        n_eps = df_sorted["n_episodes"].to_numpy(dtype=float)
        n_eps_safe = np.where(n_eps > 0, n_eps, 1.0)
        sr = df_sorted["n_successes"].to_numpy(dtype=float) / n_eps_safe
        cr = df_sorted["n_collisions"].to_numpy(dtype=float) / n_eps_safe
        smoothed_sr = pd.Series(sr).rolling(5, min_periods=1).mean().to_numpy()
        smoothed_cr = pd.Series(cr).rolling(5, min_periods=1).mean().to_numpy()

        # Find earliest t where smoothed SR first crosses tau_sr
        first_cross_idx = int(np.argmax(smoothed_sr >= tau_sr)) if (smoothed_sr >= tau_sr).any() else None
        first_cross_step = int(ts[first_cross_idx]) if first_cross_idx is not None else None

        # Compute the trailing-50k window stats at every t and report the windows that fail
        win_stats = []
        for i in range(len(ts)):
            if ts[i] < WINDOW_STEPS: continue
            mask = (ts >= ts[i] - WINDOW_STEPS) & (ts <= ts[i])
            if mask.sum() < 5: continue
            ws = smoothed_sr[mask]; wc = smoothed_cr[mask]
            win_stats.append({
                "t": int(ts[i]),
                "mean_SR_window": float(ws.mean()),
                "std_SR_window": float(ws.std(ddof=1)),
                "mean_coll_window": float(wc.mean()),
                "passes_tau_sr": bool(ws.mean() >= tau_sr),
                "passes_sigma_sr": bool(ws.std(ddof=1) <= sigma_sr),
                "passes_tau_coll": bool(wc.mean() <= tau_coll),
            })
        out.append({
            "seed": seed,
            "v2_result": r,
            "first_cross_smoothed_sr_threshold_step": first_cross_step,
            "n_windows_evaluated": len(win_stats),
            "n_windows_pass_tau_sr": sum(1 for w in win_stats if w["passes_tau_sr"]),
            "n_windows_pass_sigma_sr": sum(1 for w in win_stats if w["passes_sigma_sr"]),
            "n_windows_pass_tau_coll": sum(1 for w in win_stats if w["passes_tau_coll"]),
            "max_std_SR_in_pass_tau_sr_windows": max(
                (w["std_SR_window"] for w in win_stats if w["passes_tau_sr"]), default=None),
            "n_windows_all_three_pass": sum(1 for w in win_stats if (w["passes_tau_sr"] and w["passes_sigma_sr"] and w["passes_tau_coll"])),
        })
    return out


def calibrate():
    """Iteratively calibrate thresholds. Hard limits: tau_sr >= 0.65, sigma_sr <= 0.15."""
    grid = [
        (0.70, 0.10, 0.05),  # spec proposed
        (0.70, 0.12, 0.05),
        (0.70, 0.15, 0.05),
        (0.65, 0.10, 0.05),
        (0.65, 0.15, 0.05),
    ]
    log = []
    chosen = None
    for (ts, ss, tc) in grid:
        diag = calibration_diagnostic(ts, ss, tc)
        all_three_converged = all(
            d.get("v2_result", {}).get("converged", False) for d in diag
        )
        log.append({
            "tau_sr": ts, "sigma_sr": ss, "tau_coll": tc,
            "all_3_1a_softhjb_converged": all_three_converged,
            "per_seed_converged": [d.get("v2_result", {}).get("converged") for d in diag],
            "per_seed_t_first": [d.get("v2_result", {}).get("t_first") for d in diag],
            "per_seed_first_cross_step": [d.get("first_cross_smoothed_sr_threshold_step") for d in diag],
        })
        if all_three_converged and chosen is None:
            chosen = {"tau_sr": ts, "sigma_sr": ss, "tau_coll": tc, "diagnostic": diag}
            break
    return {"calibration_log": log, "chosen_thresholds": chosen}


def retroactive_table(tau_sr: float, sigma_sr: float, tau_coll: float):
    rows = []
    disagreements = []
    for method in CALIBRATION_METHODS:
        for scen in CALIBRATION_SCENARIOS:
            for seed in SEEDS:
                df = load_metrics(method, scen, seed)
                if df is None:
                    rows.append({"method": method, "scenario": scen, "seed": seed,
                                 "v1_converged": None, "v1_t_first": None,
                                 "v2_converged": None, "v2_t_first": None,
                                 "mean_SR_post_first": None, "error": "no_metrics_csv"})
                    continue
                v1 = evaluate_convergence_v1(df)
                v2 = evaluate_convergence_v2(df, tau_sr=tau_sr, sigma_sr=sigma_sr, tau_coll=tau_coll)
                row = {
                    "method": method, "scenario": scen, "seed": seed,
                    "v1_converged": v1["converged"], "v1_t_first": v1["t_first"],
                    "v2_converged": v2["converged"], "v2_t_first": v2["t_first"],
                    "v2_n_satisfied": v2["n_satisfied_evals"],
                    "mean_SR_post_first_v2": v2["mean_SR_post_first"],
                }
                rows.append(row)
                if v1["converged"] != v2["converged"]:
                    disagreements.append(row)
    return rows, disagreements


def synthetic_sanity_check():
    """v2 must return converged=True on perfect run, converged=False on failed run."""
    n = 30
    steps = np.arange(1, n + 1) * 4096
    perfect = pd.DataFrame({
        "total_steps": steps,
        "mean_reward": np.full(n, 100.0),
        "n_episodes": np.full(n, 8.0),
        "n_collisions": np.zeros(n),
        "n_successes": np.full(n, 8.0),  # SR = 1.0
    })
    failed = pd.DataFrame({
        "total_steps": steps,
        "mean_reward": np.full(n, -1000.0),
        "n_episodes": np.full(n, 8.0),
        "n_collisions": np.full(n, 4.0),  # collision rate 0.5
        "n_successes": np.zeros(n),  # SR = 0
    })
    r_perfect = evaluate_convergence_v2(perfect)
    r_failed = evaluate_convergence_v2(failed)
    return {
        "perfect_converged": r_perfect["converged"],
        "perfect_t_first": r_perfect["t_first"],
        "failed_converged": r_failed["converged"],
        "perfect_pass": (r_perfect["converged"] is True),
        "failed_pass": (r_failed["converged"] is False),
        "all_pass": (r_perfect["converged"] is True and r_failed["converged"] is False),
    }


def run_verification_checks(rows, disagreements, sanity, chosen):
    checks = {}
    # 1) Sanity
    checks["check1_synthetic_sanity"] = {
        "pass": sanity["all_pass"],
        "detail": sanity,
    }
    # 2) 1a Soft-HJB recovery
    softhjb_1a = [r for r in rows if r["method"] == "soft_hjb_aux" and r["scenario"] == "1a"]
    all_softhjb_1a_converged = all(r.get("v2_converged") is True for r in softhjb_1a)
    checks["check2_1a_softhjb_recovery"] = {
        "pass": all_softhjb_1a_converged,
        "n_recovered": sum(1 for r in softhjb_1a if r.get("v2_converged")),
        "n_total": len(softhjb_1a),
        "detail": [{"seed": r["seed"], "v1_t_first": r["v1_t_first"], "v2_t_first": r["v2_t_first"], "mean_SR": r["mean_SR_post_first_v2"]} for r in softhjb_1a],
    }
    # 3) 2_dense honesty
    dense_jobs = [r for r in rows if r["scenario"] == "2_dense"]
    any_dense_v2_converged = any(r.get("v2_converged") is True for r in dense_jobs)
    checks["check3_2dense_honesty"] = {
        "pass": (not any_dense_v2_converged),
        "n_2dense_total": len(dense_jobs),
        "n_2dense_v2_converged": sum(1 for r in dense_jobs if r.get("v2_converged")),
        "detail": [{"method": r["method"], "seed": r["seed"], "v2_converged": r.get("v2_converged"), "v2_t_first": r.get("v2_t_first"), "mean_SR": r.get("mean_SR_post_first_v2")} for r in dense_jobs if r.get("v2_converged")],
    }
    # 4) Code unchanged outside scope — handled via mtime in caller
    checks["check4_code_unchanged_outside_scope"] = {"pass": "see_status_report",
                                                     "detail": "see closure-summary mtime check in stdout"}
    # 5) JSON validity — handled by caller
    checks["check5_json_valid"] = {"pass": True, "detail": "json.dump ensures valid output"}
    return checks


def main():
    cal = calibrate()
    chosen = cal["chosen_thresholds"]
    if chosen is None:
        # Per spec §3.3: STOP and escalate if no threshold combo within hard limits succeeds.
        out = {
            "status": "calibration_failed_within_hard_limits",
            "calibration_log": cal["calibration_log"],
            "next_step": "user_direction_required",
        }
        STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("[criterion_v2] CALIBRATION FAILED within hard limits. Stop.")
        return out

    tau_sr, sigma_sr, tau_coll = chosen["tau_sr"], chosen["sigma_sr"], chosen["tau_coll"]
    print(f"[criterion_v2] Calibrated thresholds: tau_sr={tau_sr}, sigma_sr={sigma_sr}, tau_coll={tau_coll}")

    rows, disagreements = retroactive_table(tau_sr, sigma_sr, tau_coll)
    sanity = synthetic_sanity_check()
    checks = run_verification_checks(rows, disagreements, sanity, chosen)

    out = {
        "stage": "3F-Stage3-criterion-update",
        "criterion_version": CRITERION_VERSION,
        "calibrated_thresholds": {
            "tau_sr": tau_sr, "sigma_sr": sigma_sr, "tau_coll": tau_coll,
            "window_steps": WINDOW_STEPS,
        },
        "calibration_log": cal["calibration_log"],
        "calibration_rationale": chosen.get("diagnostic"),
        "phase3_retroactive_table": rows,
        "disagreements_v1_v2": disagreements,
        "n_disagreements": len(disagreements),
        "verification_results": checks,
        "methodology_note_path": "verification/criterion_v2_methodology_note.md",
        "next_step": "step_11_with_v2_criterion",
    }
    STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[criterion_v2] wrote {STATUS_OUT}")
    print(f"[criterion_v2] Verification check #2 (1a Soft-HJB recovery): {'PASS' if checks['check2_1a_softhjb_recovery']['pass'] else 'FAIL'} ({checks['check2_1a_softhjb_recovery']['n_recovered']}/{checks['check2_1a_softhjb_recovery']['n_total']})")
    print(f"[criterion_v2] Verification check #3 (2_dense honesty): {'PASS' if checks['check3_2dense_honesty']['pass'] else 'FAIL'} (n_2dense_v2_converged={checks['check3_2dense_honesty']['n_2dense_v2_converged']}/{checks['check3_2dense_honesty']['n_2dense_total']})")
    print(f"[criterion_v2] disagreements: {len(disagreements)}")
    return out


if __name__ == "__main__":
    main()
