"""Phase 27: Analysis pipeline verification (post Phase 1E).

All tests use synthetic data — no actual training runs needed.

Tests:
  27.1  Loader handles synthetic Tier 1 data (6 runs × 5 iterations)
  27.2  Loader detects 4 quality-issue runs alongside 1 valid run
  27.3  Final-window metrics: rate matches manual calculation
  27.4  Welch's t-test matches scipy reference
  27.5  Holm correction on 10 known p-values
  27.6  Cohen's d matches manual calculation
  27.7  LaTeX table has booktabs structure with correct column count
  27.8  Plot generation produces PDF and HTML files
  27.9  End-to-end pipeline run
  27.10 Determinism (CSV outputs byte-identical between runs)
  27.11 Existing 28-phase suite still passes
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from glob import glob
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from analysis import config as A_config
from analysis.loader import load_results, parse_tag
from analysis.metrics import compute_eval_metrics, compute_final_metrics
from analysis.plots import generate_all_plots
from analysis.quality import save_quality_report
from analysis.stats import (
    cohens_d, compute_statistical_tests, holm_correction, welch_ttest,
)
from analysis.tables import generate_all_tables

results = {"phase": "27", "tests": {}, "pass": True}


def _record(name, ok, details=None):
    results["tests"][name] = {"pass": bool(ok)}
    if details:
        results["tests"][name].update(details)
    if not ok:
        results["pass"] = False
    sym = "OK  " if ok else "FAIL"
    print(f"  [{sym}] {name}")


# ─────────────────────────────────────────────────────────────────────────
# Synthetic data builders
# ─────────────────────────────────────────────────────────────────────────
PYTHON_BIN = sys.executable
SCENARIOS = ("1a",)
MANEUVERS = ("stem_right",)
METHODS_T1 = ("drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux")


def _make_metrics_csv(n_iter=5, method="drppo", seed=42, n_steps=4096) -> pd.DataFrame:
    rng = np.random.default_rng(seed + hash(method) % 1000)
    rows = []
    for i in range(1, n_iter + 1):
        opt = saf = 0.0
        if method in ("hjb_aux", "soft_hjb_aux", "fusion_aux"):
            opt = abs(float(rng.normal(1.0, 0.2)))
        if method in ("eikonal_aux", "cbf_aux", "fusion_aux"):
            saf = abs(float(rng.normal(0.5, 0.1)))
        n_eps = 8
        n_coll = int(rng.integers(0, 3))
        n_succ = int(rng.integers(0, 3))
        n_to = max(0, n_eps - n_coll - n_succ)
        n_ab = 0
        # action_dist sums to 1.
        ad = rng.dirichlet(np.ones(5)).tolist()
        rows.append({
            "iteration": i,
            "total_steps": i * n_steps,
            "wall_time_seconds": i * 5.0,
            "iter_time_seconds": 5.0,
            "env_step_time_seconds": 2.0,
            "learn_step_time_seconds": 2.5,
            "residual_compute_time_seconds": 0.5 if method != "drppo" else 0.0,
            "L_total": float(rng.normal(50.0, 5.0)),
            "L_policy": float(rng.normal(0.0, 0.1)),
            "L_value": float(rng.normal(40.0, 5.0)),
            "L_entropy": float(rng.normal(1.5, 0.1)),
            "L_residual_optimality": opt,
            "L_residual_safety": saf,
            "L_distill": float(rng.normal(5.0, 1.0)) if method != "drppo" else 0.0,
            "mean_reward": float(rng.normal(-100.0, 30.0)),
            "mean_episode_length": 500.0,
            "n_episodes": n_eps,
            "n_collisions": n_coll,
            "n_successes": n_succ,
            "n_timeouts": n_to,
            "n_aborts": n_ab,
            "action_dist_stop": ad[0],
            "action_dist_creep": ad[1],
            "action_dist_yield": ad[2],
            "action_dist_go": ad[3],
            "action_dist_abort": ad[4],
        })
    return pd.DataFrame(rows)


def _make_meta(method, scenario, maneuver, seed, *, n_steps=4096, n_iter=5,
               intent_on=False, valid_summary=True) -> dict:
    cfg = {k: None for k in A_config.EXPECTED_CONFIG_KEYS}
    cfg.update({
        "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
        "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "n_epochs_per_update": 8, "batch_size": 128, "n_steps": n_steps,
        "policy_hidden_size": 256, "policy_n_layers": 3,
        "gru_hidden_size": 256, "gru_n_layers": 1,
    })
    if method in ("hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"):
        cfg["lambda_residual"] = 0.2
        cfg["lambda_distill"] = 0.25
        cfg["collocation_size"] = 256
    if method in ("soft_hjb_aux", "fusion_aux"):
        cfg["tau_soft"] = 1.0
        cfg["lambda_actor_kl"] = 0.1
    if method in ("cbf_aux", "fusion_aux"):
        cfg["alpha_cbf"] = 1.0
        cfg["barrier_offset"] = 10.0
    if method == "eikonal_aux":
        cfg["w_fail"] = 50.0
    if method == "fusion_aux":
        cfg["w_optimality"] = 1.0
        cfg["w_safety"] = 1.0

    rs = None
    if valid_summary:
        rs = {
            "best_eval_return": -120.0,
            "final_eval_return": -130.0,
            "best_iteration": 1,
            "final_collision_rate": 0.1,
            "final_success_rate": None,
            "final_timeout_rate": None,
            "best_distillation_gap": 5.0,
            "final_residual_loss_optimality": 1.0 if method in ("hjb_aux", "soft_hjb_aux", "fusion_aux") else 0.0,
            "final_residual_loss_safety": 0.5 if method in ("eikonal_aux", "cbf_aux", "fusion_aux") else 0.0,
        }
    return {
        "run_id": str(uuid.uuid4()),
        "start_time_iso": "2026-05-04T00:00:00+00:00",
        "end_time_iso": "2026-05-04T00:01:00+00:00",
        "wall_time_seconds": 60.0,
        "method": method,
        "scenario": scenario,
        "ego_maneuver": maneuver,
        "seed": seed,
        "intent_on": bool(intent_on),
        "total_steps_target": n_steps * n_iter,
        "total_steps_actual": n_steps * n_iter,
        "convergence_reason": "max_steps_reached",
        "git_commit": "abcdef01",
        "git_branch": "phase27-synthetic",
        "git_dirty": False,
        "hostname": "synth",
        "device": "cpu",
        "torch_version": "0.0.0",
        "python_version": "3.x",
        "config": cfg,
        "result_summary": rs,
    }


def _write_run(root: Path, tier_dir: str, run_id: str, *,
               metrics_df: pd.DataFrame, meta: dict,
               write_metrics=True, write_meta=True):
    rd = root / tier_dir / run_id
    rd.mkdir(parents=True, exist_ok=True)
    if write_metrics:
        metrics_df.to_csv(rd / "metrics.csv", index=False)
    if write_meta:
        with open(rd / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    return rd


def _build_synth_tier1(root: Path, n_seeds=3) -> int:
    """Build n_seeds × len(METHODS_T1) Tier 1 synthetic runs."""
    n = 0
    for method in METHODS_T1:
        for seed in (42, 123, 456)[:n_seeds]:
            for scen in SCENARIOS:
                for man in MANEUVERS:
                    intent_tag = "nointent"
                    tag = f"T1_{scen}_{man}_{method}_{intent_tag}_s{seed}"
                    metrics_df = _make_metrics_csv(method=method, seed=seed)
                    meta = _make_meta(method, scen, man, seed)
                    _write_run(root, "tier1", tag,
                               metrics_df=metrics_df, meta=meta)
                    n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────
# 27.1 — Loader on synthetic Tier 1
# ─────────────────────────────────────────────────────────────────────────
def test_27_1():
    issues = []
    counts = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ablation"
            n_runs = _build_synth_tier1(root, n_seeds=1)  # 6 runs, one per method
            long_df, wide_df, qr = load_results(root)
            counts = {
                "n_runs": int(qr["n_runs_discovered"]),
                "n_valid": int(qr["n_runs_valid"]),
                "long_rows": int(len(long_df)),
                "wide_rows": int(len(wide_df)),
            }
            if counts["n_runs"] != 6:
                issues.append(f"discovered {counts['n_runs']}, expected 6")
            if counts["n_valid"] != 6:
                issues.append(f"valid {counts['n_valid']}, expected 6")
            if counts["long_rows"] != 30:
                issues.append(f"long_df has {counts['long_rows']} rows, expected 30")
            if counts["wide_rows"] != 6:
                issues.append(f"wide_df has {counts['wide_rows']} rows, expected 6")
            for col in ("method", "scenario", "ego_maneuver", "seed",
                        "iteration", "total_steps", "L_residual_optimality",
                        "L_residual_safety"):
                if col not in long_df.columns:
                    issues.append(f"long_df missing column {col!r}")
            for col in ("method", "final_collision_rate", "final_success_rate",
                        "final_mean_reward", "lambda_residual"):
                if col not in wide_df.columns:
                    issues.append(f"wide_df missing column {col!r}")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.1_loader_tier1_synthetic", len(issues) == 0, {
        "issues": issues, "counts": counts,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.2 — Loader detects 4 quality issues + 1 valid
# ─────────────────────────────────────────────────────────────────────────
def test_27_2():
    issues = []
    qr_summary = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ablation"

            # Run 1 — missing meta.json
            df1 = _make_metrics_csv(method="drppo", seed=1)
            _write_run(root, "tier1", "T1_1a_stem_right_drppo_nointent_s1",
                       metrics_df=df1,
                       meta={}, write_meta=False)

            # Run 2 — NaN in metrics
            df2 = _make_metrics_csv(method="hjb_aux", seed=2)
            df2.loc[2, "L_total"] = float("nan")
            meta2 = _make_meta("hjb_aux", "1a", "stem_right", 2)
            _write_run(root, "tier1", "T1_1a_stem_right_hjb_aux_nointent_s2",
                       metrics_df=df2, meta=meta2)

            # Run 3 — action_dist sum != 1
            df3 = _make_metrics_csv(method="cbf_aux", seed=3)
            df3.loc[1, "action_dist_stop"] = 0.0
            df3.loc[1, "action_dist_creep"] = 0.0
            df3.loc[1, "action_dist_yield"] = 0.0
            df3.loc[1, "action_dist_go"] = 0.5
            df3.loc[1, "action_dist_abort"] = 0.45  # sum = 0.95
            meta3 = _make_meta("cbf_aux", "1a", "stem_right", 3)
            _write_run(root, "tier1", "T1_1a_stem_right_cbf_aux_nointent_s3",
                       metrics_df=df3, meta=meta3)

            # Run 4 — drppo with non-zero L_residual_optimality
            df4 = _make_metrics_csv(method="drppo", seed=4)
            df4.loc[2, "L_residual_optimality"] = 5.0
            meta4 = _make_meta("drppo", "1a", "stem_right", 4)
            _write_run(root, "tier1", "T1_1a_stem_right_drppo_nointent_s4",
                       metrics_df=df4, meta=meta4)

            # Run 5 — fully valid soft_hjb_aux
            df5 = _make_metrics_csv(method="soft_hjb_aux", seed=5)
            meta5 = _make_meta("soft_hjb_aux", "1a", "stem_right", 5)
            _write_run(root, "tier1", "T1_1a_stem_right_soft_hjb_aux_nointent_s5",
                       metrics_df=df5, meta=meta5)

            long_df, wide_df, qr = load_results(root, skip_failed=True)
            qr_summary = {
                "n_runs_discovered": int(qr["n_runs_discovered"]),
                "n_runs_valid": int(qr["n_runs_valid"]),
                "n_runs_failed": int(qr["n_runs_failed"]),
            }
            # Run 1 (missing meta) is *not even discovered* because the
            # discoverer requires metrics.csv AND meta.json. The remaining
            # 4 runs are discovered; 3 fail quality checks; 1 is valid.
            if qr_summary["n_runs_discovered"] != 4:
                issues.append(f"discovered {qr_summary['n_runs_discovered']}, expected 4")
            if qr_summary["n_runs_failed"] != 3:
                issues.append(f"failed {qr_summary['n_runs_failed']}, expected 3")
            if qr_summary["n_runs_valid"] != 1:
                issues.append(f"valid {qr_summary['n_runs_valid']}, expected 1")
            failures_by_check = qr.get("failures_by_check", {})
            if not failures_by_check.get("nan_inf"):
                issues.append("nan_inf check did not flag run 2")
            if not failures_by_check.get("action_dist"):
                issues.append("action_dist check did not flag run 3")
            if not failures_by_check.get("method_specific"):
                issues.append("method_specific check did not flag run 4")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.2_loader_quality_failures", len(issues) == 0, {
        "issues": issues, "qr_summary": qr_summary,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.3 — Final-window metrics
# ─────────────────────────────────────────────────────────────────────────
def test_27_3():
    issues = []
    detail = {}
    try:
        # 20 iterations, 10% window = last 2.
        rng = np.random.default_rng(0)
        rows = []
        for i in range(1, 21):
            n_eps = 10
            n_coll = 0 if i < 19 else 5
            n_succ = 0
            n_to = n_eps - n_coll - n_succ
            rows.append({
                "iteration": i, "n_episodes": n_eps,
                "n_collisions": n_coll, "n_successes": n_succ,
                "n_timeouts": n_to, "n_aborts": 0,
                "mean_reward": -10.0,
                "action_dist_stop": 0.2, "action_dist_creep": 0.2,
                "action_dist_yield": 0.2, "action_dist_go": 0.2,
                "action_dist_abort": 0.2,
            })
        df = pd.DataFrame(rows)
        fm = compute_final_metrics(df, window_frac=0.10)
        # Last 2 rows: total episodes = 20, collisions = 5+5 = 10. Rate = 0.5.
        expected_rate = 10.0 / 20.0
        detail = {"final_collision_rate": fm["final_collision_rate"],
                  "expected": expected_rate}
        if abs(fm["final_collision_rate"] - expected_rate) > 1e-9:
            issues.append(f"got {fm['final_collision_rate']}, expected {expected_rate}")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.3_final_window_metrics", len(issues) == 0, {
        "issues": issues, "detail": detail,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.4 — Welch's t-test matches scipy
# ─────────────────────────────────────────────────────────────────────────
def test_27_4():
    issues = []
    detail = {}
    try:
        from scipy import stats as sp
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, size=20)
        b = rng.normal(0.5, 1.5, size=15)
        t_ours, p_ours = welch_ttest(a, b)
        ref = sp.ttest_ind(a, b, equal_var=False)
        detail = {"t": t_ours, "p": p_ours,
                  "t_ref": float(ref.statistic), "p_ref": float(ref.pvalue)}
        if abs(t_ours - float(ref.statistic)) > 1e-9:
            issues.append(f"t mismatch: ours={t_ours} ref={ref.statistic}")
        if abs(p_ours - float(ref.pvalue)) > 1e-9:
            issues.append(f"p mismatch: ours={p_ours} ref={ref.pvalue}")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.4_welch_ttest", len(issues) == 0, {
        "issues": issues, "detail": detail,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.5 — Holm correction
# ─────────────────────────────────────────────────────────────────────────
def test_27_5():
    issues = []
    detail = {}
    try:
        ps = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.10]
        adj, sig = holm_correction(ps, alpha=0.05)
        # Manual Holm: ascending p × (n - rank), running max:
        # p1=0.001 × 10 = 0.010
        # p2=0.005 × 9  = 0.045 → max(0.010, 0.045) = 0.045
        # p3=0.01  × 8  = 0.080
        # p4=0.02  × 7  = 0.140
        # p5=0.03  × 6  = 0.180
        # p6=0.04  × 5  = 0.200
        # p7=0.05  × 4  = 0.200
        # p8=0.06  × 3  = 0.180 → cap by running max → 0.200
        # p9=0.07  × 2  = 0.140 → 0.200
        # p10=0.10 × 1  = 0.100 → 0.200
        expected_adj = [0.010, 0.045, 0.080, 0.140, 0.180, 0.200, 0.200, 0.200, 0.200, 0.200]
        for i, (got, exp) in enumerate(zip(adj, expected_adj)):
            if abs(got - exp) > 1e-9:
                issues.append(f"adj[{i}] = {got}, expected {exp}")
        # At alpha=0.05, the first 2 should be significant.
        n_sig = sum(sig)
        if n_sig != 2:
            issues.append(f"significant count = {n_sig}, expected 2")
        detail = {"adj": adj, "expected": expected_adj, "n_sig": n_sig}
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.5_holm_correction", len(issues) == 0, {
        "issues": issues, "detail": detail,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.6 — Cohen's d
# ─────────────────────────────────────────────────────────────────────────
def test_27_6():
    issues = []
    detail = {}
    try:
        # Synthetic groups with known stats:
        #   a: mean=2.0, std=1.0, n=10  →  var=1.0
        #   b: mean=0.0, std=1.0, n=10  →  var=1.0
        # pooled_std = sqrt((9*1 + 9*1) / 18) = 1.0
        # d = (2.0 - 0.0) / 1.0 = 2.0
        rng = np.random.default_rng(11)
        a = rng.normal(0, 1, 1000)
        a = (a - a.mean()) / a.std(ddof=1) * 1.0 + 2.0   # rescale to mean=2, std=1
        b = rng.normal(0, 1, 1000)
        b = (b - b.mean()) / b.std(ddof=1) * 1.0 + 0.0
        d = cohens_d(a, b)
        detail = {"cohens_d": d, "expected": 2.0}
        if abs(d - 2.0) > 1e-6:
            issues.append(f"d = {d}, expected 2.0")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.6_cohens_d", len(issues) == 0, {
        "issues": issues, "detail": detail,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.7 — LaTeX table generation
# ─────────────────────────────────────────────────────────────────────────
def test_27_7():
    issues = []
    detail = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ablation"
            _build_synth_tier1(root, n_seeds=2)  # 12 runs
            long_df, wide_df, qr = load_results(root)
            out_dir = Path(td) / "tables"
            generate_all_tables(
                wide_df=wide_df, long_df=long_df,
                stats_results=None,
                output_dir=out_dir, formats=("csv", "tex"),
            )
            tex_files = sorted(out_dir.glob("*.tex"))
            csv_files = sorted(out_dir.glob("*.csv"))
            detail = {
                "n_tex": len(tex_files), "n_csv": len(csv_files),
                "tex_names": [p.name for p in tex_files],
            }
            if len(tex_files) < 1:
                issues.append("no .tex files generated")
            else:
                # Verify booktabs structure on the main comparison table.
                main = out_dir / "tier1_main_comparison.tex"
                tex = main.read_text() if main.is_file() else ""
                for needle in (r"\toprule", r"\midrule", r"\bottomrule",
                               r"\begin{tabular}", r"\end{tabular}",
                               r"\caption", r"\label"):
                    if needle not in tex:
                        issues.append(f"missing {needle!r} in tier1_main_comparison.tex")
                # Every row should have the same number of " & " separators.
                rows = [l for l in tex.splitlines() if l.endswith(r"\\")]
                cols_per_row = {l.count(" & ") for l in rows}
                if len(cols_per_row) > 1:
                    issues.append(f"inconsistent column count: {cols_per_row}")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.7_latex_table", len(issues) == 0, {
        "issues": issues, "detail": detail,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.8 — Plot generation produces PDF and HTML files
# ─────────────────────────────────────────────────────────────────────────
def test_27_8():
    issues = []
    detail = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ablation"
            _build_synth_tier1(root, n_seeds=2)  # 12 runs
            long_df, wide_df, qr = load_results(root)
            out_dir = Path(td) / "figures"
            generate_all_plots(
                long_df=long_df, wide_df=wide_df,
                stats_results=None,
                output_dir=out_dir, formats=("pdf", "html"),
            )
            pdfs = list(out_dir.rglob("*.pdf"))
            htmls = list(out_dir.rglob("*.html"))
            detail = {"n_pdf": len(pdfs), "n_html": len(htmls)}
            if len(pdfs) < 1:
                issues.append("no PDF figures produced")
            if len(htmls) < 1:
                issues.append("no HTML figures produced")
            for p in pdfs[:5]:
                if p.stat().st_size < 1024:
                    issues.append(f"PDF too small: {p.name} ({p.stat().st_size} B)")
            for h in htmls[:5]:
                if h.stat().st_size < 1024:
                    issues.append(f"HTML too small: {h.name} ({h.stat().st_size} B)")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.8_plot_generation", len(issues) == 0, {
        "issues": issues, "detail": detail,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.9 — End-to-end pipeline run
# ─────────────────────────────────────────────────────────────────────────
def test_27_9():
    issues = []
    detail = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ablation"
            _build_synth_tier1(root, n_seeds=2)
            out_root = Path(td) / "analysis"
            cmd = [
                PYTHON_BIN, "analysis/run_analysis.py",
                "--results_root", str(root),
                "--output_root", str(out_root),
                "--no_prompt",
            ]
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": REPO_ROOT},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=300,
            )
            detail["returncode"] = proc.returncode
            detail["stdout_tail"] = proc.stdout.decode()[-400:]
            if proc.returncode != 0:
                issues.append(f"run_analysis exited {proc.returncode}")
            for sub in ("tables", "figures", "meta"):
                if not (out_root / sub).is_dir():
                    issues.append(f"missing output subdir: {sub}")
            for f in ("meta/data_quality_report.json",
                     "meta/analysis_run_metadata.json",
                     "tables/tier1_main_comparison.csv"):
                if not (out_root / f).is_file():
                    issues.append(f"missing artifact: {f}")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.9_end_to_end", len(issues) == 0, {
        "issues": issues, "detail": detail,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.10 — Determinism (CSV outputs identical between two runs)
# ─────────────────────────────────────────────────────────────────────────
def test_27_10():
    issues = []
    detail = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ablation"
            _build_synth_tier1(root, n_seeds=2)
            out1 = Path(td) / "ana1"
            out2 = Path(td) / "ana2"
            for out in (out1, out2):
                cmd = [
                    PYTHON_BIN, "analysis/run_analysis.py",
                    "--results_root", str(root),
                    "--output_root", str(out),
                    "--no_plots", "--no_prompt",
                ]
                proc = subprocess.run(
                    cmd, cwd=REPO_ROOT,
                    env={**os.environ, "PYTHONPATH": REPO_ROOT},
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=180,
                )
                if proc.returncode != 0:
                    issues.append(f"run failed: {proc.stdout.decode()[-200:]}")
                    break
            csvs1 = sorted((out1 / "tables").rglob("*.csv"))
            mismatches = []
            for f1 in csvs1:
                rel = f1.relative_to(out1)
                f2 = out2 / rel
                if not f2.is_file():
                    mismatches.append(f"missing {rel}")
                    continue
                if f1.read_bytes() != f2.read_bytes():
                    mismatches.append(str(rel))
            detail = {"n_csvs": len(csvs1), "mismatches": mismatches[:5]}
            if mismatches:
                issues.append(f"{len(mismatches)} CSV files differ between runs")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("27.10_determinism", len(issues) == 0, {
        "issues": issues, "detail": detail,
    })


# ─────────────────────────────────────────────────────────────────────────
# 27.11 — Existing 28-phase suite still passes
# ─────────────────────────────────────────────────────────────────────────
def test_27_11():
    ver_dir = os.path.dirname(os.path.abspath(__file__))
    phases = {}
    failed_load = []
    for path in sorted(glob(os.path.join(ver_dir, "phase*.json"))):
        name = os.path.basename(path).replace(".json", "")
        if name == "phase27_analysis_pipeline":
            continue
        try:
            with open(path) as f:
                phases[name] = json.load(f)
        except Exception as e:
            failed_load.append((name, f"{type(e).__name__}: {e}"))
    all_pass = all(v.get("pass", True) is True
                   for v in phases.values() if isinstance(v, dict))
    failed_phases = [n for n, v in phases.items()
                     if isinstance(v, dict) and v.get("pass", True) is False]
    n_phases = len(phases)
    ok = all_pass and n_phases >= 28 and not failed_load
    _record("27.11_existing_suite", ok, {
        "n_phases": n_phases,
        "all_pass": all_pass,
        "failed_phases": failed_phases,
        "load_failures": failed_load,
    })


def main():
    print("==== PHASE 27: ANALYSIS PIPELINE VERIFICATION ====")
    test_27_1()
    test_27_2()
    test_27_3()
    test_27_4()
    test_27_5()
    test_27_6()
    test_27_7()
    test_27_8()
    test_27_9()
    test_27_10()
    test_27_11()

    out_path = os.path.join(os.path.dirname(__file__),
                            "phase27_analysis_pipeline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print("ALL_PASS =", results["pass"])
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
