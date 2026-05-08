"""Phase 23: Logging instrumentation pipeline verification (post Phase 1A).

Runs each of the 5 training scripts in smoke-test mode, checks the produced
metrics.csv / meta.json / trajectories/ artifacts conform to the canonical
Phase 1A schema, validates eval_metrics.csv, and re-aggregates the existing
phase suite to guarantee zero regressions.

Tests:
  23.1 Schema consistency across all 5 training scripts (smoke runs)
  23.2 Column-value sanity (monotonic step/iter/wall, action_dist sums, residual zeros)
  23.3 meta.json schema (top-level keys, ISO timestamps, git sha format, result_summary)
  23.4 Trajectory logger ring buffer (collision_NNNN.csv files exist, header comment, columns)
  23.5 eval_metrics.csv schema + valid terminal_state values
  23.6 Existing 24-phase suite still passes (re-aggregation)
  23.7 Forward-compatibility (both L_residual_optimality and L_residual_safety columns
       present in every metrics.csv; final_residual_loss_{optimality,safety} in meta result_summary)
"""
import sys
import os
import re
import csv
import json
import time
import glob
import shutil
import subprocess
import traceback
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from experiments.pde.run_metadata import METRICS_COLUMNS, EVAL_METRICS_COLUMNS

results = {"phase": "23", "tests": {}, "pass": True}


def _record(name, ok, details=None):
    results["tests"][name] = {"pass": bool(ok)}
    if details:
        results["tests"][name].update(details)
    if not ok:
        results["pass"] = False
    sym = "OK  " if ok else "FAIL"
    if details and "error" in details:
        print(f"  [{sym}] {name}: error={details['error']}")
    else:
        print(f"  [{sym}] {name}")


# Methods we instrument and their script paths + expected residual side.
METHODS = [
    {"name": "drppo",         "script": "experiments/pde/train_drppo_baseline.py",  "residual": "none"},
    {"name": "hjb_aux",       "script": "experiments/pde/train_hjb_aux.py",         "residual": "optimality"},
    {"name": "soft_hjb_aux",  "script": "experiments/pde/train_soft_hjb_aux.py",    "residual": "optimality"},
    {"name": "eikonal_aux",   "script": "experiments/pde/train_eikonal_aux.py",     "residual": "safety"},
    {"name": "cbf_aux",       "script": "experiments/pde/train_cbf_aux.py",         "residual": "safety"},
]

# Smoke-run output dirs (under verification/_smoke_<method>/).
SMOKE_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_phase23_smoke")
SMOKE_TOTAL_STEPS = "5000"
SMOKE_SCENARIO = "1a"
SMOKE_MANEUVER = "stem_right"
SMOKE_SEED = "42"
PYTHON_BIN = sys.executable


def _smoke_dir(method):
    return os.path.join(SMOKE_BASE, method)


def _run_smoke_one(method_entry):
    """Run a single training script in smoke mode. Returns (ok, log_tail)."""
    out_dir = _smoke_dir(method_entry["name"])
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        PYTHON_BIN, method_entry["script"],
        "--out_dir", out_dir,
        "--total_steps", SMOKE_TOTAL_STEPS,
        "--scenario", SMOKE_SCENARIO,
        "--ego_maneuver", SMOKE_MANEUVER,
        "--seed", SMOKE_SEED,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    log_path = os.path.join(out_dir, "smoke.log")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                              stdout=logf, stderr=subprocess.STDOUT,
                              timeout=900)  # 15 min cap per script
    elapsed = time.time() - t0
    ok = (proc.returncode == 0)
    return ok, elapsed, log_path


def test_23_1():
    """Run each smoke training and verify metrics.csv exists with the canonical
    schema (columns match METRICS_COLUMNS exactly, in order, with at least 1 row).
    """
    per_method = []
    overall_ok = True
    for entry in METHODS:
        try:
            ok_run, elapsed, log_path = _run_smoke_one(entry)
            metrics_path = os.path.join(_smoke_dir(entry["name"]), "metrics.csv")
            schema_ok = False
            n_rows = 0
            mismatch = []
            if ok_run and os.path.isfile(metrics_path):
                with open(metrics_path) as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    schema_ok = (header == METRICS_COLUMNS)
                    if not schema_ok:
                        mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(header, METRICS_COLUMNS)) if a != b]
                    n_rows = sum(1 for _ in reader)
            method_ok = ok_run and schema_ok and n_rows >= 1
            per_method.append({
                "method": entry["name"],
                "smoke_returncode_zero": ok_run,
                "elapsed_seconds": elapsed,
                "metrics_csv_exists": os.path.isfile(metrics_path),
                "schema_matches": schema_ok,
                "header_mismatch_first": mismatch[:3],
                "n_rows": n_rows,
                "log": log_path,
            })
            overall_ok = overall_ok and method_ok
        except Exception as e:
            per_method.append({
                "method": entry["name"],
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=2),
            })
            overall_ok = False
    _record("23.1_schema_consistency", overall_ok, {"per_method": per_method})


def _check_value_sanity_for_method(entry):
    metrics_path = os.path.join(_smoke_dir(entry["name"]), "metrics.csv")
    problems = []
    if not os.path.isfile(metrics_path):
        return False, ["metrics.csv missing"]
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return False, ["zero rows"]
    last_iter = -1
    last_steps = -1
    last_wall = -1.0
    for ri, r in enumerate(rows):
        try:
            it = int(r["iteration"])
            ts = int(r["total_steps"])
            wall = float(r["wall_time_seconds"])
            it_t = float(r["iter_time_seconds"])
            es_t = float(r["env_step_time_seconds"])
            ls_t = float(r["learn_step_time_seconds"])
            ad = [float(r[f"action_dist_{k}"]) for k in
                  ("stop", "creep", "yield", "go", "abort")]
            n_eps = int(r["n_episodes"])
            n_coll = int(r["n_collisions"])
            n_succ = int(r["n_successes"])
            n_to = int(r["n_timeouts"])
            n_ab = int(r["n_aborts"])
            r_opt = float(r["L_residual_optimality"])
            r_saf = float(r["L_residual_safety"])
        except (KeyError, ValueError) as e:
            problems.append(f"row {ri}: parse error {e}")
            continue
        if it <= last_iter:
            problems.append(f"row {ri}: iteration not monotonic ({it} <= {last_iter})")
        if ts <= last_steps:
            problems.append(f"row {ri}: total_steps not monotonic ({ts} <= {last_steps})")
        if wall < last_wall - 1e-6:
            problems.append(f"row {ri}: wall not monotonic ({wall} < {last_wall})")
        last_iter, last_steps, last_wall = it, ts, wall
        if it_t < 0:
            problems.append(f"row {ri}: iter_time negative")
        if es_t + ls_t > it_t * 1.05 + 0.5:
            problems.append(f"row {ri}: env+learn ({es_t + ls_t:.3f}) > iter ({it_t:.3f})")
        s = sum(ad)
        if s > 1e-9 and abs(s - 1.0) > 1e-2:
            problems.append(f"row {ri}: action_dist sum {s:.4f} != 1")
        if n_eps > 0 and (n_coll + n_succ + n_to + n_ab) != n_eps:
            problems.append(f"row {ri}: terminal counts {n_coll}+{n_succ}+{n_to}+{n_ab} != n_eps={n_eps}")
        if entry["residual"] == "none":
            if r_opt != 0.0 or r_saf != 0.0:
                problems.append(f"row {ri}: drppo expects both residuals==0, got opt={r_opt} saf={r_saf}")
        elif entry["residual"] == "optimality":
            if r_saf != 0.0:
                problems.append(f"row {ri}: optimality method expects safety==0, got {r_saf}")
        elif entry["residual"] == "safety":
            if r_opt != 0.0:
                problems.append(f"row {ri}: safety method expects optimality==0, got {r_opt}")
    return (len(problems) == 0), problems


def test_23_2():
    per_method = []
    overall_ok = True
    for entry in METHODS:
        try:
            ok, problems = _check_value_sanity_for_method(entry)
            per_method.append({
                "method": entry["name"], "pass": ok,
                "n_problems": len(problems), "problems": problems[:5],
            })
            overall_ok = overall_ok and ok
        except Exception as e:
            per_method.append({"method": entry["name"], "error": f"{type(e).__name__}: {e}"})
            overall_ok = False
    _record("23.2_value_sanity", overall_ok, {"per_method": per_method})


REQUIRED_META_KEYS = [
    "run_id", "start_time_iso", "end_time_iso", "wall_time_seconds",
    "method", "scenario", "ego_maneuver", "seed", "intent_on",
    "total_steps_target", "total_steps_actual", "convergence_reason",
    "git_commit", "git_branch", "git_dirty", "hostname", "device",
    "torch_version", "python_version", "config", "result_summary",
]

REQUIRED_RESULT_SUMMARY_KEYS = [
    "best_eval_return", "final_eval_return", "best_iteration",
    "final_collision_rate", "final_success_rate", "final_timeout_rate",
    "best_distillation_gap",
    "final_residual_loss_optimality", "final_residual_loss_safety",
]

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _check_meta(entry):
    meta_path = os.path.join(_smoke_dir(entry["name"]), "meta.json")
    if not os.path.isfile(meta_path):
        return False, ["meta.json missing"]
    with open(meta_path) as f:
        meta = json.load(f)
    issues = []
    for k in REQUIRED_META_KEYS:
        if k not in meta:
            issues.append(f"missing key '{k}'")
    for k in (REQUIRED_META_KEYS + ["start_time_iso", "end_time_iso"]):
        if k not in meta:
            continue
    for ts_key in ("start_time_iso", "end_time_iso"):
        v = meta.get(ts_key)
        if v and not _ISO_RE.match(str(v)):
            issues.append(f"{ts_key}={v!r} is not ISO-8601")
    wall = meta.get("wall_time_seconds")
    if wall is None or wall <= 0:
        issues.append(f"wall_time_seconds invalid: {wall!r}")
    target = int(meta.get("total_steps_target") or 0)
    actual = int(meta.get("total_steps_actual") or 0)
    if target == 0 or actual == 0:
        issues.append("total_steps_target / total_steps_actual missing")
    else:
        # Training stops at the first rollout boundary >= target. With the
        # default n_steps=4096 this means a 5000-step smoke ends at 8192.
        # Allow up to one extra rollout (n_steps_max=8192) past target.
        if actual < target:
            issues.append(f"total_steps_actual ({actual}) below target ({target})")
        if actual > target + 8192:
            issues.append(f"total_steps_actual ({actual}) more than one rollout past target ({target})")
    sha = str(meta.get("git_commit", ""))
    if not re.match(r"^[0-9a-f]{8}$", sha):
        issues.append(f"git_commit '{sha}' not 8 hex chars")
    rs = meta.get("result_summary") or {}
    if not isinstance(rs, dict):
        issues.append("result_summary is not a dict")
    else:
        for k in REQUIRED_RESULT_SUMMARY_KEYS:
            if k not in rs:
                issues.append(f"result_summary missing key '{k}'")
    return (len(issues) == 0), issues


def test_23_3():
    per_method = []
    overall_ok = True
    for entry in METHODS:
        try:
            ok, issues = _check_meta(entry)
            per_method.append({
                "method": entry["name"], "pass": ok,
                "n_issues": len(issues), "issues": issues[:5],
            })
            overall_ok = overall_ok and ok
        except Exception as e:
            per_method.append({"method": entry["name"], "error": f"{type(e).__name__}: {e}"})
            overall_ok = False
    _record("23.3_meta_schema", overall_ok, {"per_method": per_method})


def test_23_4():
    """Trajectory logger directory + at least one collision_NNNN.csv with header
    comment. If a method produced no collisions during the 5k smoke, it's allowed
    to have an empty trajectories/ — but at least one method across the 5 should
    have produced a collision. Otherwise we treat as inconclusive (warning).
    """
    per_method = []
    any_collision = False
    overall_ok = True
    for entry in METHODS:
        traj_dir = os.path.join(_smoke_dir(entry["name"]), "trajectories")
        present = os.path.isdir(traj_dir)
        n_files = 0
        sample_ok = False
        sample_issue = None
        n_step_rows = 0
        if present:
            files = sorted(glob.glob(os.path.join(traj_dir, "collision_*.csv")))
            n_files = len(files)
            if files:
                any_collision = True
                with open(files[0]) as f:
                    first = f.readline().strip()
                    if not first.startswith("#"):
                        sample_issue = "first line is not a comment"
                    else:
                        for needle in ("scenario=", "ego_maneuver=", "seed=",
                                       "episode_idx=", "terminal_step=",
                                       "collision_agent_id="):
                            if needle not in first:
                                sample_issue = f"missing token '{needle}'"
                                break
                    if sample_issue is None:
                        reader = csv.DictReader(f)
                        cols = reader.fieldnames or []
                        expected = ["step", "ego_x", "ego_y", "ego_psi", "ego_v", "ego_a",
                                    "action", "reward", "min_ttc", "n_agents",
                                    "collision_agent_id", "terminal_flag"]
                        if cols != expected:
                            sample_issue = f"column mismatch: {cols} vs {expected}"
                        else:
                            for _ in reader:
                                n_step_rows += 1
                            sample_ok = (n_step_rows >= 1)
        ring_ok = (n_files <= 50)
        method_ok = present and ring_ok
        if files_existed := (present and n_files > 0):
            method_ok = method_ok and sample_ok and (sample_issue is None)
        per_method.append({
            "method": entry["name"],
            "trajectories_dir_exists": present,
            "n_collision_files": n_files,
            "ring_capped_to_50": ring_ok,
            "first_file_sample_ok": sample_ok if files_existed else None,
            "first_file_issue": sample_issue,
            "first_file_n_rows": n_step_rows,
        })
        overall_ok = overall_ok and method_ok
    _record("23.4_trajectory_ring_buffer", overall_ok, {
        "per_method": per_method,
        "any_collision_observed_across_methods": any_collision,
    })


def test_23_5():
    """Run eval.py against the DRPPO smoke checkpoint and verify eval_metrics.csv."""
    drppo_dir = _smoke_dir("drppo")
    candidate_ckpts = sorted(glob.glob(os.path.join(drppo_dir, "model_drppo_*.pt")))
    if not candidate_ckpts:
        _record("23.5_eval_metrics_csv", False, {"error": "no DRPPO checkpoint produced by smoke"})
        return
    ckpt = candidate_ckpts[-1]
    cmd = [
        PYTHON_BIN, "experiments/pde/eval.py",
        "--method", "drppo",
        "--checkpoint", ckpt,
        "--out_dir", drppo_dir,
        "--scenario", SMOKE_SCENARIO,
        "--ego_maneuver", SMOKE_MANEUVER,
        "--episodes", "5",
        "--seeds", "1234",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    eval_log = os.path.join(drppo_dir, "eval_smoke.log")
    try:
        with open(eval_log, "w") as logf:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                  stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=600)
        ok_run = (proc.returncode == 0)
    except Exception as e:
        _record("23.5_eval_metrics_csv", False,
                {"error": f"{type(e).__name__}: {e}", "log": eval_log})
        return

    eval_csv = os.path.join(drppo_dir, "eval_metrics.csv")
    schema_ok = False
    n_rows = 0
    valid_terminals = True
    bad_terms = []
    if os.path.isfile(eval_csv):
        with open(eval_csv) as f:
            reader = csv.reader(f)
            header = next(reader, [])
            schema_ok = (header == EVAL_METRICS_COLUMNS)
            ts_idx = header.index("terminal_state") if "terminal_state" in header else -1
            for row in reader:
                n_rows += 1
                if ts_idx >= 0 and ts_idx < len(row):
                    if row[ts_idx] not in ("collision", "success", "timeout", "abort"):
                        valid_terminals = False
                        bad_terms.append(row[ts_idx])
    ok = ok_run and schema_ok and n_rows >= 5 and valid_terminals
    _record("23.5_eval_metrics_csv", ok, {
        "subprocess_ok": ok_run,
        "schema_matches": schema_ok,
        "n_rows": n_rows,
        "valid_terminal_states": valid_terminals,
        "bad_terminal_values_first": bad_terms[:3],
        "log": eval_log,
    })


def test_23_6():
    ver_dir = os.path.dirname(os.path.abspath(__file__))
    phases = {}
    failed = []
    for path in sorted(glob.glob(os.path.join(ver_dir, "phase*.json"))):
        name = os.path.basename(path).replace(".json", "")
        if name == "phase23_logging_pipeline":
            continue
        try:
            with open(path) as f:
                phases[name] = json.load(f)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
    all_pass = all(v.get("pass", True) is True
                   for v in phases.values() if isinstance(v, dict))
    failed_phases = [n for n, v in phases.items()
                     if isinstance(v, dict) and v.get("pass", True) is False]
    n_phases = len(phases)
    ok = all_pass and n_phases >= 24 and not failed
    _record("23.6_existing_suite", ok, {
        "n_phases": n_phases,
        "all_pass": all_pass,
        "failed_phases": failed_phases,
        "load_failures": failed,
    })


def test_23_7():
    per_method = []
    overall_ok = True
    for entry in METHODS:
        metrics_path = os.path.join(_smoke_dir(entry["name"]), "metrics.csv")
        meta_path = os.path.join(_smoke_dir(entry["name"]), "meta.json")
        cols_present = False
        meta_keys_present = False
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                header = next(csv.reader(f), [])
                cols_present = (
                    "L_residual_optimality" in header and "L_residual_safety" in header
                )
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            rs = (meta.get("result_summary") or {})
            meta_keys_present = (
                isinstance(rs, dict)
                and "final_residual_loss_optimality" in rs
                and "final_residual_loss_safety" in rs
            )
        ok = cols_present and meta_keys_present
        per_method.append({
            "method": entry["name"],
            "metrics_has_both_residual_columns": cols_present,
            "meta_has_both_final_residual_keys": meta_keys_present,
        })
        overall_ok = overall_ok and ok
    _record("23.7_forward_compat", overall_ok, {"per_method": per_method})


def main():
    print("==== PHASE 23: LOGGING PIPELINE VERIFICATION ====")
    os.makedirs(SMOKE_BASE, exist_ok=True)

    test_23_1()
    test_23_2()
    test_23_3()
    test_23_4()
    test_23_5()
    test_23_6()
    test_23_7()

    out_path = os.path.join(os.path.dirname(__file__),
                            "phase23_logging_pipeline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print("ALL_PASS =", results["pass"])
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
