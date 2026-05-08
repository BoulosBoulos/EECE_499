"""Phase 30 — Calibration verification (SPEC_PHASE_3 Step 9).

Six tests must PASS:
  30.1 stress_test_passed       — stress test completed cleanly
  30.2 all_36_jobs_completed    — full calibration produced 36 valid run dirs
  30.3 convergence_output_valid — calibrated_total_steps.json contains a calibrated_steps int
  30.4 all_cells_converged      — all 36 cells (post-extension if used) classified as converged
  30.5 yaml_updated             — config_frozen_v1.yaml total_steps reflects the calibration result
  30.6 existing_suite           — every other phase JSON still pass=True
"""
from __future__ import annotations
import os, sys, json, glob, hashlib
from pathlib import Path

VERIF = Path(__file__).resolve().parent
ROOT = VERIF.parent

results = {"phase": "30", "tests": {}}


def _record(test_id: str, ok: bool, details=None) -> None:
    badge = "OK  " if ok else "FAIL"
    print(f"[{badge}] {test_id}")
    if details and not ok:
        s = json.dumps(details, default=str)
        print(f"       details: {s[:250]}")
    results["tests"][test_id] = {"pass": bool(ok), "details": details}


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# 30.1 — Stress test passed
def test_30_1() -> None:
    j = _load_json(VERIF / "phase30_stress_test.json")
    ok = bool(j and j.get("pass") and j.get("n_completed", 0) == 15 and j.get("n_failed", 0) == 0)
    _record("30.1_stress_test_passed", ok, j or "missing phase30_stress_test.json")


# 30.2 — All 36 calibration jobs completed
def test_30_2() -> None:
    cal_root = ROOT / "results" / "calibration"
    metas = sorted(cal_root.glob("CAL_*/meta.json"))
    n_dirs = len(metas)
    issues = []
    n_valid = 0
    for mp in metas:
        try:
            meta = json.loads(mp.read_text())
        except Exception as e:
            issues.append({"meta_path": str(mp), "error": str(e)}); continue
        if int(meta.get("total_steps_actual") or 0) < 480_000:
            issues.append({"run_id": meta.get("run_id"),
                            "total_steps_actual": meta.get("total_steps_actual"),
                            "issue": "below 480k step threshold"})
            continue
        metrics_csv = mp.parent / "metrics.csv"
        if not metrics_csv.is_file():
            issues.append({"run_id": meta.get("run_id"), "issue": "metrics.csv missing"}); continue
        try:
            import pandas as pd
            df = pd.read_csv(metrics_csv)
            if len(df) < 100:
                issues.append({"run_id": meta.get("run_id"), "issue": f"metrics.csv only {len(df)} rows"})
                continue
            if df.isna().any().any():
                issues.append({"run_id": meta.get("run_id"), "issue": "NaN in metrics.csv"})
                continue
        except Exception as e:
            issues.append({"run_id": meta.get("run_id"), "issue": f"csv read failed: {e}"})
            continue
        n_valid += 1
    ok = (n_dirs == 36 and n_valid == 36 and not issues)
    _record("30.2_all_36_jobs_completed", ok, {"n_dirs": n_dirs, "n_valid": n_valid,
                                                 "issues": issues[:5]})


# 30.3 — Convergence output valid
def test_30_3() -> None:
    p = ROOT / "results" / "calibration_analysis" / "calibrated_total_steps.json"
    j = _load_json(p)
    cal = j.get("calibrated_steps") if j else None
    ok = bool(j and isinstance(cal, int) and 50_000 <= cal <= 1_000_000)
    _record("30.3_convergence_output_valid", ok, j or f"missing {p}")


# 30.4 — All cells converged
def test_30_4() -> None:
    csv_path = ROOT / "results" / "calibration_analysis" / "convergence_per_run.csv"
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        _record("30.4_all_cells_converged", False, f"could not read {csv_path}: {e}")
        return
    n_total = len(df)
    n_conv  = int(df["converged"].sum()) if "converged" in df.columns else 0
    ok = (n_total == 36 and n_conv == 36)
    _record("30.4_all_cells_converged", ok, {"n_total": n_total, "n_converged": n_conv,
                                              "non_converged_run_ids":
                                              df.loc[~df.get("converged", False), "run_id"].tolist()
                                              if "converged" in df.columns else []})


# 30.5 — YAML updated
def test_30_5() -> None:
    yaml_path = ROOT / "config_frozen_v1.yaml"
    lock_path = ROOT / "config_lock.json"
    cal_path = ROOT / "results" / "calibration_analysis" / "calibrated_total_steps.json"
    cal_steps = None
    try:
        cal_steps = int(json.loads(cal_path.read_text())["calibrated_steps"])
    except Exception as e:
        _record("30.5_yaml_updated", False, f"cannot read calibrated_total_steps.json: {e}")
        return
    yaml_text = yaml_path.read_text() if yaml_path.is_file() else ""
    needles = [f"total_steps: {cal_steps}", f"total_steps: {cal_steps}.0"]
    contains_value = any(n in yaml_text for n in needles)
    # Lock matches YAML hash?
    lock_ok = False
    try:
        from config_loader import check_config_lock
        lock_ok = bool(check_config_lock()["matches"])
    except Exception:
        pass
    ok = contains_value and lock_ok
    _record("30.5_yaml_updated", ok, {"yaml_contains_calibrated": contains_value,
                                       "lock_matches": lock_ok,
                                       "calibrated_steps": cal_steps})


# 30.6 — Existing suite still green (excluding self)
def test_30_6() -> None:
    self_path = str(VERIF / "phase30_calibration.json")
    failing = []
    n_loaded = 0
    for p in sorted(VERIF.glob("phase*.json")):
        if str(p) == self_path:
            continue
        try:
            j = json.loads(Path(p).read_text())
        except Exception as e:
            failing.append({"path": p.name, "error": str(e)}); continue
        n_loaded += 1
        if isinstance(j, dict) and j.get("pass") is False:
            failing.append({"path": p.name})
    ok = (not failing)
    _record("30.6_existing_suite", ok, {"n_phase_files": n_loaded, "failing": failing})


def main() -> int:
    print("=" * 60)
    print("PHASE 30 VERIFICATION")
    print("=" * 60)
    test_30_1(); test_30_2(); test_30_3(); test_30_4(); test_30_5(); test_30_6()
    n_pass = sum(1 for t in results["tests"].values() if t["pass"])
    n_total = len(results["tests"])
    all_pass = n_pass == n_total
    results["pass"] = all_pass
    results["n_pass"] = n_pass
    results["n_total"] = n_total
    out = VERIF / "phase30_calibration.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print("=" * 60)
    print(f"{n_pass}/{n_total} tests passed")
    print(f"ALL_PASS = {all_pass}")
    print(f"Wrote {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
