"""Phase 29 — Comprehensive verification gate before compute (SPEC_PHASE_2 Step 11).

Aggregates the 10 sub-tests of the gate into a single pass/fail.
Each sub-test reads its own structured JSON written by an upstream Step:

  29.1 orchestrator_filter_flags    (Step 2; checked live in this script)
  29.2 schema_consistency           (Step 5; verification/phase29_schema_consistency.json)
  29.3 obs_dim_audit_documented     (Step 3; verification/phase29_obs_dim_audit.json + grep)
  29.4 tier3_reconciliation         (Step 4; verification/phase29_tier3_reconciliation.json)
  29.5 synthetic_stats_path         (Step 6; verification/phase29_synthetic_stats.json)
  29.6 e2e_orchestrator_smoke       (Step 7; verification/phase29_e2e_smoke.json)
  29.7 checkpoint_load_round_trip   (Step 8; verification/phase29_ckpt_round_trip.json)
  29.8 determinism_gpu              (Step 9; verification/phase29_determinism_gpu.json)
  29.9 determinism_cpu_strict       (Step 9; verification/phase29_determinism_cpu.json)
  29.10 existing_suite              (Step 10; aggregate_report.py shows ALL_PASS=True)

Usage:
    python3 verification/phase29_verification_gate.py
"""
from __future__ import annotations
import os, sys, json, subprocess
from pathlib import Path

VERIF = Path(__file__).resolve().parent
ROOT = VERIF.parent

results: dict = {"phase": "29", "tests": {}}


def _record(test_id: str, ok: bool, details: dict | str = "") -> None:
    badge = "OK  " if ok else "FAIL"
    print(f"[{badge}] {test_id}")
    if details and not ok:
        print(f"       details: {details!r:.300}")
    results["tests"][test_id] = {"pass": bool(ok), "details": details}


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── 29.1 — Orchestrator filter flags (live recheck) ─────────────────────────
def test_29_1() -> None:
    cmd = [
        "python3", "experiments/pde/run_full_ablation.py",
        "--tier", "1", "--seeds", "42",
        "--methods", "drppo", "hjb_aux",
        "--scenarios", "1a", "--maneuvers", "stem_right",
        "--intents", "false", "--dry_run",
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    text = p.stdout + p.stderr
    n = 0
    for line in text.splitlines():
        if line.startswith("Generated ") and "jobs for tier" in line:
            try:
                n = int(line.split()[1])
            except Exception:
                n = -1
    ok = (p.returncode == 0 and n == 2)
    _record("29.1_orchestrator_filter_flags", ok, {"exit": p.returncode, "n_jobs": n})


# ── 29.2 — Schema consistency ───────────────────────────────────────────────
def test_29_2() -> None:
    j = _load_json(VERIF / "phase29_schema_consistency.json")
    ok = bool(j and j.get("pass") and j.get("consistent") and not j.get("issues"))
    _record("29.2_schema_consistency", ok, j or "missing JSON")


# ── 29.3 — Obs dim audit documented ─────────────────────────────────────────
def test_29_3() -> None:
    j = _load_json(VERIF / "phase29_obs_dim_audit.json")
    audit_json_ok = bool(j and j.get("pass") and j.get("plus_one_semantic_name") == "d_pothole")

    # Verify the audit comment is actually in env/sumo_env.py near the concat site.
    src = (ROOT / "env" / "sumo_env.py").read_text()
    grep_ok = "Phase 2 audit" in src and "d_pothole" in src and "obs_dim_base" in src

    # Verify YAML matches.
    yaml_text = (ROOT / "config_frozen_v1.yaml").read_text()
    yaml_ok = "obs_dim_base: 135" in yaml_text

    ok = audit_json_ok and grep_ok and yaml_ok
    _record("29.3_obs_dim_audit_documented", ok, {
        "audit_json_pass": audit_json_ok, "comment_in_source": grep_ok, "yaml_match": yaml_ok,
    })


# ── 29.4 — Tier 3 reconciliation documented ─────────────────────────────────
def test_29_4() -> None:
    md = (VERIF / "MASTER_PLAN_RECONCILIATION.md")
    md_exists = md.is_file()
    md_text = md.read_text() if md_exists else ""
    contents_ok = (
        md_exists
        and "275" in md_text
        and "T3S_" in md_text
        and "T3B_" in md_text
        and "T3D_" in md_text
        and "--tier 3 --dry_run" in md_text
    )
    j = _load_json(VERIF / "phase29_tier3_reconciliation.json")
    json_ok = bool(j and j.get("pass") and j.get("tier3_total") == 275)
    ok = contents_ok and json_ok
    _record("29.4_tier3_reconciliation", ok, {"md_ok": contents_ok, "json_ok": json_ok})


# ── 29.5 — Synthetic stats path ─────────────────────────────────────────────
def test_29_5() -> None:
    j = _load_json(VERIF / "phase29_synthetic_stats.json")
    ok = bool(j and j.get("pass") and j.get("n_rows") == 5 and not j.get("mismatches"))
    _record("29.5_synthetic_stats_path", ok, j or "missing")


# ── 29.6 — End-to-end orchestrator smoke ────────────────────────────────────
def test_29_6() -> None:
    j = _load_json(VERIF / "phase29_e2e_smoke.json")
    ok = bool(
        j
        and j.get("pass")
        and j.get("n_jobs_completed") == 6
        and j.get("n_jobs_failed") == 0
    )
    _record("29.6_e2e_orchestrator_smoke", ok, j or "missing")


# ── 29.7 — Checkpoint load round-trip ───────────────────────────────────────
def test_29_7() -> None:
    j = _load_json(VERIF / "phase29_ckpt_round_trip.json")
    ok = bool(j and j.get("pass") and len(j.get("per_method", [])) == 6
              and all(m.get("eval_ok") for m in j.get("per_method", [])))
    _record("29.7_checkpoint_load_round_trip", ok, j or "missing")


# ── 29.8 — Determinism (GPU, relaxed 3-sigma per Prompt11) ──────────────────
# Per-spec decision (see MASTER_PLAN_RECONCILIATION.md "Determinism Notes"):
# strict bit-reproducibility is unachievable with SUMO-coupled training due to
# TraCI subprocess timing. Relaxed test compares two consecutive runs at the
# same seed against tolerances measured from 5 identical-config runs (3*sigma).
def test_29_8() -> None:
    j = _load_json(VERIF / "phase29_determinism_gpu.json")
    ok = bool(j and j.get("pass") and j.get("int_cols_match_exactly"))
    _record("29.8_determinism_gpu", ok, j or "missing")


# ── 29.9 — Determinism (CPU, relaxed 3-sigma per Prompt11) ──────────────────
# Same tolerance as 29.8; CPU/GPU were collapsed into one test per Prompt11.
def test_29_9() -> None:
    j = _load_json(VERIF / "phase29_determinism_cpu.json")
    ok = bool(j and j.get("pass") and j.get("int_cols_match_exactly"))
    _record("29.9_determinism_cpu_strict", ok, j or "missing")


# ── 29.10 — Existing 31-phase suite still green ─────────────────────────────
# Self-recursion avoidance: aggregate_report.py loads every verification/phase*.json
# including phase29_verification_gate.json (this gate's own aggregate). Reading our
# own pass-state would create a loop. We instead enumerate all *other* phase JSONs
# and assert each is pass=True (or has no pass field).
def test_29_10() -> None:
    import glob
    self_path = str(VERIF / "phase29_verification_gate.json")
    failing = []
    n_loaded = 0
    for path in sorted(glob.glob(str(VERIF / "phase*.json"))):
        if path == self_path:
            continue
        try:
            j = json.loads(Path(path).read_text())
        except Exception as e:
            failing.append({"path": path, "load_error": str(e)})
            continue
        n_loaded += 1
        if isinstance(j, dict) and j.get("pass") is False:
            failing.append({"path": Path(path).name})
    ok = (not failing)
    _record("29.10_existing_suite", ok, {"n_phase_files": n_loaded, "failing": failing})


def main() -> int:
    print("=" * 60)
    print("PHASE 29 VERIFICATION GATE")
    print("=" * 60)
    test_29_1()
    test_29_2()
    test_29_3()
    test_29_4()
    test_29_5()
    test_29_6()
    test_29_7()
    test_29_8()
    test_29_9()
    test_29_10()

    n_pass = sum(1 for t in results["tests"].values() if t["pass"])
    n_total = len(results["tests"])
    all_pass = (n_pass == n_total)
    results["pass"] = all_pass
    results["n_pass"] = n_pass
    results["n_total"] = n_total

    out_path = VERIF / "phase29_verification_gate.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("=" * 60)
    print(f"{n_pass}/{n_total} tests passed.")
    print(f"ALL_PASS = {all_pass}")
    print(f"Wrote {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
