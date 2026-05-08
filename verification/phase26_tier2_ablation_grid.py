"""Phase 26: Tier 2 ablation grid verification (post Phase 1D).

Validates the orchestrator's Tier 2 sub-grids (2a lambda, 2b occlusion,
2c fusion weights) defined in SPEC_PHASE_1D_TIER2_ABLATION_HOOKS.md.

Tests:
  26.1 Dry-run total job count = 1160 (600 + 400 + 160) with summary line
  26.2 Sub-grid 2a content correctness
  26.3 Sub-grid 2b content correctness (incl. --no_buildings on occOFF)
  26.4 Sub-grid 2c content correctness (8 weight tuples × 20 each)
  26.5 --subgrid filtering produces 600 / 400 / 160 jobs respectively
  26.6 Determinism (two consecutive dry-runs produce identical output)
  26.7 Tier 1 regression: count unchanged at 1680 after Phase 1D edits
  26.8 Existing 27-phase suite still passes
"""
import sys
import os
import re
import json
import subprocess
import traceback
from collections import Counter
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

results = {"phase": "26", "tests": {}, "pass": True}


def _record(name, ok, details=None):
    results["tests"][name] = {"pass": bool(ok)}
    if details:
        results["tests"][name].update(details)
    if not ok:
        results["pass"] = False
    sym = "OK  " if ok else "FAIL"
    print(f"  [{sym}] {name}")


PYTHON_BIN = sys.executable
ORCH_PATH = "experiments/pde/run_full_ablation.py"


def _run_dry(tier: str, subgrid: str | None = None) -> str:
    cmd = [PYTHON_BIN, ORCH_PATH, "--tier", tier, "--dry_run"]
    if subgrid is not None:
        cmd.extend(["--subgrid", subgrid])
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"orchestrator dry_run (tier={tier} subgrid={subgrid}) "
            f"exited {proc.returncode}: {proc.stdout.decode()[:500]}"
        )
    return proc.stdout.decode()


# ──────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"^  (T[0-9a-z]+_[^\s:]+):$")
_TRAIN_RE = re.compile(r"^    TRAIN: (.+)$")


def _parse_jobs(stdout: str) -> list[dict]:
    """Return list of {tag, cmd_train_str} dicts parsed from a dry-run."""
    jobs = []
    cur = None
    for line in stdout.splitlines():
        m = _TAG_RE.match(line)
        if m:
            cur = {"tag": m.group(1)}
            jobs.append(cur)
            continue
        m = _TRAIN_RE.match(line)
        if m and cur is not None:
            cur["cmd_train_str"] = m.group(1)
    return jobs


def _summary_total_tier2(stdout: str) -> int | None:
    m = re.search(r"\[DRY_RUN\] Total Tier 2 jobs: (\d+)", stdout)
    return int(m.group(1)) if m else None


def _summary_breakdown_tier2(stdout: str) -> dict[str, int]:
    out = {}
    for label, key in (
        ("2a (lambda sweep):", "2a"),
        ("2b (occlusion sweep):", "2b"),
        ("2c (fusion weights):", "2c"),
    ):
        m = re.search(rf"\[DRY_RUN\]   {re.escape(label)}\s+(\d+)", stdout)
        if m:
            out[key] = int(m.group(1))
    return out


def _generated_count(stdout: str) -> int | None:
    m = re.search(r"^Generated (\d+) jobs for tier '", stdout, flags=re.MULTILINE)
    return int(m.group(1)) if m else None


# ──────────────────────────────────────────────────────────────────────────
# 26.1 — total + breakdown summary present
# ──────────────────────────────────────────────────────────────────────────
def test_26_1():
    issues = []
    summary_total = None
    summary_breakdown = {}
    try:
        stdout = _run_dry("2")
        summary_total = _summary_total_tier2(stdout)
        summary_breakdown = _summary_breakdown_tier2(stdout)
        if summary_total != 1160:
            issues.append(f"Total Tier 2 summary {summary_total!r}, expected 1160")
        if summary_breakdown.get("2a") != 600:
            issues.append(f"2a breakdown {summary_breakdown.get('2a')!r}, expected 600")
        if summary_breakdown.get("2b") != 400:
            issues.append(f"2b breakdown {summary_breakdown.get('2b')!r}, expected 400")
        if summary_breakdown.get("2c") != 160:
            issues.append(f"2c breakdown {summary_breakdown.get('2c')!r}, expected 160")
        gen_count = _generated_count(stdout)
        if gen_count != 1160:
            issues.append(f"'Generated N jobs' line says {gen_count!r}, expected 1160")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("26.1_total_count_summary", len(issues) == 0, {
        "issues": issues,
        "summary_total": summary_total,
        "summary_breakdown": summary_breakdown,
    })


# ──────────────────────────────────────────────────────────────────────────
# 26.2 — sub-grid 2a content correctness
# ──────────────────────────────────────────────────────────────────────────
TIER2_PDE_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]
TIER2A_SCENARIOS = ["1a", "2_dense"]
TIER2A_LAMBDA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
TIER2_MANEUVERS = ["stem_right", "right_left"]
TIER2_SEEDS = [42, 123, 456, 789, 7]
# Sort longest-first so e.g. "soft_hjb_aux" wins over "hjb_aux".
_METHOD_BY_LEN = sorted(TIER2_PDE_METHODS, key=len, reverse=True)


def _method_in_tag(tag: str) -> str | None:
    for m in _METHOD_BY_LEN:
        if f"_{m}_" in f"_{tag}_":
            return m
    return None


def _scenario_in_tag(tag: str, scenarios: list[str]) -> str | None:
    # Tag layout: T2{x}_{scenario}_{maneuver}_{method}_…
    parts = tag.split("_")
    # We need to handle "2_dense" (two-token scenario) vs "1a" (one-token).
    # Greedy match of scenarios sorted by length-desc.
    for sc in sorted(scenarios, key=len, reverse=True):
        sc_tokens = sc.split("_")
        if parts[1:1+len(sc_tokens)] == sc_tokens:
            return sc
    return None


def _maneuver_in_tag(tag: str, scenarios: list[str], maneuvers: list[str]) -> str | None:
    parts = tag.split("_")
    sc = _scenario_in_tag(tag, scenarios)
    if sc is None:
        return None
    sc_tokens = sc.split("_")
    after = parts[1+len(sc_tokens):]
    for mv in sorted(maneuvers, key=len, reverse=True):
        mv_tokens = mv.split("_")
        if after[:len(mv_tokens)] == mv_tokens:
            return mv
    return None


def test_26_2():
    issues = []
    counts = {}
    try:
        stdout = _run_dry("2", subgrid="2a")
        jobs = _parse_jobs(stdout)
        t2a = [j for j in jobs if j["tag"].startswith("T2a_")]
        if len(t2a) != 600:
            issues.append(f"Expected 600 T2a_* jobs, got {len(t2a)}")
        # Counters per dimension.
        method_c = Counter()
        scenario_c = Counter()
        maneuver_c = Counter()
        seed_c = Counter()
        lambda_c = Counter()
        for j in t2a:
            tag = j["tag"]
            m = _method_in_tag(tag)
            sc = _scenario_in_tag(tag, TIER2A_SCENARIOS)
            mv = _maneuver_in_tag(tag, TIER2A_SCENARIOS, TIER2_MANEUVERS)
            sd_m = re.search(r"_s(-?\d+)$", tag)
            lam_m = re.search(r"_lam([\d.]+)_s", tag)
            method_c[m] += 1
            scenario_c[sc] += 1
            maneuver_c[mv] += 1
            if sd_m:
                seed_c[int(sd_m.group(1))] += 1
            if lam_m:
                # Float key to compare against TIER2A_LAMBDA_VALUES.
                lambda_c[float(lam_m.group(1))] += 1
        counts = {
            "by_method": dict(method_c),
            "by_scenario": dict(scenario_c),
            "by_maneuver": dict(maneuver_c),
            "by_seed": dict(seed_c),
            "by_lambda": {f"{k}": v for k, v in lambda_c.items()},
        }
        for m in TIER2_PDE_METHODS:
            if method_c[m] != 120:
                issues.append(f"method {m}: {method_c[m]}, expected 120")
        for sc in TIER2A_SCENARIOS:
            if scenario_c[sc] != 300:
                issues.append(f"scenario {sc}: {scenario_c[sc]}, expected 300")
        for mv in TIER2_MANEUVERS:
            if maneuver_c[mv] != 300:
                issues.append(f"maneuver {mv}: {maneuver_c[mv]}, expected 300")
        for sd in TIER2_SEEDS:
            if seed_c[sd] != 120:
                issues.append(f"seed {sd}: {seed_c[sd]}, expected 120")
        for lam in TIER2A_LAMBDA_VALUES:
            if lambda_c[lam] != 100:
                issues.append(f"lambda {lam}: {lambda_c[lam]}, expected 100")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("26.2_subgrid_2a_content", len(issues) == 0, {
        "issues": issues, "counts": counts,
    })


# ──────────────────────────────────────────────────────────────────────────
# 26.3 — sub-grid 2b content correctness
# ──────────────────────────────────────────────────────────────────────────
TIER2B_SCENARIOS = ["1a", "1b", "3", "2_dense"]


def test_26_3():
    issues = []
    counts = {}
    try:
        stdout = _run_dry("2", subgrid="2b")
        jobs = _parse_jobs(stdout)
        t2b = [j for j in jobs if j["tag"].startswith("T2b_")]
        if len(t2b) != 400:
            issues.append(f"Expected 400 T2b_* jobs, got {len(t2b)}")
        method_c = Counter()
        scenario_c = Counter()
        maneuver_c = Counter()
        seed_c = Counter()
        occ_c = Counter()
        nb_violations_on = []
        nb_missing_off = []
        for j in t2b:
            tag = j["tag"]
            m = _method_in_tag(tag)
            sc = _scenario_in_tag(tag, TIER2B_SCENARIOS)
            mv = _maneuver_in_tag(tag, TIER2B_SCENARIOS, TIER2_MANEUVERS)
            sd_m = re.search(r"_s(-?\d+)$", tag)
            occ_m = re.search(r"_occ(ON|OFF)_s", tag)
            method_c[m] += 1
            scenario_c[sc] += 1
            maneuver_c[mv] += 1
            if sd_m:
                seed_c[int(sd_m.group(1))] += 1
            if occ_m:
                occ_c[occ_m.group(1)] += 1
            train = j.get("cmd_train_str", "")
            if occ_m and occ_m.group(1) == "ON":
                if "--no_buildings" in train:
                    nb_violations_on.append(tag)
            elif occ_m and occ_m.group(1) == "OFF":
                if "--no_buildings" not in train:
                    nb_missing_off.append(tag)
        counts = {
            "by_method": dict(method_c),
            "by_scenario": dict(scenario_c),
            "by_maneuver": dict(maneuver_c),
            "by_seed": dict(seed_c),
            "by_occ": dict(occ_c),
        }
        for m in TIER2_PDE_METHODS:
            if method_c[m] != 80:
                issues.append(f"method {m}: {method_c[m]}, expected 80")
        for sc in TIER2B_SCENARIOS:
            if scenario_c[sc] != 100:
                issues.append(f"scenario {sc}: {scenario_c[sc]}, expected 100")
        for mv in TIER2_MANEUVERS:
            if maneuver_c[mv] != 200:
                issues.append(f"maneuver {mv}: {maneuver_c[mv]}, expected 200")
        for sd in TIER2_SEEDS:
            if seed_c[sd] != 80:
                issues.append(f"seed {sd}: {seed_c[sd]}, expected 80")
        if occ_c["ON"] != 200:
            issues.append(f"occON: {occ_c['ON']}, expected 200")
        if occ_c["OFF"] != 200:
            issues.append(f"occOFF: {occ_c['OFF']}, expected 200")
        if nb_violations_on:
            issues.append(
                f"{len(nb_violations_on)} occON jobs unexpectedly include --no_buildings "
                f"(first: {nb_violations_on[0]})"
            )
        if nb_missing_off:
            issues.append(
                f"{len(nb_missing_off)} occOFF jobs missing --no_buildings "
                f"(first: {nb_missing_off[0]})"
            )
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("26.3_subgrid_2b_content", len(issues) == 0, {
        "issues": issues, "counts": counts,
    })


# ──────────────────────────────────────────────────────────────────────────
# 26.4 — sub-grid 2c content correctness
# ──────────────────────────────────────────────────────────────────────────
TIER2C_SCENARIOS = ["1a", "2_dense"]
TIER2C_FUSION_WEIGHTS = [
    (1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0),
    (2.0, 1.0), (1.0, 2.0), (3.0, 1.0), (1.0, 3.0),
]


def test_26_4():
    issues = []
    counts = {}
    try:
        stdout = _run_dry("2", subgrid="2c")
        jobs = _parse_jobs(stdout)
        t2c = [j for j in jobs if j["tag"].startswith("T2c_")]
        if len(t2c) != 160:
            issues.append(f"Expected 160 T2c_* jobs, got {len(t2c)}")
        scenario_c = Counter()
        maneuver_c = Counter()
        seed_c = Counter()
        weight_c = Counter()
        non_fusion = []
        weight_arg_mismatch = []
        for j in t2c:
            tag = j["tag"]
            train = j.get("cmd_train_str", "")
            if "train_fusion_aux.py" not in train:
                non_fusion.append(tag)
            sc = _scenario_in_tag(tag, TIER2C_SCENARIOS)
            mv = _maneuver_in_tag(tag, TIER2C_SCENARIOS, TIER2_MANEUVERS)
            sd_m = re.search(r"_s(-?\d+)$", tag)
            w_m = re.search(r"_w([\d.]+)_([\d.]+)_s", tag)
            scenario_c[sc] += 1
            maneuver_c[mv] += 1
            if sd_m:
                seed_c[int(sd_m.group(1))] += 1
            if w_m:
                w_o, w_s = float(w_m.group(1)), float(w_m.group(2))
                weight_c[(w_o, w_s)] += 1
                # Verify the cmd_train carries the matching --w_optimality / --w_safety.
                wo_match = re.search(r"--w_optimality\s+([\d.]+)", train)
                ws_match = re.search(r"--w_safety\s+([\d.]+)", train)
                if (
                    wo_match is None or ws_match is None
                    or abs(float(wo_match.group(1)) - w_o) > 1e-9
                    or abs(float(ws_match.group(1)) - w_s) > 1e-9
                ):
                    weight_arg_mismatch.append(tag)
        counts = {
            "by_scenario": dict(scenario_c),
            "by_maneuver": dict(maneuver_c),
            "by_seed": dict(seed_c),
            "by_weight_tuple": {f"{k[0]}_{k[1]}": v for k, v in weight_c.items()},
        }
        if non_fusion:
            issues.append(
                f"{len(non_fusion)} jobs do not call train_fusion_aux.py "
                f"(first: {non_fusion[0]})"
            )
        for sc in TIER2C_SCENARIOS:
            if scenario_c[sc] != 80:
                issues.append(f"scenario {sc}: {scenario_c[sc]}, expected 80")
        for mv in TIER2_MANEUVERS:
            if maneuver_c[mv] != 80:
                issues.append(f"maneuver {mv}: {maneuver_c[mv]}, expected 80")
        for sd in TIER2_SEEDS:
            if seed_c[sd] != 32:
                issues.append(f"seed {sd}: {seed_c[sd]}, expected 32")
        for tup in TIER2C_FUSION_WEIGHTS:
            if weight_c[tup] != 20:
                issues.append(f"weights {tup}: {weight_c[tup]}, expected 20")
        if weight_arg_mismatch:
            issues.append(
                f"{len(weight_arg_mismatch)} 2c jobs' --w_optimality/--w_safety "
                f"don't match their tag (first: {weight_arg_mismatch[0]})"
            )
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("26.4_subgrid_2c_content", len(issues) == 0, {
        "issues": issues, "counts": counts,
    })


# ──────────────────────────────────────────────────────────────────────────
# 26.5 — subgrid filtering CLI
# ──────────────────────────────────────────────────────────────────────────
def test_26_5():
    issues = []
    by_subgrid = {}
    try:
        for sg, expected in (("2a", 600), ("2b", 400), ("2c", 160)):
            stdout = _run_dry("2", subgrid=sg)
            n = _generated_count(stdout)
            by_subgrid[sg] = n
            if n != expected:
                issues.append(
                    f"--subgrid {sg}: 'Generated N' = {n!r}, expected {expected}"
                )
            jobs = _parse_jobs(stdout)
            wrong_prefix = [
                j["tag"] for j in jobs if not j["tag"].startswith(f"T{sg}_")
            ]
            if wrong_prefix:
                issues.append(
                    f"--subgrid {sg}: {len(wrong_prefix)} jobs have wrong prefix "
                    f"(first: {wrong_prefix[0]})"
                )
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("26.5_subgrid_filtering", len(issues) == 0, {
        "issues": issues, "by_subgrid": by_subgrid,
    })


# ──────────────────────────────────────────────────────────────────────────
# 26.6 — determinism
# ──────────────────────────────────────────────────────────────────────────
def test_26_6():
    issues = []
    try:
        out1 = _run_dry("2")
        out2 = _run_dry("2")
        if out1 != out2:
            # Emit a small diff hint (first differing line).
            lines1 = out1.splitlines()
            lines2 = out2.splitlines()
            n = min(len(lines1), len(lines2))
            first_diff = None
            for i in range(n):
                if lines1[i] != lines2[i]:
                    first_diff = i
                    break
            issues.append(
                f"two consecutive dry-runs differ; "
                f"first diff at line {first_diff} "
                f"({len(lines1)} vs {len(lines2)} total lines)"
            )
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("26.6_determinism", len(issues) == 0, {"issues": issues})


# ──────────────────────────────────────────────────────────────────────────
# 26.7 — no regression of Tier 1 (count = 1680)
# ──────────────────────────────────────────────────────────────────────────
def test_26_7():
    issues = []
    n = None
    try:
        stdout = _run_dry("1")
        n = _generated_count(stdout)
        # Phase 1C added fusion_aux as 7th method, lifting Tier 1 count to
        # 1680 (12 combos × 7 methods × 10 seeds × 2 intents). Phase 1D
        # must keep this number unchanged.
        if n != 1680:
            issues.append(f"Tier 1 count = {n!r}, expected 1680")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    _record("26.7_tier1_regression", len(issues) == 0, {
        "issues": issues, "tier1_count": n,
    })


# ──────────────────────────────────────────────────────────────────────────
# 26.8 — existing 27-phase suite still passes
# ──────────────────────────────────────────────────────────────────────────
def test_26_8():
    ver_dir = os.path.dirname(os.path.abspath(__file__))
    phases = {}
    failed_load = []
    for path in sorted(glob(os.path.join(ver_dir, "phase*.json"))):
        name = os.path.basename(path).replace(".json", "")
        if name == "phase26_tier2_ablation_grid":
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
    ok = all_pass and n_phases >= 27 and not failed_load
    _record("26.8_existing_suite", ok, {
        "n_phases": n_phases,
        "all_pass": all_pass,
        "failed_phases": failed_phases,
        "load_failures": failed_load,
    })


def main():
    print("==== PHASE 26: TIER 2 ABLATION GRID VERIFICATION ====")
    test_26_1()
    test_26_2()
    test_26_3()
    test_26_4()
    test_26_5()
    test_26_6()
    test_26_7()
    test_26_8()

    out_path = os.path.join(os.path.dirname(__file__),
                            "phase26_tier2_ablation_grid.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print("ALL_PASS =", results["pass"])
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
