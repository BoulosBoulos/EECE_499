"""Phase 22: Sigma-aligned style relabeling verification (post SPEC_PHASE_0C).

Verifies the new label dictionaries (CAR_STYLE_LABELS, MOTO_STYLE_LABELS,
PED_STYLE_LABELS) are consistent with the sigma-aligned breakpoints and that
the v5 LSTM trained on the new labels meets the accuracy gate.

Tests:
  22.1 Label dict consistency: every style key exists in the corresponding
       *_STYLE_PARAMS table (or PED_SYNTHETIC_SIGMA for peds), labels are in
       {0,1,2}, and the breakpoints hold:
         sigma <= 0.10 -> 0;  0.10 < sigma <= 0.25 -> 1;  sigma > 0.25 -> 2.
  22.2 Class distribution: 1000 BehaviorSampler.sample() draws yield all
       three classes with at least 100 samples each (per agent type).
  22.3 V5 style accuracy gate (the gate): overall style accuracy >= 65%,
       per-class normal (class 1) accuracy >= 30%, intent accuracy >= 80%.
  22.4 Existing 24-phase suite (incl. phase 21) still passes.
"""
import sys, os, json, math, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

results = {"phase": "22", "tests": {}, "pass": True}


def _record(name, ok, details=None):
    results["tests"][name] = {"pass": bool(ok)}
    if details:
        results["tests"][name].update(details)
    if not ok:
        results["pass"] = False
    sym = "OK  " if ok else "FAIL"
    print(f"  [{sym}] {name}: {details if details else ''}")


def _expected_label_from_sigma(sigma: float) -> int:
    if sigma <= 0.10:
        return 0
    if sigma <= 0.25:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Test 22.1 -- Label dict consistency
# ---------------------------------------------------------------------------
def test_22_1():
    """Every style in *_STYLE_LABELS exists in its parameter table, all values
    are in {0,1,2}, and breakpoints (sigma -> label) hold for every style.
    """
    try:
        from scenario.behavior_sampler import (
            CAR_STYLE_LABELS, MOTO_STYLE_LABELS, PED_STYLE_LABELS,
            CAR_STYLE_PARAMS, MOTO_STYLE_PARAMS, PED_STYLE_PARAMS,
            PED_SYNTHETIC_SIGMA,
        )

        problems = []

        for name, label in CAR_STYLE_LABELS.items():
            if name not in CAR_STYLE_PARAMS:
                problems.append(f"car '{name}' label={label} but not in CAR_STYLE_PARAMS")
                continue
            if label not in (0, 1, 2):
                problems.append(f"car '{name}' label={label} not in {{0,1,2}}")
            sigma = CAR_STYLE_PARAMS[name]["sigma"]
            expected = _expected_label_from_sigma(sigma)
            if label != expected:
                problems.append(f"car '{name}' sigma={sigma} expects {expected}, got {label}")

        for name, label in MOTO_STYLE_LABELS.items():
            if name not in MOTO_STYLE_PARAMS:
                problems.append(f"moto '{name}' label={label} but not in MOTO_STYLE_PARAMS")
                continue
            if label not in (0, 1, 2):
                problems.append(f"moto '{name}' label={label} not in {{0,1,2}}")
            sigma = MOTO_STYLE_PARAMS[name]["sigma"]
            expected = _expected_label_from_sigma(sigma)
            if label != expected:
                problems.append(f"moto '{name}' sigma={sigma} expects {expected}, got {label}")

        for name, label in PED_STYLE_LABELS.items():
            if name not in PED_STYLE_PARAMS:
                problems.append(f"ped '{name}' label={label} but not in PED_STYLE_PARAMS")
                continue
            if label not in (0, 1, 2):
                problems.append(f"ped '{name}' label={label} not in {{0,1,2}}")
            if name not in PED_SYNTHETIC_SIGMA:
                problems.append(f"ped '{name}' has label but not in PED_SYNTHETIC_SIGMA")
                continue
            sigma = PED_SYNTHETIC_SIGMA[name]
            expected = _expected_label_from_sigma(sigma)
            if label != expected:
                problems.append(f"ped '{name}' synth_sigma={sigma} expects {expected}, got {label}")

        # Also verify coverage: every style in *_STYLE_PARAMS should have a label
        for name in CAR_STYLE_PARAMS:
            if name not in CAR_STYLE_LABELS:
                problems.append(f"car '{name}' has params but no label entry")
        for name in MOTO_STYLE_PARAMS:
            if name not in MOTO_STYLE_LABELS:
                problems.append(f"moto '{name}' has params but no label entry")
        for name in PED_STYLE_PARAMS:
            if name not in PED_STYLE_LABELS:
                problems.append(f"ped '{name}' has params but no label entry")

        ok = (len(problems) == 0)
        _record("22.1_label_consistency", ok, {
            "n_problems": len(problems),
            "problems": problems,
            "n_car_styles": len(CAR_STYLE_LABELS),
            "n_moto_styles": len(MOTO_STYLE_LABELS),
            "n_ped_styles": len(PED_STYLE_LABELS),
        })
    except Exception as e:
        _record("22.1_label_consistency", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=3),
        })


# ---------------------------------------------------------------------------
# Test 22.2 -- Class distribution
# ---------------------------------------------------------------------------
def test_22_2():
    """1000 sample() draws yield all three classes for car/moto/ped styles
    with at least 100 each.
    """
    try:
        from scenario.behavior_sampler import BehaviorSampler

        rng = np.random.RandomState(42)
        sampler = BehaviorSampler(rng=rng)
        car_counts = [0, 0, 0]
        ped_counts = [0, 0, 0]
        moto_counts = [0, 0, 0]
        N = 1000
        for _ in range(N):
            cfg = sampler.sample(
                has_car=True, has_ped=True, has_moto=True, has_pothole=False,
                bar_len=50.0, stem_len=60.0, ego_maneuver="stem_right",
            )
            car_counts[cfg.car_style_label] += 1
            ped_counts[cfg.ped_style_label] += 1
            moto_counts[cfg.moto_style_label] += 1

        car_ok = all(c >= 100 for c in car_counts)
        ped_ok = all(c >= 100 for c in ped_counts)
        moto_ok = all(c >= 100 for c in moto_counts)
        ok = car_ok and ped_ok and moto_ok
        _record("22.2_class_distribution", ok, {
            "n_samples": N,
            "car_counts": car_counts,
            "ped_counts": ped_counts,
            "moto_counts": moto_counts,
            "all_classes_>=_100": ok,
        })
    except Exception as e:
        _record("22.2_class_distribution", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=3),
        })


# ---------------------------------------------------------------------------
# Test 22.3 -- V5 style accuracy gate (THE GATE)
# ---------------------------------------------------------------------------
def test_22_3():
    """Read v5 per_class_breakdown.json and verify:
      - overall style accuracy >= 0.65
      - per-class "normal" (class 1) accuracy >= 0.30
      - intent overall accuracy >= 0.80
      - no class with 0 validation samples
    """
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bd_path = os.path.join(repo_root, "results", "intent_v9_final",
                               "per_class_breakdown.json")
        if not os.path.isfile(bd_path):
            _record("22.3_v5_accuracy_gate", False, {
                "error": f"per_class_breakdown.json not found at {bd_path}",
            })
            return

        with open(bd_path) as f:
            bd = json.load(f)

        style = bd.get("style", {})
        intent = bd.get("intent", {})
        per_class = style.get("per_class_acc", [0, 0, 0])
        counts = style.get("counts", [0, 0, 0])
        overall_style = float(style.get("overall", 0.0))
        overall_intent = float(intent.get("overall", 0.0))
        normal_acc = float(per_class[1]) if len(per_class) > 1 else 0.0

        no_empty_class = all(int(c) > 0 for c in counts)
        gate_passes = overall_style >= 0.65
        normal_recovers = normal_acc >= 0.30
        intent_passes = overall_intent >= 0.80

        ok = gate_passes and normal_recovers and intent_passes and no_empty_class
        _record("22.3_v5_accuracy_gate", ok, {
            "overall_style_accuracy": overall_style,
            "overall_intent_accuracy": overall_intent,
            "style_per_class_acc": [float(x) for x in per_class],
            "style_counts": [int(x) for x in counts],
            "normal_class_accuracy": normal_acc,
            "gate_overall_style_ge_65": gate_passes,
            "normal_class_ge_30": normal_recovers,
            "intent_ge_80": intent_passes,
            "no_empty_class": no_empty_class,
        })
    except Exception as e:
        _record("22.3_v5_accuracy_gate", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=3),
        })


# ---------------------------------------------------------------------------
# Test 22.4 -- Existing 24-phase suite still passes
# ---------------------------------------------------------------------------
def test_22_4():
    """Re-aggregate every phase*.json (excluding self) and confirm ALL_PASS.
    Expects at least 24 prior phase JSONs (the 23 from before phase21 + phase21).
    """
    try:
        import glob
        ver_dir = os.path.dirname(os.path.abspath(__file__))
        phases = {}
        for path in sorted(glob.glob(os.path.join(ver_dir, "phase*.json"))):
            name = os.path.basename(path).replace(".json", "")
            if name == "phase22_relabel_verification":
                continue  # don't include self
            with open(path) as f:
                phases[name] = json.load(f)
        all_pass = all(v.get("pass", True) is True
                       for v in phases.values() if isinstance(v, dict))
        n_phases = len(phases)
        failed = [n for n, v in phases.items()
                  if isinstance(v, dict) and v.get("pass", True) is False]
        ok = all_pass and n_phases >= 24
        _record("22.4_existing_suite", ok, {
            "n_phases": n_phases,
            "all_pass": all_pass,
            "failed_phases": failed,
        })
    except Exception as e:
        _record("22.4_existing_suite", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=2),
        })


def main():
    print("==== PHASE 22: SIGMA-ALIGNED RELABEL VERIFICATION ====")
    test_22_1()
    test_22_2()
    test_22_3()
    test_22_4()
    out_path = os.path.join(os.path.dirname(__file__),
                            "phase22_relabel_verification.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print("ALL_PASS =", results["pass"])
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
