"""End-to-end smoke test for all 5 methods.

Runs a short training (512 steps), saves checkpoint, loads it,
runs 5 eval episodes, and verifies all outputs exist.

Usage:
    python experiments/pde/smoke_test.py --scenario 1a --ego_maneuver stem_right
"""

import argparse
import os
import sys
import subprocess
import time

METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]
STEPS = 512


def run_smoke_test(scenario, maneuver):
    out_dir = f"results/smoke_test_{scenario}_{maneuver}"
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for method in METHODS:
        print(f"\n{'=' * 60}")
        print(f"Smoke test: {method} on {scenario}/{maneuver}")
        print(f"{'=' * 60}")

        if method == "drppo":
            script = "experiments/pde/train_drppo_baseline.py"
        else:
            script = f"experiments/pde/train_{method}.py"

        t0 = time.time()

        # Train
        cmd = [
            sys.executable, script,
            "--scenario", scenario,
            "--ego_maneuver", maneuver,
            "--total_steps", str(STEPS),
            "--out_dir", out_dir,
            "--seed", "42",
        ]
        print(f"  Training: {' '.join(cmd)}")
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if ret.returncode != 0:
            print(f"  FAILED (train): {ret.stderr[-500:]}")
            results[method] = {"status": "TRAIN_FAIL", "error": ret.stderr[-200:]}
            continue

        train_time = time.time() - t0

        # Check outputs
        ckpt = os.path.join(out_dir, f"model_{method}_{scenario}_{maneuver}.pt")
        csv_file = os.path.join(out_dir, f"train_{method}_{scenario}_{maneuver}.csv")
        meta_file = os.path.join(out_dir, f"meta_{method}_{scenario}_{maneuver}.json")

        missing = []
        if not os.path.isfile(ckpt):
            missing.append("checkpoint")
        if not os.path.isfile(csv_file):
            missing.append("csv")
        if not os.path.isfile(meta_file):
            missing.append("meta.json")

        if missing:
            print(f"  FAILED (missing outputs): {missing}")
            results[method] = {"status": "OUTPUT_FAIL", "missing": missing}
            continue

        # Eval
        eval_cmd = [
            sys.executable, "experiments/pde/eval.py",
            "--checkpoint", ckpt,
            "--method", method,
            "--scenario", scenario,
            "--ego_maneuver", maneuver,
            "--episodes", "5",
            "--seeds", "42",
            "--out_dir", out_dir,
            "--save_failures",
        ]
        ret_eval = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)

        if ret_eval.returncode != 0:
            print(f"  FAILED (eval): {ret_eval.stderr[-500:]}")
            results[method] = {"status": "EVAL_FAIL", "error": ret_eval.stderr[-200:]}
            continue

        print(f"  OK ({train_time:.0f}s train)")
        results[method] = {"status": "OK", "train_time": train_time}

    # Summary
    print(f"\n{'=' * 60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'=' * 60}")
    for method, r in results.items():
        status = r["status"]
        extra = f" ({r.get('train_time', 0):.0f}s)" if status == "OK" else f" -- {r.get('error', r.get('missing', ''))}"
        symbol = "OK" if status == "OK" else "FAIL"
        print(f"  [{symbol}] {method}: {status}{extra}")

    n_ok = sum(1 for r in results.values() if r["status"] == "OK")
    print(f"\n{n_ok}/{len(METHODS)} methods passed.")
    return n_ok == len(METHODS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="1a")
    parser.add_argument("--ego_maneuver", default="stem_right")
    args = parser.parse_args()

    success = run_smoke_test(args.scenario, args.ego_maneuver)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
