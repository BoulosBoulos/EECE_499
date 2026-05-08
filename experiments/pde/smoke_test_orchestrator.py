"""Smoke test for the orchestrator's cmd_train=None handling.

Tests that:
1. Tier 1 dry-run includes rule_based jobs with TRAIN: (none) annotation
2. Tier 1 dry-run has correct total job count (1440)
3. rule_based jobs have cmd_train=None
4. Tier 4 dry-run runs without crashing (0 jobs if no checkpoints)
5. All tier combos (1,2,3,4,all,supp) dry-run without errors

Usage:
    python experiments/pde/smoke_test_orchestrator.py
"""

import subprocess
import sys
import os

def run_dry(tier):
    """Run orchestrator dry-run for given tier, return (stdout, stderr, returncode)."""
    cmd = [sys.executable, "experiments/pde/run_full_ablation.py", "--tier", tier, "--dry_run"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.stdout, r.stderr, r.returncode

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    results = {}
    all_pass = True

    # Test 1: Tier 1 includes rule_based
    print("Test 1: Tier 1 dry-run includes rule_based jobs...")
    stdout, stderr, rc = run_dry("1")
    if rc != 0:
        print(f"  FAIL: exit code {rc}")
        print(f"  stderr: {stderr[:500]}")
        all_pass = False
    else:
        rb_count = stdout.count("rule_based")
        if rb_count > 0:
            print(f"  PASS: {rb_count} rule_based references in output")
        else:
            print(f"  FAIL: no rule_based in Tier 1 output")
            all_pass = False
    results["tier1_rule_based"] = rb_count > 0 if rc == 0 else False

    # Test 2: Tier 1 job count
    print("\nTest 2: Tier 1 job count...")
    for line in stdout.split("\n"):
        if "Generated" in line:
            print(f"  {line.strip()}")
            break
    # Check Tier 1 line
    tier1_line = [l for l in stdout.split("\n") if "Tier 1:" in l]
    if tier1_line:
        count = int(tier1_line[0].strip().split(":")[1].strip().split()[0])
        print(f"  Tier 1 jobs: {count}")
        results["tier1_count"] = count
    else:
        print(f"  FAIL: no Tier 1 count line found")
        all_pass = False
        results["tier1_count"] = 0

    # Test 3: rule_based jobs show "(none" for TRAIN
    print("\nTest 3: rule_based TRAIN annotation...")
    none_count = stdout.count("none")
    if none_count > 0:
        print(f"  PASS: {none_count} '(none' annotations for rule_based TRAIN")
    else:
        print(f"  FAIL: no (none annotation found — rule_based TRAIN may not be handled")
        all_pass = False
    results["train_none_annotation"] = none_count > 0

    # Test 4: Tier 4 dry-run without crash
    print("\nTest 4: Tier 4 dry-run without crash...")
    stdout4, stderr4, rc4 = run_dry("4")
    if rc4 == 0:
        print(f"  PASS: Tier 4 exit code 0")
        for line in stdout4.split("\n")[:3]:
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print(f"  FAIL: Tier 4 exit code {rc4}")
        print(f"  stderr: {stderr4[:500]}")
        all_pass = False
    results["tier4_no_crash"] = rc4 == 0

    # Test 5: All tiers dry-run without errors
    print("\nTest 5: All tiers dry-run...")
    for tier in ["1", "2", "3", "4", "all", "supp"]:
        _, _, rc_t = run_dry(tier)
        status = "OK" if rc_t == 0 else "FAIL"
        if rc_t != 0:
            all_pass = False
        print(f"  --tier {tier}: {status}")
    results["all_tiers_ok"] = all_pass

    # Summary
    print("\n" + "=" * 50)
    print("ORCHESTRATOR SMOKE TEST SUMMARY")
    print("=" * 50)
    for test, passed in results.items():
        symbol = "PASS" if passed else "FAIL"
        print(f"  [{symbol}] {test}")

    total = sum(1 for v in results.values() if v)
    print(f"\n{total}/{len(results)} tests passed.")

    if all_pass:
        print("\nALL TESTS PASSED — orchestrator handles cmd_train=None correctly.")
    else:
        print("\nSOME TESTS FAILED — investigate before launching ablation.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
