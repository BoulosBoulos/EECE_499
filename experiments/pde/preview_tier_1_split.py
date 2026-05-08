#!/usr/bin/env python3
"""Preview the Tier 1 manifest split across N machines.

Usage:
    python3 experiments/pde/preview_tier_1_split.py --n_machines 4

Prints, for each machine, the exact `--job_index_start` and `--job_index_end`
values to pass to `run_full_ablation.py` plus the first and last job in the
slice (for sanity-check).

CRITICAL: this script imports `manifest_sort_key` from `run_full_ablation`
so the sort key matches the orchestrator's exactly. If the keys diverge,
slice boundaries on different machines correspond to different jobs and
the multi-machine launch breaks (gaps + overlaps).

Field-name note: the actual job dicts (created in `generate_jobs`) use
`ego_maneuver` and `intent_on` (not `maneuver` / `intent`); the spec
example uses different names, but `manifest_sort_key` adapts to the
canonical fields used by the orchestrator.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

# Allow `python3 experiments/pde/preview_tier_1_split.py` from repo root,
# `cd experiments/pde && python3 preview_tier_1_split.py`, and PYTHONPATH-
# based invocation. We resolve the repo root from this file's location.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.pde.run_full_ablation import (  # noqa: E402
    _apply_filters,
    _partition_balance_jobs,
    generate_jobs,
    manifest_sort_key,
)


def _build_filter_args(*, include_rule_based: bool) -> SimpleNamespace:
    """Build the args namespace `_apply_filters` expects when no per-key
    filters are active. `_apply_filters` reads attributes off `args`
    (`seeds`, `methods`, `scenarios`, `maneuvers`, `intents`,
    `include_rule_based`); leaving them at None / True (defaults) preserves
    the full manifest.
    """
    return SimpleNamespace(
        seeds=None,
        methods=None,
        scenarios=None,
        maneuvers=None,
        intents=None,
        include_rule_based=include_rule_based,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preview the Tier 1 manifest split across N machines.",
    )
    ap.add_argument("--n_machines", type=int, default=4)
    ap.add_argument("--tier", default="1")
    # Mirror run_full_ablation.py default — production Tier 1 includes
    # rule_based eval-only baselines (1,680 = 1,440 trainable + 240 RB).
    ap.add_argument(
        "--include_rule_based",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include rule_based eval-only baseline (default: True). "
             "Pass --no-include_rule_based to preview the trainable-only split.",
    )
    ap.add_argument(
        "--total_steps", type=int, default=400000,
        help="Calibrated value from config_frozen_v1.yaml::calibration."
             "total_steps_calibrated. Affects job-dict cmd_train args, not "
             "the slice boundaries (which are sort-key based).",
    )
    args = ap.parse_args()

    if args.n_machines < 1:
        ap.error("--n_machines must be >= 1")

    jobs = generate_jobs(
        tier=args.tier,
        total_steps=args.total_steps,
    )
    jobs = _apply_filters(jobs, _build_filter_args(
        include_rule_based=args.include_rule_based,
    ))
    # CRITICAL: same balanced ordering as the orchestrator uses (both call
    # `_partition_balance_jobs` from run_full_ablation). The function
    # interleaves rule_based jobs at deterministic stride positions among
    # the trainable jobs so any contiguous slice gets a proportional
    # rule_based share. Editing the balance algorithm in one script
    # without the other would break slice correspondence across machines.
    jobs = _partition_balance_jobs(jobs)

    total = len(jobs)
    if total == 0:
        ap.error("Empty manifest after filtering — check tier / filter args.")
    per_machine = total // args.n_machines
    remainder = total % args.n_machines

    n_rb_total = sum(1 for j in jobs if j.get("method") == "rule_based")
    n_train_total = total - n_rb_total
    print(f"Total jobs: {total} ({n_train_total} trainable + {n_rb_total} rule_based)")
    print(
        f"Per machine (base): {per_machine}, remainder distributed to "
        f"first {remainder} machine(s)"
    )
    print()

    start = 0
    for m in range(args.n_machines):
        n = per_machine + (1 if m < remainder else 0)
        end = start + n
        slice_jobs = jobs[start:end]
        n_rb = sum(1 for j in slice_jobs if j.get("method") == "rule_based")
        n_train = n - n_rb
        # Within-slice deterministic ordering for the first/last display so
        # the preview matches what the orchestrator launches per-rental.
        slice_sorted = sorted(slice_jobs, key=manifest_sort_key)
        first = slice_sorted[0]
        last = slice_sorted[-1]
        print(
            f"Machine {m + 1}: --job_index_start {start} "
            f"--job_index_end {end} ({n} jobs: {n_train} trainable + "
            f"{n_rb} rule_based)"
        )
        print(
            f"  first: method={first['method']} | scen={first['scenario']} "
            f"| man={first['ego_maneuver']} | seed={first['seed']} "
            f"| intent_on={first.get('intent_on', False)} "
            f"| tier={first.get('tier')}"
        )
        print(
            f"  last:  method={last['method']} | scen={last['scenario']} "
            f"| man={last['ego_maneuver']} | seed={last['seed']} "
            f"| intent_on={last.get('intent_on', False)} "
            f"| tier={last.get('tier')}"
        )
        start = end


if __name__ == "__main__":
    main()
