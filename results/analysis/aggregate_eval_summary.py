"""Aggregate per-run eval_metrics.csv summaries across all machine buckets.

Cluster machines only pushed eval_metrics.csv (no metrics.csv / meta.json), so
analysis.run_analysis cannot pick them up. This script uses the codebase's
analysis.metrics.compute_eval_metrics function on every eval_metrics.csv it
finds, parses (method, scenario, maneuver, intent, seed) from the job dir
name, and writes a wide per-run table.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.metrics import compute_eval_metrics

KNOWN_SCENARIOS = ["1a", "1b", "1c", "1d", "2_dense", "3_dense", "4_dense", "2", "3", "4"]
KNOWN_MANEUVERS = ["stem_right", "stem_left", "right_left", "right_stem", "left_right", "left_stem"]
KNOWN_METHODS = ["drppo", "soft_hjb_aux", "hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux", "rule_based"]


def parse_job_dir(name: str) -> dict | None:
    """Parse '{scenario}_{maneuver}_{method}_{intent|nointent}_s{seed}'."""
    for sc in sorted(KNOWN_SCENARIOS, key=len, reverse=True):
        if not name.startswith(sc + "_"):
            continue
        rest = name[len(sc) + 1:]
        for mv in sorted(KNOWN_MANEUVERS, key=len, reverse=True):
            if not rest.startswith(mv + "_"):
                continue
            rest2 = rest[len(mv) + 1:]
            for me in sorted(KNOWN_METHODS, key=len, reverse=True):
                if not rest2.startswith(me + "_"):
                    continue
                tail = rest2[len(me) + 1:]
                m = re.match(r"^(intent|nointent)_s(-?\d+)$", tail)
                if m:
                    return {
                        "scenario": sc, "maneuver": mv, "method": me,
                        "intent_on": m.group(1) == "intent",
                        "seed": int(m.group(2)),
                    }
    return None


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/tier_1_combined")
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/analysis_new/stats/eval_summary_all.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    for em_path in sorted(root.rglob("eval_metrics.csv")):
        job_name = em_path.parent.name
        parsed = parse_job_dir(job_name)
        if parsed is None:
            skipped += 1
            continue
        try:
            df = pd.read_csv(em_path)
        except Exception:
            skipped += 1
            continue
        em = compute_eval_metrics(df)
        rows.append({
            "job": job_name,
            "method": parsed["method"],
            "scenario": parsed["scenario"],
            "maneuver": parsed["maneuver"],
            "intent_on": parsed["intent_on"],
            "seed": parsed["seed"],
            **em,
        })

    if not rows:
        print("No eval_metrics.csv files parsed.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} runs; skipped {skipped})")


if __name__ == "__main__":
    main()
