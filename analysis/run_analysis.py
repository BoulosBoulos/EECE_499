"""Main analysis orchestrator (Phase 1E component 7).

Single CLI entry point that runs the full pipeline:
    discover runs → quality report → stats tests → tables → plots → metadata
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from analysis.config import (
    ALPHA, ANALYSIS_OUTPUT_DEFAULT, FAILURE_RATE_GATE,
    PRIMARY_METRICS, RESULTS_ROOT_DEFAULT,
)
from analysis.loader import load_results
from analysis.plots import generate_all_plots
from analysis.quality import save_quality_report
from analysis.stats import compute_statistical_tests
from analysis.tables import generate_all_tables
from config_loader import check_config_lock


def _save_analysis_metadata(
    output_root: Path,
    *, args: argparse.Namespace,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    quality_report: dict,
    elapsed_seconds: float,
) -> str:
    out_dir = Path(output_root) / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "args": vars(args),
        "n_runs_loaded": int(len(wide_df)),
        "n_iterations_loaded": int(len(long_df)),
        "quality_summary": {
            "n_runs_discovered": quality_report.get("n_runs_discovered"),
            "n_runs_valid": quality_report.get("n_runs_valid"),
            "n_runs_failed": quality_report.get("n_runs_failed"),
            "failure_rate": quality_report.get("failure_rate"),
        },
    }
    out_path = out_dir / "analysis_run_metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return str(out_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run analysis pipeline.")
    parser.add_argument("--results_root", default=RESULTS_ROOT_DEFAULT)
    parser.add_argument("--output_root", default=ANALYSIS_OUTPUT_DEFAULT)
    parser.add_argument("--tiers", nargs="*", default=None,
                        help="Tier names to load (e.g. tier1 tier2 2a 2b 2c). Default = all.")
    parser.add_argument("--skip_failed", action="store_true", default=True,
                        help="Exclude runs flagged by quality checks (default).")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip plot generation.")
    parser.add_argument("--no_tex", action="store_true",
                        help="Generate CSV tables only, skip LaTeX.")
    parser.add_argument("--no_prompt", action="store_true",
                        help="Don't prompt on failure-rate gate; abort silently.")
    args = parser.parse_args(argv)

    # Phase 1F: warn (don't block) if config_frozen_v1.yaml has been modified
    # post-freeze. Analysis may still be valid on data produced before the
    # change; we just surface the warning so reviewers can investigate.
    _lock_status = check_config_lock(strict=False)
    if not _lock_status["matches"] and _lock_status.get("warning"):
        print(_lock_status["warning"])

    t0 = time.time()
    print(f"[run_analysis] loading runs from {args.results_root}")
    long_df, wide_df, quality_report = load_results(
        results_root=args.results_root,
        skip_failed=args.skip_failed,
        tiers=args.tiers,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    quality_path = save_quality_report(quality_report, output_root)
    print(
        f"[run_analysis] {quality_report.get('n_runs_valid', 0)}/"
        f"{quality_report.get('n_runs_discovered', 0)} runs valid; "
        f"failure rate = {quality_report.get('failure_rate', 0.0):.1%}; "
        f"report → {quality_path}"
    )

    failure_rate = quality_report.get("failure_rate", 0.0) or 0.0
    if failure_rate > FAILURE_RATE_GATE:
        print(f"[run_analysis] WARNING: {failure_rate:.1%} of runs failed quality checks "
              f"(gate = {FAILURE_RATE_GATE:.0%}).")
        if args.no_prompt:
            print("[run_analysis] --no_prompt set; aborting.")
            return 2
        try:
            confirm = input("Continue anyway? [y/N]: ")
        except EOFError:
            confirm = ""
        if confirm.strip().lower() != "y":
            print("[run_analysis] aborted by user.")
            return 2

    # Statistical tests per tier × family.
    stats_results: dict[str, pd.DataFrame] = {}
    if not wide_df.empty:
        for tier_name in sorted(wide_df["tier"].dropna().unique()):
            tier_df = wide_df[wide_df["tier"] == tier_name]
            for family in ("A", "B"):
                key = f"{tier_name}_{family}"
                stats_results[key] = compute_statistical_tests(
                    wide_df=tier_df,
                    metrics=PRIMARY_METRICS,
                    family=family,
                    alpha=ALPHA,
                )
                # Persist per (tier × family) CSV.
                stats_dir = output_root / "statistical_tests"
                stats_dir.mkdir(parents=True, exist_ok=True)
                if not stats_results[key].empty:
                    for metric, mg in stats_results[key].groupby("metric"):
                        mg.to_csv(
                            stats_dir / f"{tier_name}_family_{family}_{metric}.csv",
                            index=False,
                        )

    # Tables.
    formats_tab: tuple[str, ...] = ("csv",) if args.no_tex else ("csv", "tex")
    generate_all_tables(
        wide_df=wide_df,
        long_df=long_df,
        stats_results=stats_results,
        output_dir=output_root / "tables",
        formats=formats_tab,
    )
    print(f"[run_analysis] tables → {output_root / 'tables'}")

    # Plots.
    if not args.no_plots:
        generate_all_plots(
            long_df=long_df,
            wide_df=wide_df,
            stats_results=stats_results,
            output_dir=output_root / "figures",
            formats=("pdf", "html"),
        )
        print(f"[run_analysis] figures → {output_root / 'figures'}")
    else:
        print("[run_analysis] --no_plots set; skipping figures.")

    elapsed = time.time() - t0
    _save_analysis_metadata(
        output_root, args=args, long_df=long_df, wide_df=wide_df,
        quality_report=quality_report, elapsed_seconds=elapsed,
    )
    print(f"[run_analysis] complete ({elapsed:.1f}s); output → {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
