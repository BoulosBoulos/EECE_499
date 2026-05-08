"""Data quality validation (Phase 1E component 2).

Eight per-run checks. Each run gets a structured report; the loader
aggregates these into a phase-level report and persists it to
results/analysis/meta/data_quality_report.json.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.config import (
    EXPECTED_CONFIG_KEYS,
    EXPECTED_META_KEYS,
    EXPECTED_METRICS_COLUMNS,
)


CHECK_NAMES = (
    "schema",
    "completeness",
    "nan_inf",
    "convergence",
    "action_dist",
    "terminal_counts",
    "residual_sanity",
    "method_specific",
)


def _check_schema(meta: dict, metrics_df: pd.DataFrame) -> tuple[bool, str | None]:
    missing_metrics = [c for c in EXPECTED_METRICS_COLUMNS if c not in metrics_df.columns]
    missing_meta = [k for k in EXPECTED_META_KEYS if k not in meta]
    cfg = meta.get("config") or {}
    missing_cfg = [k for k in EXPECTED_CONFIG_KEYS if k not in cfg]
    issues = []
    if missing_metrics:
        issues.append(f"metrics.csv missing {len(missing_metrics)} cols: {missing_metrics[:3]}")
    if missing_meta:
        issues.append(f"meta.json missing {len(missing_meta)} keys: {missing_meta[:3]}")
    if missing_cfg:
        issues.append(f"meta.config missing {len(missing_cfg)} keys: {missing_cfg[:3]}")
    if issues:
        return False, "; ".join(issues)
    return True, None


def _check_completeness(meta: dict, metrics_df: pd.DataFrame) -> tuple[bool, str | None]:
    if len(metrics_df) < 1:
        return False, "metrics.csv has zero rows"
    rs = meta.get("result_summary")
    if rs is None or not isinstance(rs, dict):
        return False, "meta.result_summary is null/missing"
    return True, None


def _check_nan_inf(meta: dict, metrics_df: pd.DataFrame) -> tuple[bool, str | None]:
    bad_cells = []
    for c in metrics_df.columns:
        s = metrics_df[c]
        if s.dtype.kind in "fc":
            mask = ~np.isfinite(s.fillna(np.inf))
            if mask.any():
                bad_cells.append(c)
    rs = meta.get("result_summary") or {}
    bad_keys = []
    for k, v in rs.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if not math.isfinite(float(v)):
                bad_keys.append(k)
    if bad_cells or bad_keys:
        bits = []
        if bad_cells:
            bits.append(f"metrics NaN/Inf cols: {bad_cells[:3]}")
        if bad_keys:
            bits.append(f"result_summary NaN/Inf keys: {bad_keys[:3]}")
        return False, "; ".join(bits)
    return True, None


def _check_convergence(meta: dict, metrics_df: pd.DataFrame) -> tuple[bool, str | None]:
    target = meta.get("total_steps_target")
    actual = meta.get("total_steps_actual")
    try:
        target = int(target) if target is not None else 0
        actual = int(actual) if actual is not None else 0
    except (TypeError, ValueError):
        return False, "total_steps_{target,actual} unparseable"
    if target == 0:
        return False, "total_steps_target = 0"
    ratio = actual / target
    if ratio < 0.80:
        return False, f"total_steps_actual ({actual}) < 80% of target ({target}); ratio={ratio:.2f}"
    return True, None


def _check_action_dist(meta: dict, metrics_df: pd.DataFrame) -> tuple[bool, str | None]:
    cols = ["action_dist_stop", "action_dist_creep", "action_dist_yield",
            "action_dist_go", "action_dist_abort"]
    if not all(c in metrics_df.columns for c in cols):
        return False, "missing action_dist columns"
    sums = metrics_df[cols].sum(axis=1)
    # Allow either ~1.0 or all zeros (the "no actions taken" case).
    bad = ((sums - 1.0).abs() > 0.01) & (sums.abs() > 1e-9)
    if bad.any():
        bad_idx = bad[bad].index.tolist()[:3]
        return False, f"action_dist sum != 1.0 on rows {bad_idx}"
    return True, None


def _check_terminal_counts(meta: dict, metrics_df: pd.DataFrame) -> tuple[bool, str | None]:
    needed = ["n_episodes", "n_collisions", "n_successes", "n_timeouts", "n_aborts"]
    if not all(c in metrics_df.columns for c in needed):
        return False, "missing terminal-count columns"
    eps = metrics_df["n_episodes"].astype(int)
    s = (
        metrics_df["n_collisions"].astype(int)
        + metrics_df["n_successes"].astype(int)
        + metrics_df["n_timeouts"].astype(int)
        + metrics_df["n_aborts"].astype(int)
    )
    bad = (s != eps) & (eps > 0)
    if bad.any():
        return False, f"terminal counts != n_episodes on {int(bad.sum())} rows"
    return True, None


def _check_residual_sanity(meta: dict, metrics_df: pd.DataFrame) -> tuple[bool, str | None]:
    for c in ("L_residual_optimality", "L_residual_safety"):
        if c not in metrics_df.columns:
            continue
        if (metrics_df[c] < 0).any():
            return False, f"{c} has negative values"
    return True, None


def _check_method_specific(meta: dict, metrics_df: pd.DataFrame) -> tuple[bool, str | None]:
    method = meta.get("method")
    opt = metrics_df.get("L_residual_optimality")
    saf = metrics_df.get("L_residual_safety")
    if opt is None or saf is None:
        return True, None
    opt_max = float(opt.abs().max()) if len(opt) else 0.0
    saf_max = float(saf.abs().max()) if len(saf) else 0.0

    if method == "drppo":
        if opt_max > 1e-12 or saf_max > 1e-12:
            return False, f"drppo expects both residuals 0; opt_max={opt_max}, saf_max={saf_max}"
    elif method in ("hjb_aux", "soft_hjb_aux"):
        if saf_max > 1e-12:
            return False, f"{method} expects safety residual 0; saf_max={saf_max}"
    elif method in ("eikonal_aux", "cbf_aux"):
        if opt_max > 1e-12:
            return False, f"{method} expects optimality residual 0; opt_max={opt_max}"
    elif method == "fusion_aux":
        # Fusion: at least one row should have both residuals > 0.
        if opt_max <= 0 or saf_max <= 0:
            return False, (
                f"fusion expects both residuals > 0 in some row; "
                f"opt_max={opt_max}, saf_max={saf_max}"
            )
    return True, None


_CHECKS = {
    "schema": _check_schema,
    "completeness": _check_completeness,
    "nan_inf": _check_nan_inf,
    "convergence": _check_convergence,
    "action_dist": _check_action_dist,
    "terminal_counts": _check_terminal_counts,
    "residual_sanity": _check_residual_sanity,
    "method_specific": _check_method_specific,
}


def check_run(
    *, meta: dict, metrics_df: pd.DataFrame,
    run_id: str, tier: str, subgrid: str,
) -> dict:
    """Run all 8 checks on a single (meta, metrics_df) pair. Returns a dict
    with `valid`, per-check pass/fail flags, and human-readable issues."""
    report: dict = {
        "run_id": run_id,
        "tier": tier,
        "subgrid": subgrid,
        "method": meta.get("method"),
        "valid": True,
        "checks": {},
        "issues": [],
    }
    for name, fn in _CHECKS.items():
        try:
            ok, msg = fn(meta, metrics_df)
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"
        report["checks"][name] = ok
        if not ok:
            report["valid"] = False
            report["issues"].append(f"{name}: {msg}")
    return report


def aggregate_report(per_run: dict[str, dict]) -> dict:
    """Aggregate per-run reports into a phase-level summary."""
    n_total = len(per_run)
    n_valid = sum(1 for r in per_run.values() if r.get("valid"))
    failures_by_check: dict[str, list[str]] = {n: [] for n in CHECK_NAMES}
    summary_by_tier: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_total": 0, "n_valid": 0, "n_failed": 0},
    )
    summary_by_subgrid: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_total": 0, "n_valid": 0, "n_failed": 0},
    )
    warnings = []
    for run_id, r in per_run.items():
        for check, passed in r.get("checks", {}).items():
            if not passed:
                failures_by_check.setdefault(check, []).append(run_id)
        sg = r.get("subgrid", "unknown")
        tier = r.get("tier", "unknown")
        summary_by_tier[tier]["n_total"] += 1
        summary_by_subgrid[sg]["n_total"] += 1
        if r.get("valid"):
            summary_by_tier[tier]["n_valid"] += 1
            summary_by_subgrid[sg]["n_valid"] += 1
        else:
            summary_by_tier[tier]["n_failed"] += 1
            summary_by_subgrid[sg]["n_failed"] += 1
            for issue in r.get("issues", []):
                warnings.append({"run_id": run_id, "warning": issue})
    return {
        "n_runs_discovered": n_total,
        "n_runs_valid": n_valid,
        "n_runs_failed": n_total - n_valid,
        "failure_rate": (n_total - n_valid) / max(1, n_total),
        "failures_by_check": failures_by_check,
        "warnings": warnings,
        "summary_by_tier": dict(summary_by_tier),
        "summary_by_subgrid": dict(summary_by_subgrid),
        "per_run": per_run,
    }


def save_quality_report(report: dict, output_root: str | os.PathLike) -> str:
    """Persist the aggregate report to {output_root}/meta/data_quality_report.json."""
    out_dir = Path(output_root) / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_quality_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return str(out_path)
