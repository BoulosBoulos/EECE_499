"""Result discovery and DataFrame construction (Phase 1E component 1).

Walks `results_root` recursively, parses every (metrics.csv + meta.json)
pair into a long DataFrame (one row per (run, iteration)) and a wide
DataFrame (one row per run). Quality issues are detected per-run and
reported alongside the DataFrames. eval_metrics.csv, when present, is
folded into the wide DataFrame.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from analysis.config import (
    EXPECTED_CONFIG_KEYS,
    EXPECTED_META_KEYS,
    EXPECTED_METRICS_COLUMNS,
    FINAL_WINDOW_FRAC,
)
from analysis import metrics as metrics_mod
from analysis import quality as quality_mod

log = logging.getLogger(__name__)

KNOWN_SCENARIOS = (
    "1a", "1b", "1c", "1d", "2", "3", "4",
    "2_dense", "3_dense", "4_dense",
)
KNOWN_MANEUVERS = (
    "stem_right", "stem_left", "right_left",
    "right_stem", "left_right", "left_stem",
)
KNOWN_METHODS = (
    "drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux",
    "fusion_aux", "rule_based",
)


# ─────────────────────────────────────────────────────────────────────────
# Tag parsing
# ─────────────────────────────────────────────────────────────────────────
def parse_tag(tag: str) -> Optional[dict]:
    """Parse a Tier 1/2 tag into structured fields. Returns None on failure.

    Supports the following grammars:
        T1_{scenario}_{maneuver}_{method}_{intent_tag}_s{seed}
        T2a_{scenario}_{maneuver}_{method}_lam{lambda}_s{seed}
        T2b_{scenario}_{maneuver}_{method}_occ{ON|OFF}_s{seed}
        T2c_{scenario}_{maneuver}_fusion_aux_w{w_o}_{w_s}_s{seed}
    """
    if not tag.startswith("T"):
        return None

    # Tier prefix.
    m = re.match(r"^T(1|2a|2b|2c|3[A-Za-z]?|SUP)_", tag)
    if not m:
        return None
    tier_label = m.group(1).lower()
    body = tag[len(m.group(0)):]

    # Greedy match of scenario then maneuver from the front of the body.
    scenarios_by_len = sorted(KNOWN_SCENARIOS, key=len, reverse=True)
    maneuvers_by_len = sorted(KNOWN_MANEUVERS, key=len, reverse=True)
    methods_by_len = sorted(KNOWN_METHODS, key=len, reverse=True)

    sc = None
    rem = body
    for s in scenarios_by_len:
        if rem.startswith(s + "_"):
            sc = s
            rem = rem[len(s) + 1:]
            break
    if sc is None:
        return None

    mv = None
    for m_ in maneuvers_by_len:
        if rem.startswith(m_ + "_"):
            mv = m_
            rem = rem[len(m_) + 1:]
            break
    if mv is None:
        return None

    method = None
    for m_ in methods_by_len:
        if rem.startswith(m_):
            # Method is followed by either end-of-body or "_<extras>" markers.
            tail = rem[len(m_):]
            if tail == "" or tail.startswith("_"):
                method = m_
                rem = tail.lstrip("_")
                break
    if method is None:
        return None

    out = {
        "tier_label": tier_label,
        "scenario": sc,
        "ego_maneuver": mv,
        "method": method,
    }

    # Tier-specific extras.
    if tier_label == "1":
        # rem expected: "{intent|nointent}_s{seed}"
        m2 = re.match(r"^(intent|nointent)_s(-?\d+)$", rem)
        if not m2:
            return None
        out["intent_on"] = (m2.group(1) == "intent")
        out["seed"] = int(m2.group(2))
    elif tier_label == "2a":
        m2 = re.match(r"^lam(-?\d+(?:\.\d+)?)_s(-?\d+)$", rem)
        if not m2:
            return None
        out["lambda_residual"] = float(m2.group(1))
        out["seed"] = int(m2.group(2))
    elif tier_label == "2b":
        m2 = re.match(r"^occ(ON|OFF)_s(-?\d+)$", rem)
        if not m2:
            return None
        out["occlusion"] = (m2.group(1) == "ON")
        out["seed"] = int(m2.group(2))
    elif tier_label == "2c":
        m2 = re.match(
            r"^w(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_s(-?\d+)$", rem,
        )
        if not m2:
            return None
        out["w_optimality"] = float(m2.group(1))
        out["w_safety"] = float(m2.group(2))
        out["seed"] = int(m2.group(3))
    else:
        m2 = re.search(r"_s(-?\d+)$", rem)
        if m2:
            out["seed"] = int(m2.group(1))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Path → tier/subgrid inference
# ─────────────────────────────────────────────────────────────────────────
_SUBGRID_DIR_TO_KEY = {
    "2a_lambda_sweep": "2a",
    "2b_occlusion_sweep": "2b",
    "2c_fusion_weights": "2c",
}


def _infer_tier_and_subgrid(run_dir: Path, results_root: Path) -> tuple[str, str]:
    """Return (tier, subgrid) inferred from the directory layout.

    Layouts handled:
        results_root/tier1/<run_id>/                 → ("tier1", "1")
        results_root/tier2/<sub>/<run_id>/           → ("tier2", "2a"|"2b"|"2c")
        results_root/tier3_<flavor>/<run_id>/        → ("tier3", flavor)
        results_root/tier4_<ho>/<run_id>/            → ("tier4", ho)
        results_root/supplementary/<run_id>/         → ("supp", "supp")
    Anything else: ("unknown", "unknown").
    """
    try:
        rel = run_dir.relative_to(results_root)
    except ValueError:
        return ("unknown", "unknown")
    parts = rel.parts
    if len(parts) < 1:
        return ("unknown", "unknown")
    top = parts[0]
    if top == "tier1":
        return ("tier1", "1")
    if top == "tier2":
        if len(parts) >= 2 and parts[1] in _SUBGRID_DIR_TO_KEY:
            return ("tier2", _SUBGRID_DIR_TO_KEY[parts[1]])
        return ("tier2", "2")
    if top.startswith("tier3"):
        flavor = top[len("tier3"):].lstrip("_") or "3"
        return ("tier3", flavor or "3")
    if top.startswith("tier4"):
        flavor = top[len("tier4"):].lstrip("_") or "4"
        return ("tier4", flavor or "4")
    if top in ("supp", "supplementary"):
        return ("supp", "supp")
    return ("unknown", "unknown")


# ─────────────────────────────────────────────────────────────────────────
# Run loading
# ─────────────────────────────────────────────────────────────────────────
def _read_meta(meta_path: Path) -> Optional[dict]:
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as e:
        log.warning("failed to read %s: %s", meta_path, e)
        return None


def _read_metrics(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        log.warning("failed to read %s: %s", csv_path, e)
        return None


def _flatten_config(cfg: dict | None) -> dict:
    if not isinstance(cfg, dict):
        return {k: None for k in EXPECTED_CONFIG_KEYS}
    return {k: cfg.get(k) for k in EXPECTED_CONFIG_KEYS}


def _flatten_result_summary(rs: dict | None) -> dict:
    keys = (
        "best_eval_return", "final_eval_return", "best_iteration",
        "final_collision_rate", "final_success_rate", "final_timeout_rate",
        "best_distillation_gap",
        "final_residual_loss_optimality", "final_residual_loss_safety",
    )
    if not isinstance(rs, dict):
        return {f"meta_{k}": None for k in keys}
    return {f"meta_{k}": rs.get(k) for k in keys}


def discover_runs(results_root: str | os.PathLike) -> list[Path]:
    """Return all directories under results_root that contain both
    metrics.csv and meta.json."""
    root = Path(results_root)
    out: list[Path] = []
    if not root.exists():
        return out
    for path in sorted(root.rglob("metrics.csv")):
        run_dir = path.parent
        if (run_dir / "meta.json").is_file():
            out.append(run_dir)
    return out


def _load_eval_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    p = run_dir / "eval_metrics.csv"
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        log.warning("failed to read %s: %s", p, e)
        return None


def _build_long_rows(
    run_id: str, tier: str, subgrid: str, tag: str,
    meta: dict, metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    df = metrics_df.copy()
    df["run_id"] = run_id
    df["tier"] = tier
    df["subgrid"] = subgrid
    df["tag"] = tag
    df["method"] = meta.get("method")
    df["scenario"] = meta.get("scenario")
    df["ego_maneuver"] = meta.get("ego_maneuver")
    df["seed"] = meta.get("seed")
    df["intent_on"] = bool(meta.get("intent_on", False))
    cfg = _flatten_config(meta.get("config"))
    for k, v in cfg.items():
        df[k] = v
    return df


def _build_wide_row(
    run_id: str, tier: str, subgrid: str, tag: str,
    meta: dict, metrics_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame],
) -> dict:
    row = {
        "run_id": run_id,
        "tier": tier,
        "subgrid": subgrid,
        "tag": tag,
        "method": meta.get("method"),
        "scenario": meta.get("scenario"),
        "ego_maneuver": meta.get("ego_maneuver"),
        "seed": meta.get("seed"),
        "intent_on": bool(meta.get("intent_on", False)),
        "total_iterations": int(metrics_df["iteration"].iloc[-1]) if len(metrics_df) else 0,
        "total_steps_actual": meta.get("total_steps_actual"),
        "convergence_reason": meta.get("convergence_reason"),
        "wall_time_total_seconds": meta.get("wall_time_seconds"),
    }
    # Last-row "final_*" loss columns.
    if len(metrics_df):
        last = metrics_df.iloc[-1]
        for col in ("L_total", "L_policy", "L_value",
                    "L_residual_optimality", "L_residual_safety",
                    "L_distill", "mean_reward"):
            row[f"final_{col}"] = float(last[col]) if col in metrics_df.columns else None
    # Final-window outcome metrics.
    fm = metrics_mod.compute_final_metrics(metrics_df, FINAL_WINDOW_FRAC)
    row.update(fm)
    # Config columns.
    row.update(_flatten_config(meta.get("config")))
    # Result_summary fields (prefixed `meta_` to avoid colliding with computed
    # final_* columns).
    row.update(_flatten_result_summary(meta.get("result_summary")))
    # Eval metrics (optional).
    if eval_df is not None and len(eval_df):
        em = metrics_mod.compute_eval_metrics(eval_df)
        row.update(em)
    else:
        row.update({
            "eval_collision_rate": None, "eval_success_rate": None,
            "eval_timeout_rate": None, "eval_abort_rate": None,
            "mean_return_eval": None, "min_ttc_eval": None,
            "mean_ttc_eval": None, "n_eval_episodes": 0,
        })
    return row


def load_results(
    results_root: str | os.PathLike,
    skip_failed: bool = True,
    tiers: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Eager-load all runs under results_root.

    Returns (long_df, wide_df, quality_report).
    If skip_failed is True (default), runs flagged by quality checks are
    excluded from the returned DataFrames; the quality report still lists
    them.
    """
    results_root = Path(results_root)
    run_dirs = discover_runs(results_root)

    long_rows: list[pd.DataFrame] = []
    wide_rows: list[dict] = []
    seen_run_ids: set[str] = set()
    per_run_quality: dict[str, dict] = {}

    for run_dir in run_dirs:
        meta = _read_meta(run_dir / "meta.json")
        metrics_df = _read_metrics(run_dir / "metrics.csv")
        if meta is None or metrics_df is None:
            # Cannot continue without both — record completeness failure.
            run_id = run_dir.name
            per_run_quality[run_id] = {
                "completeness": False,
                "missing": [],
                "issues": ["meta.json or metrics.csv unreadable"],
                "tier": _infer_tier_and_subgrid(run_dir, results_root)[0],
                "subgrid": _infer_tier_and_subgrid(run_dir, results_root)[1],
            }
            continue
        run_id = str(meta.get("run_id") or run_dir.name)
        tier, subgrid = _infer_tier_and_subgrid(run_dir, results_root)
        tag = run_dir.name

        if tiers is not None and tier not in tiers and subgrid not in tiers:
            continue

        if run_id in seen_run_ids:
            log.warning("duplicate run_id %s at %s; skipping later occurrence",
                        run_id, run_dir)
            continue
        seen_run_ids.add(run_id)

        report = quality_mod.check_run(
            meta=meta, metrics_df=metrics_df, run_id=run_id,
            tier=tier, subgrid=subgrid,
        )
        per_run_quality[run_id] = report
        if skip_failed and not report["valid"]:
            continue

        eval_df = _load_eval_metrics(run_dir)
        long_rows.append(_build_long_rows(
            run_id, tier, subgrid, tag, meta, metrics_df,
        ))
        wide_rows.append(_build_wide_row(
            run_id, tier, subgrid, tag, meta, metrics_df, eval_df,
        ))

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    wide_df = pd.DataFrame(wide_rows) if wide_rows else pd.DataFrame()
    quality_report = quality_mod.aggregate_report(per_run_quality)
    return long_df, wide_df, quality_report
