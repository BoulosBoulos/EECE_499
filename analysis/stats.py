"""Statistical tests (Phase 1E component 4).

Welch's t-test, Holm-Bonferroni correction, Cohen's d, and bootstrap CIs.
Holm correction is applied within each (family × metric × tier) group.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from analysis.config import (
    ALPHA, BOOTSTRAP_CI, BOOTSTRAP_N, PDE_METHODS, RNG_SEED_BOOTSTRAP,
)


# ─────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────
def welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Welch's t-test. Returns (t_stat, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = sp_stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled std (ddof=1)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def effect_size_label(d: float) -> str:
    if not np.isfinite(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def holm_correction(p_values: list[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down correction.

    Returns (adjusted_p, significant_mask), aligned with the input order.
    NaN p-values are skipped (their adjusted p stays NaN; significant=False).
    """
    p = np.array(p_values, dtype=float)
    n = len(p)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return p.tolist(), [False] * n
    # Sort valid p-values ascending; Holm step-down.
    order = valid_idx[np.argsort(p[valid_idx])]
    m = len(order)
    adj = np.full(n, np.nan, dtype=float)
    cur_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        cand = float(p[idx]) * factor
        cur_max = max(cur_max, cand)
        adj[idx] = min(cur_max, 1.0)
    sig = [bool(np.isfinite(adj[i]) and adj[i] < alpha) for i in range(n)]
    return adj.tolist(), sig


def bootstrap_ci(
    x: np.ndarray, n_boot: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI, seed: int = RNG_SEED_BOOTSTRAP,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the mean. Deterministic via RNG seed."""
    x = np.asarray(x, dtype=float)
    if len(x) < 1:
        return (float("nan"), float("nan"))
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(samples, (1 - ci) / 2 * 100))
    hi = float(np.percentile(samples, (1 + ci) / 2 * 100))
    return (lo, hi)


# ─────────────────────────────────────────────────────────────────────────
# Family construction
# ─────────────────────────────────────────────────────────────────────────
def _family_pairs(family: str, methods_present: list[str]) -> list[tuple[str, str]]:
    """Return the (test_method, baseline_method) pairs for a family."""
    if family == "A":
        # PDE vs DRPPO baseline.
        if "drppo" not in methods_present:
            return []
        pdes = [m for m in PDE_METHODS if m in methods_present]
        return [(m, "drppo") for m in pdes]
    if family == "B":
        # Pairwise PDE.
        pdes = [m for m in PDE_METHODS if m in methods_present]
        out = []
        for i, a in enumerate(pdes):
            for b in pdes[i + 1:]:
                out.append((a, b))
        return out
    raise ValueError(f"unknown family {family!r}")


def compute_statistical_tests(
    wide_df: pd.DataFrame,
    metrics: Iterable[str],
    family: str = "A",
    cells: Optional[list[tuple[str, str, bool]]] = None,
    alpha: float = ALPHA,
    bootstrap_n: int = BOOTSTRAP_N,
) -> pd.DataFrame:
    """Compute statistical tests across (cell × metric) for a family.

    A "cell" is (scenario, ego_maneuver, intent_on). If `cells` is None it
    is auto-discovered from `wide_df`.
    """
    if wide_df is None or len(wide_df) == 0:
        return pd.DataFrame()
    metrics = list(metrics)

    if cells is None:
        cells = (
            wide_df[["scenario", "ego_maneuver", "intent_on"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        cells = list(cells)

    rows: list[dict] = []
    for metric in metrics:
        # Collect raw rows; Holm correction applied per-metric across all cells.
        cell_rows: list[dict] = []
        if metric not in wide_df.columns:
            continue
        for (scen, man, intent) in cells:
            sub = wide_df[
                (wide_df["scenario"] == scen)
                & (wide_df["ego_maneuver"] == man)
                & (wide_df["intent_on"] == intent)
            ]
            methods_present = sorted(sub["method"].dropna().unique().tolist())
            for test_m, base_m in _family_pairs(family, methods_present):
                a = sub.loc[sub["method"] == test_m, metric].dropna().to_numpy(dtype=float)
                b = sub.loc[sub["method"] == base_m, metric].dropna().to_numpy(dtype=float)
                if len(a) < 3 or len(b) < 3:
                    cell_rows.append({
                        "metric": metric, "scenario": scen,
                        "ego_maneuver": man, "intent_on": intent,
                        "method_test": test_m, "method_baseline": base_m,
                        "n_test": int(len(a)), "n_baseline": int(len(b)),
                        "mean_test": float(a.mean()) if len(a) else float("nan"),
                        "mean_baseline": float(b.mean()) if len(b) else float("nan"),
                        "std_test": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
                        "std_baseline": float(b.std(ddof=1)) if len(b) > 1 else float("nan"),
                        "t_stat": float("nan"), "p_raw": float("nan"),
                        "cohens_d": float("nan"),
                        "mean_test_ci_low": float("nan"), "mean_test_ci_high": float("nan"),
                        "mean_baseline_ci_low": float("nan"), "mean_baseline_ci_high": float("nan"),
                        "insufficient_n": True,
                    })
                    continue
                t_stat, p_raw = welch_ttest(a, b)
                d = cohens_d(a, b)
                a_lo, a_hi = bootstrap_ci(a, n_boot=bootstrap_n)
                b_lo, b_hi = bootstrap_ci(b, n_boot=bootstrap_n,
                                          seed=RNG_SEED_BOOTSTRAP + 1)
                cell_rows.append({
                    "metric": metric, "scenario": scen,
                    "ego_maneuver": man, "intent_on": intent,
                    "method_test": test_m, "method_baseline": base_m,
                    "n_test": int(len(a)), "n_baseline": int(len(b)),
                    "mean_test": float(a.mean()),
                    "mean_baseline": float(b.mean()),
                    "std_test": float(a.std(ddof=1)),
                    "std_baseline": float(b.std(ddof=1)),
                    "t_stat": t_stat, "p_raw": p_raw,
                    "cohens_d": d,
                    "mean_test_ci_low": a_lo, "mean_test_ci_high": a_hi,
                    "mean_baseline_ci_low": b_lo, "mean_baseline_ci_high": b_hi,
                    "insufficient_n": False,
                })
        if not cell_rows:
            continue
        # Holm correction per metric.
        p_raws = [r["p_raw"] for r in cell_rows]
        adj, sig = holm_correction(p_raws, alpha=alpha)
        for r, p_h, s in zip(cell_rows, adj, sig):
            r["p_holm"] = p_h
            r["significant_holm"] = bool(s) and not r.get("insufficient_n", False)
            r["effect_size_label"] = effect_size_label(r["cohens_d"])
            r["family"] = family
        rows.extend(cell_rows)

    return pd.DataFrame(rows)
