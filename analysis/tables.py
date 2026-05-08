"""Table generation (Phase 1E component 5).

Seven tables, each emitted to {output_dir}/{name}.csv and (if `tex` in
formats) {output_dir}/{name}.tex. The CSV is the source of truth; the
TeX is generated for direct paper inclusion using booktabs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from analysis.config import METHOD_LABELS, METHOD_ORDER, PDE_METHODS


def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in a free-text string."""
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _df_to_latex_booktabs(
    df: pd.DataFrame, *, caption: str, label: str,
    escape_cells: bool = True,
) -> str:
    """Render `df` as a booktabs LaTeX table (compatible across pandas versions)."""
    n_cols = len(df.columns)
    col_align = "l" + "r" * (n_cols - 1) if n_cols > 0 else "l"

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + _latex_escape(caption) + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + col_align + "}")
    lines.append(r"\toprule")
    headers = " & ".join(_latex_escape(h) for h in df.columns)
    lines.append(headers + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, float) and not np.isfinite(v):
                cells.append("--")
            elif isinstance(v, float):
                cells.append(f"{v:.3g}")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(_latex_escape(v) if escape_cells else str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _save(
    df: pd.DataFrame, name: str, output_dir: Path, formats: Iterable[str],
    *, caption: str, label: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if "csv" in formats:
        df.to_csv(output_dir / f"{name}.csv", index=False)
    if "tex" in formats:
        tex = _df_to_latex_booktabs(df, caption=caption, label=label)
        (output_dir / f"{name}.tex").write_text(tex)


# ─────────────────────────────────────────────────────────────────────────
# Individual table builders
# ─────────────────────────────────────────────────────────────────────────
def _fmt_mean_std(mean: float, std: float, fmt: str = ".3g") -> str:
    if mean is None or not np.isfinite(mean):
        return "--"
    if std is None or not np.isfinite(std):
        return format(mean, fmt)
    return f"{mean:{fmt}}±{std:{fmt}}"


def table_tier1_main_comparison(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Mean±std per (scenario, maneuver, intent, method) of the primary metrics."""
    metrics = (
        "final_collision_rate", "final_success_rate",
        "final_mean_reward", "min_ttc_eval", "mean_ttc_eval",
    )
    if wide_df.empty:
        return pd.DataFrame(columns=("scenario", "maneuver", "intent", "method", "n", *metrics))
    sub = wide_df[wide_df["tier"] == "tier1"].copy()
    if sub.empty:
        return pd.DataFrame(columns=("scenario", "maneuver", "intent", "method", "n", *metrics))
    rows = []
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    grp_cols = ["scenario", "ego_maneuver", "intent_on", "method"]
    for keys, g in sub.groupby(grp_cols):
        scen, man, intent, method = keys
        row = {
            "scenario": scen, "maneuver": man, "intent": bool(intent),
            "method": METHOD_LABELS.get(method, method),
            "n": int(len(g)),
            "_method_key": method,
        }
        for m in metrics:
            if m in g.columns:
                vals = g[m].astype(float)
                row[m] = _fmt_mean_std(vals.mean(), vals.std(ddof=1) if len(vals) > 1 else float("nan"))
            else:
                row[m] = "--"
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_rank"] = out["_method_key"].map(method_rank).fillna(99)
        out = out.sort_values(
            ["scenario", "maneuver", "intent", "_rank"]
        ).drop(columns=["_rank", "_method_key"])
    return out


def table_tier1_statistical_summary(stats_a: pd.DataFrame) -> pd.DataFrame:
    """Per (metric × method), n significant cells / mean Cohen's d / win-rate."""
    if stats_a is None or stats_a.empty:
        return pd.DataFrame(columns=(
            "metric", "method", "n_cells_significant",
            "mean_cohens_d", "win_rate_pct",
        ))
    rows = []
    for (metric, method), g in stats_a.groupby(["metric", "method_test"]):
        n_cells = int(len(g))
        n_sig = int(g["significant_holm"].sum()) if "significant_holm" in g else 0
        d_vals = g["cohens_d"].dropna()
        mean_d = float(d_vals.mean()) if len(d_vals) else float("nan")
        # "Win" = significant AND mean_test < mean_baseline for "lower is better"
        # metrics (collision/timeout/abort), or mean_test > mean_baseline for
        # "higher is better" metrics (success/return/ttc). We treat lower as
        # better when the metric name contains "collision", "timeout", or "abort".
        lower_better = any(
            tok in metric for tok in ("collision", "timeout", "abort")
        )
        if "mean_test" in g and "mean_baseline" in g:
            mt = g["mean_test"]
            mb = g["mean_baseline"]
            if lower_better:
                wins = (g["significant_holm"] & (mt < mb)).sum()
            else:
                wins = (g["significant_holm"] & (mt > mb)).sum()
            win_pct = 100.0 * float(wins) / float(n_cells)
        else:
            win_pct = float("nan")
        rows.append({
            "metric": metric,
            "method": METHOD_LABELS.get(method, method),
            "n_cells_significant": f"{n_sig}/{n_cells}",
            "mean_cohens_d": f"{mean_d:.3f}" if np.isfinite(mean_d) else "--",
            "win_rate_pct": f"{win_pct:.0f}%" if np.isfinite(win_pct) else "--",
        })
    return pd.DataFrame(rows)


def table_tier2a_lambda_sensitivity(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame(columns=("method", "lambda_residual",
                                     "n", "final_collision_rate", "final_mean_reward"))
    sub = wide_df[wide_df["tier"] == "tier2"].copy()
    sub = sub[sub["subgrid"] == "2a"]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (method, lam), g in sub.groupby(["method", "lambda_residual"]):
        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "lambda_residual": lam,
            "n": int(len(g)),
            "final_collision_rate": _fmt_mean_std(
                g["final_collision_rate"].mean(),
                g["final_collision_rate"].std(ddof=1) if len(g) > 1 else float("nan"),
            ),
            "final_mean_reward": _fmt_mean_std(
                g["final_mean_reward"].mean(),
                g["final_mean_reward"].std(ddof=1) if len(g) > 1 else float("nan"),
            ),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["method", "lambda_residual"])
    return out


def table_tier2b_occlusion_impact(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame(columns=(
            "method", "occlusion", "n",
            "final_collision_rate", "final_mean_reward",
        ))
    sub = wide_df[(wide_df["tier"] == "tier2") & (wide_df["subgrid"] == "2b")].copy()
    if sub.empty:
        return pd.DataFrame()
    # Tag-derived occlusion column.
    if "occlusion" not in sub.columns:
        sub["occlusion"] = sub["tag"].fillna("").apply(
            lambda t: "ON" if "_occON_" in t else ("OFF" if "_occOFF_" in t else None)
        )
    rows = []
    for (method, occ), g in sub.groupby(["method", "occlusion"]):
        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "occlusion": occ,
            "n": int(len(g)),
            "final_collision_rate": _fmt_mean_std(
                g["final_collision_rate"].mean(),
                g["final_collision_rate"].std(ddof=1) if len(g) > 1 else float("nan"),
            ),
            "final_mean_reward": _fmt_mean_std(
                g["final_mean_reward"].mean(),
                g["final_mean_reward"].std(ddof=1) if len(g) > 1 else float("nan"),
            ),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["method", "occlusion"])
    return out


def table_tier2c_fusion_weights(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame(columns=(
            "w_optimality", "w_safety", "n",
            "final_collision_rate", "final_mean_reward", "decomposition",
        ))
    sub = wide_df[(wide_df["tier"] == "tier2") & (wide_df["subgrid"] == "2c")].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (w_o, w_s), g in sub.groupby(["w_optimality", "w_safety"]):
        decomposition = ""
        if w_o == 0.0 and w_s != 0.0:
            decomposition = "CBF only (Soft-HJB off)"
        elif w_s == 0.0 and w_o != 0.0:
            decomposition = "Soft-HJB only (CBF off)"
        rows.append({
            "w_optimality": float(w_o),
            "w_safety": float(w_s),
            "n": int(len(g)),
            "final_collision_rate": _fmt_mean_std(
                g["final_collision_rate"].mean(),
                g["final_collision_rate"].std(ddof=1) if len(g) > 1 else float("nan"),
            ),
            "final_mean_reward": _fmt_mean_std(
                g["final_mean_reward"].mean(),
                g["final_mean_reward"].std(ddof=1) if len(g) > 1 else float("nan"),
            ),
            "decomposition": decomposition,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["w_optimality", "w_safety"])
    return out


def table_computational_overhead(wide_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean iter wall time and residual compute time per method, with
    overhead vs DRPPO."""
    if long_df.empty:
        return pd.DataFrame(columns=(
            "method", "n_runs",
            "mean_iter_time_s", "mean_residual_time_s",
            "overhead_vs_drppo_pct",
        ))
    base_iter = None
    if "method" in long_df and "iter_time_seconds" in long_df:
        drppo = long_df[long_df["method"] == "drppo"]
        base_iter = float(drppo["iter_time_seconds"].mean()) if len(drppo) else None
    rows = []
    for method, g in long_df.groupby("method"):
        mean_it = float(g["iter_time_seconds"].mean()) if "iter_time_seconds" in g else float("nan")
        mean_rc = float(g["residual_compute_time_seconds"].mean()) if "residual_compute_time_seconds" in g else float("nan")
        if base_iter and base_iter > 0:
            overhead = 100.0 * (mean_it - base_iter) / base_iter
            overhead_str = f"{overhead:+.1f}%"
        else:
            overhead_str = "--"
        n_runs = int(wide_df[wide_df["method"] == method].shape[0]) if not wide_df.empty else 0
        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "n_runs": n_runs,
            "mean_iter_time_s": f"{mean_it:.3f}",
            "mean_residual_time_s": f"{mean_rc:.3f}",
            "overhead_vs_drppo_pct": overhead_str,
            "_method_key": method,
        })
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_rank"] = out["_method_key"].map(method_rank).fillna(99)
        out = out.sort_values("_rank").drop(columns=["_rank", "_method_key"])
    return out


def table_tier4_holdout(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()
    sub = wide_df[wide_df["tier"] == "tier4"].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (method, ho), g in sub.groupby(["method", "subgrid"]):
        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "holdout_config": ho,
            "n": int(len(g)),
            "final_collision_rate": _fmt_mean_std(
                g["final_collision_rate"].mean(),
                g["final_collision_rate"].std(ddof=1) if len(g) > 1 else float("nan"),
            ),
            "final_mean_reward": _fmt_mean_std(
                g["final_mean_reward"].mean(),
                g["final_mean_reward"].std(ddof=1) if len(g) > 1 else float("nan"),
            ),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["holdout_config", "method"])
    return out


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────
def generate_all_tables(
    wide_df: pd.DataFrame,
    long_df: Optional[pd.DataFrame] = None,
    stats_results: Optional[dict] = None,
    output_dir: str | os.PathLike = "results/analysis/tables",
    formats: Iterable[str] = ("csv", "tex"),
) -> dict[str, str]:
    """Build all 7 tables and persist them. Returns map of name → output_dir path."""
    output_dir = Path(output_dir)
    formats = tuple(formats)

    out_paths: dict[str, str] = {}

    # 1. Tier 1 main comparison
    df = table_tier1_main_comparison(wide_df)
    _save(df, "tier1_main_comparison", output_dir, formats,
          caption="Tier 1 main comparison: mean ± std per cell.",
          label="tab:tier1_main")
    out_paths["tier1_main_comparison"] = str(output_dir)

    # 2. Tier 1 statistical summary (uses stats family A)
    stats_a = stats_results.get("tier1_A") if stats_results else None
    if stats_a is None:
        stats_a = pd.DataFrame()
    df = table_tier1_statistical_summary(stats_a)
    _save(df, "tier1_statistical_summary", output_dir, formats,
          caption="Tier 1 statistical summary: PDE methods vs DRPPO (Holm-corrected).",
          label="tab:tier1_stats")
    out_paths["tier1_statistical_summary"] = str(output_dir)

    # 3. Tier 2a lambda sensitivity
    df = table_tier2a_lambda_sensitivity(wide_df)
    _save(df, "tier2a_lambda_sensitivity", output_dir, formats,
          caption="Tier 2a: lambda residual sweep.",
          label="tab:tier2a_lambda")
    out_paths["tier2a_lambda_sensitivity"] = str(output_dir)

    # 4. Tier 2b occlusion impact
    df = table_tier2b_occlusion_impact(wide_df)
    _save(df, "tier2b_occlusion_impact", output_dir, formats,
          caption="Tier 2b: occlusion impact.",
          label="tab:tier2b_occlusion")
    out_paths["tier2b_occlusion_impact"] = str(output_dir)

    # 5. Tier 2c fusion weights
    df = table_tier2c_fusion_weights(wide_df)
    _save(df, "tier2c_fusion_weights", output_dir, formats,
          caption="Tier 2c: fusion weight study.",
          label="tab:tier2c_fusion_weights")
    out_paths["tier2c_fusion_weights"] = str(output_dir)

    # 6. Computational overhead
    if long_df is None:
        long_df = pd.DataFrame()
    df = table_computational_overhead(wide_df, long_df)
    _save(df, "computational_overhead", output_dir, formats,
          caption="Computational overhead per method (vs DRPPO baseline).",
          label="tab:overhead")
    out_paths["computational_overhead"] = str(output_dir)

    # 7. Tier 4 holdout
    df = table_tier4_holdout(wide_df)
    _save(df, "tier4_holdout", output_dir, formats,
          caption="Tier 4 held-out evaluation.",
          label="tab:tier4_holdout")
    out_paths["tier4_holdout"] = str(output_dir)

    return out_paths
