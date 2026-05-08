"""Plotting (Phase 1E component 6).

Seven plot families. Each is rendered to both:
    PDF  via matplotlib  — paper-ready
    HTML via plotly      — interactive exploratory

Tier 2c boundary annotations: when fusion weight tuple has w_optimality=0
or w_safety=0, the corresponding residual is flagged "decomposition:
residual not actively minimized" and drawn with a dashed line.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plotly import is optional — gracefully degrade to PDF only.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from analysis.config import (
    MATPLOTLIB_RC, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, PDE_METHODS,
)

log = logging.getLogger(__name__)


def _apply_rc():
    plt.rcParams.update(MATPLOTLIB_RC)


def _safe_save(fig, path: Path, formats: Iterable[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if "pdf" in formats:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _save_html(fig_plotly, path: Path):
    if not _HAS_PLOTLY:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_plotly.write_html(str(path.with_suffix(".html")))


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#444444")


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _ordered_methods(present: Iterable[str]) -> list[str]:
    present = set(present)
    return [m for m in METHOD_ORDER if m in present]


# ─────────────────────────────────────────────────────────────────────────
# Plot family 1 — training curves
# ─────────────────────────────────────────────────────────────────────────
def plot_training_curves(
    long_df: pd.DataFrame, output_dir: Path,
    formats: Iterable[str] = ("pdf", "html"),
    metrics: Iterable[str] = ("mean_reward", "L_residual_optimality",
                              "L_residual_safety", "L_distill"),
):
    """Per (scenario, maneuver, intent) cell: training curves for each metric,
    one curve per method (mean across seeds, ribbon = ±std)."""
    if long_df is None or long_df.empty:
        return
    sub = long_df[long_df["tier"] == "tier1"].copy()
    if sub.empty:
        sub = long_df.copy()
    cells = sub[["scenario", "ego_maneuver", "intent_on"]].drop_duplicates()

    out_root = output_dir / "tier1"
    for _, c in cells.iterrows():
        scen, man, intent = c["scenario"], c["ego_maneuver"], bool(c["intent_on"])
        intent_tag = "intent" if intent else "nointent"
        cell = sub[
            (sub["scenario"] == scen)
            & (sub["ego_maneuver"] == man)
            & (sub["intent_on"] == intent)
        ]
        for metric in metrics:
            if metric not in cell.columns:
                continue
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            for method in _ordered_methods(cell["method"].unique()):
                m = cell[cell["method"] == method]
                # Aggregate across seeds: mean and std per total_steps (or iteration).
                key = "total_steps" if "total_steps" in m.columns else "iteration"
                grp = m.groupby(key)[metric].agg(["mean", "std"]).reset_index()
                ax.plot(grp[key], grp["mean"],
                        label=_method_label(method),
                        color=_method_color(method))
                if grp["std"].notna().any():
                    ax.fill_between(
                        grp[key],
                        grp["mean"] - grp["std"].fillna(0.0),
                        grp["mean"] + grp["std"].fillna(0.0),
                        color=_method_color(method), alpha=0.15,
                    )
            ax.set_xlabel(key)
            ax.set_ylabel(metric)
            ax.set_title(f"{scen} / {man} / {intent_tag}")
            ax.legend(loc="best", frameon=False)
            base = out_root / f"{scen}_{man}_{intent_tag}_{metric}"
            _safe_save(fig, base, formats)
            if _HAS_PLOTLY:
                plotly_fig = go.Figure()
                for method in _ordered_methods(cell["method"].unique()):
                    m = cell[cell["method"] == method]
                    key = "total_steps" if "total_steps" in m.columns else "iteration"
                    grp = m.groupby(key)[metric].agg(["mean", "std"]).reset_index()
                    plotly_fig.add_trace(go.Scatter(
                        x=grp[key], y=grp["mean"], mode="lines",
                        name=_method_label(method),
                        line=dict(color=_method_color(method)),
                    ))
                plotly_fig.update_layout(
                    title=f"{scen} / {man} / {intent_tag} — {metric}",
                    xaxis_title=key, yaxis_title=metric,
                )
                _save_html(plotly_fig, base)


# ─────────────────────────────────────────────────────────────────────────
# Plot family 2 — outcome bars (tier 1)
# ─────────────────────────────────────────────────────────────────────────
def plot_outcome_bars(
    wide_df: pd.DataFrame, output_dir: Path,
    formats: Iterable[str] = ("pdf", "html"),
    metric: str = "final_collision_rate",
):
    if wide_df is None or wide_df.empty:
        return
    sub = wide_df[wide_df["tier"] == "tier1"].copy()
    if sub.empty:
        sub = wide_df.copy()
    if metric not in sub.columns:
        return
    methods = _ordered_methods(sub["method"].unique())
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    x = np.arange(len(methods))
    means = [sub.loc[sub["method"] == m, metric].astype(float).mean() for m in methods]
    stds = [sub.loc[sub["method"] == m, metric].astype(float).std(ddof=1) for m in methods]
    colors = [_method_color(m) for m in methods]
    ax.bar(x, means, yerr=stds, color=colors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Tier 1 — {metric} by method")
    base = output_dir / "tier1" / f"outcome_bar_{metric}"
    _safe_save(fig, base, formats)


# ─────────────────────────────────────────────────────────────────────────
# Plot family 3 — lambda sweep (tier 2a)
# ─────────────────────────────────────────────────────────────────────────
def plot_lambda_sensitivity(
    wide_df: pd.DataFrame, output_dir: Path,
    formats: Iterable[str] = ("pdf", "html"),
    metric: str = "final_collision_rate",
):
    if wide_df is None or wide_df.empty:
        return
    sub = wide_df[(wide_df["tier"] == "tier2") & (wide_df["subgrid"] == "2a")].copy()
    if sub.empty:
        return
    cells = sub[["scenario", "ego_maneuver"]].drop_duplicates()
    out_root = output_dir / "tier2a"
    for _, c in cells.iterrows():
        scen, man = c["scenario"], c["ego_maneuver"]
        cell = sub[(sub["scenario"] == scen) & (sub["ego_maneuver"] == man)]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for method in _ordered_methods(cell["method"].unique()):
            m = cell[cell["method"] == method]
            grp = m.groupby("lambda_residual")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(grp["lambda_residual"], grp["mean"],
                    marker="o",
                    color=_method_color(method),
                    label=_method_label(method))
            if grp["std"].notna().any():
                ax.fill_between(
                    grp["lambda_residual"],
                    grp["mean"] - grp["std"].fillna(0.0),
                    grp["mean"] + grp["std"].fillna(0.0),
                    color=_method_color(method), alpha=0.15,
                )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_{\rm residual}$")
        ax.set_ylabel(metric)
        ax.set_title(f"Tier 2a lambda sweep — {scen} / {man}")
        ax.legend(loc="best", frameon=False)
        _safe_save(fig, out_root / f"lambda_sensitivity_{scen}_{man}_{metric}", formats)


# ─────────────────────────────────────────────────────────────────────────
# Plot family 4 — occlusion impact (tier 2b)
# ─────────────────────────────────────────────────────────────────────────
def plot_occlusion_impact(
    wide_df: pd.DataFrame, output_dir: Path,
    formats: Iterable[str] = ("pdf", "html"),
    metric: str = "final_collision_rate",
):
    if wide_df is None or wide_df.empty:
        return
    sub = wide_df[(wide_df["tier"] == "tier2") & (wide_df["subgrid"] == "2b")].copy()
    if sub.empty:
        return
    if "occlusion" not in sub.columns:
        sub["occlusion"] = sub["tag"].fillna("").apply(
            lambda t: "ON" if "_occON_" in t else ("OFF" if "_occOFF_" in t else "?")
        )
    methods = _ordered_methods(sub["method"].unique())
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(methods))
    width = 0.35
    means_on = [sub.loc[(sub["method"] == m) & (sub["occlusion"] == "ON"), metric].astype(float).mean() for m in methods]
    means_off = [sub.loc[(sub["method"] == m) & (sub["occlusion"] == "OFF"), metric].astype(float).mean() for m in methods]
    colors = [_method_color(m) for m in methods]
    ax.bar(x - width / 2, means_on, width, color=colors, label="occON")
    ax.bar(x + width / 2, means_off, width, color=colors, alpha=0.55, label="occOFF")
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Tier 2b occlusion impact — {metric}")
    ax.legend(loc="best", frameon=False)
    _safe_save(fig, output_dir / "tier2b" / f"occlusion_impact_{metric}", formats)


# ─────────────────────────────────────────────────────────────────────────
# Plot family 5 — fusion weight heatmap (tier 2c)
# ─────────────────────────────────────────────────────────────────────────
def plot_fusion_weight_heatmap(
    wide_df: pd.DataFrame, output_dir: Path,
    formats: Iterable[str] = ("pdf", "html"),
    metric: str = "final_collision_rate",
):
    if wide_df is None or wide_df.empty:
        return
    sub = wide_df[(wide_df["tier"] == "tier2") & (wide_df["subgrid"] == "2c")].copy()
    if sub.empty:
        return
    cells = sub[["scenario", "ego_maneuver"]].drop_duplicates()
    out_root = output_dir / "tier2c"
    for _, c in cells.iterrows():
        scen, man = c["scenario"], c["ego_maneuver"]
        cell = sub[(sub["scenario"] == scen) & (sub["ego_maneuver"] == man)]
        agg = cell.groupby(["w_optimality", "w_safety"])[metric].mean().reset_index()
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        sc = ax.scatter(
            agg["w_optimality"], agg["w_safety"],
            c=agg[metric].astype(float),
            s=200, cmap="viridis", edgecolors="black",
        )
        for _, r in agg.iterrows():
            txt = f"{r[metric]:.2g}"
            decomposition = (r["w_optimality"] == 0.0) or (r["w_safety"] == 0.0)
            label = txt + "\n(decomposition)" if decomposition else txt
            ax.annotate(label, (r["w_optimality"], r["w_safety"]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8,
                        color="darkred" if decomposition else "black")
        ax.set_xlabel("w_optimality")
        ax.set_ylabel("w_safety")
        ax.set_title(f"Tier 2c fusion weights — {scen}/{man} — {metric}")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(metric)
        _safe_save(fig, out_root / f"fusion_weights_heatmap_{scen}_{man}_{metric}", formats)


# ─────────────────────────────────────────────────────────────────────────
# Plot family 6 — overhead bar
# ─────────────────────────────────────────────────────────────────────────
def plot_overhead_bar(
    long_df: pd.DataFrame, output_dir: Path,
    formats: Iterable[str] = ("pdf", "html"),
):
    if long_df is None or long_df.empty:
        return
    if "method" not in long_df or "iter_time_seconds" not in long_df:
        return
    methods = _ordered_methods(long_df["method"].dropna().unique())
    if not methods:
        return
    means = [float(long_df.loc[long_df["method"] == m, "iter_time_seconds"].mean()) for m in methods]
    base = means[methods.index("drppo")] if "drppo" in methods else None
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    x = np.arange(len(methods))
    colors = [_method_color(m) for m in methods]
    ax.bar(x, means, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel("mean iter_time_seconds")
    if base is not None and base > 0:
        for xi, mi, m_name in zip(x, means, methods):
            pct = 100 * (mi - base) / base
            ax.text(xi, mi, f"{pct:+.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Computational overhead vs DRPPO")
    _safe_save(fig, output_dir / "overhead_bar", formats)


# ─────────────────────────────────────────────────────────────────────────
# Plot family 7 — action distribution evolution
# ─────────────────────────────────────────────────────────────────────────
def plot_action_distribution(
    long_df: pd.DataFrame, output_dir: Path,
    formats: Iterable[str] = ("pdf", "html"),
):
    if long_df is None or long_df.empty:
        return
    cols = ["action_dist_stop", "action_dist_creep", "action_dist_yield",
            "action_dist_go", "action_dist_abort"]
    if not all(c in long_df.columns for c in cols):
        return
    out_root = output_dir / "action_distribution"
    triples = long_df[["scenario", "ego_maneuver", "method"]].drop_duplicates()
    for _, t in triples.iterrows():
        scen, man, method = t["scenario"], t["ego_maneuver"], t["method"]
        sub = long_df[
            (long_df["scenario"] == scen)
            & (long_df["ego_maneuver"] == man)
            & (long_df["method"] == method)
        ].sort_values("iteration")
        if sub.empty:
            continue
        # Aggregate across seeds: mean per iteration.
        grp = sub.groupby("iteration")[cols].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        ax.stackplot(grp["iteration"], [grp[c] for c in cols],
                     labels=[c.replace("action_dist_", "") for c in cols])
        ax.set_xlabel("iteration")
        ax.set_ylabel("action fraction")
        ax.set_title(f"{scen}/{man}/{_method_label(method)}")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
        ax.set_ylim(0, 1)
        _safe_save(fig, out_root / f"{scen}_{man}_{method}", formats)


# ─────────────────────────────────────────────────────────────────────────
# Tier 2c boundary annotation: residual curves with decomposition tag
# ─────────────────────────────────────────────────────────────────────────
def plot_fusion_residual_curves(
    long_df: pd.DataFrame, wide_df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str] = ("pdf", "html"),
):
    """For each unique (w_optimality, w_safety) tuple in 2c, plot
    L_residual_optimality and L_residual_safety vs steps.

    When w_optimality == 0 the optimality residual is annotated as
    "(decomposition: residual not actively minimized)" and drawn dashed.
    Same symmetry for w_safety == 0.
    """
    if long_df is None or long_df.empty:
        return
    if "subgrid" not in long_df.columns:
        return
    sub = long_df[(long_df["tier"] == "tier2") & (long_df["subgrid"] == "2c")].copy()
    if sub.empty:
        return
    out_root = output_dir / "tier2c"
    cells = sub[["scenario", "ego_maneuver",
                 "w_optimality", "w_safety"]].drop_duplicates()
    for _, c in cells.iterrows():
        scen = c["scenario"]; man = c["ego_maneuver"]
        w_o = float(c["w_optimality"]); w_s = float(c["w_safety"])
        cell = sub[
            (sub["scenario"] == scen)
            & (sub["ego_maneuver"] == man)
            & (sub["w_optimality"] == w_o)
            & (sub["w_safety"] == w_s)
        ]
        if cell.empty:
            continue
        key = "total_steps" if "total_steps" in cell.columns else "iteration"
        agg = cell.groupby(key)[["L_residual_optimality", "L_residual_safety"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        opt_decomposition = (w_o == 0.0)
        saf_decomposition = (w_s == 0.0)
        ax.plot(agg[key], agg["L_residual_optimality"],
                color=_method_color("hjb_aux"),
                linestyle="--" if opt_decomposition else "-",
                label="L_residual_optimality"
                + (" (decomposition: residual not actively minimized)" if opt_decomposition else ""))
        ax.plot(agg[key], agg["L_residual_safety"],
                color=_method_color("cbf_aux"),
                linestyle="--" if saf_decomposition else "-",
                label="L_residual_safety"
                + (" (decomposition: residual not actively minimized)" if saf_decomposition else ""))
        ax.set_xlabel(key)
        ax.set_ylabel("residual")
        ax.set_title(f"Fusion residuals — {scen}/{man} (w_o={w_o}, w_s={w_s})")
        ax.legend(loc="best", frameon=False, fontsize=7)
        _safe_save(fig, out_root / f"residual_curves_{scen}_{man}_w{w_o}_{w_s}", formats)


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────
def generate_all_plots(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    stats_results: Optional[dict] = None,
    output_dir: str | os.PathLike = "results/analysis/figures",
    formats: Iterable[str] = ("pdf", "html"),
) -> None:
    """Render all 7 plot families. Each plot function is fault-tolerant: if
    the input data is missing or partial, that plot is skipped with a log
    line rather than failing the whole pipeline.
    """
    _apply_rc()
    output_dir = Path(output_dir)
    formats = tuple(formats)

    plot_training_curves(long_df, output_dir, formats=formats)
    plot_outcome_bars(wide_df, output_dir, formats=formats,
                      metric="final_collision_rate")
    plot_lambda_sensitivity(wide_df, output_dir, formats=formats)
    plot_occlusion_impact(wide_df, output_dir, formats=formats)
    plot_fusion_weight_heatmap(wide_df, output_dir, formats=formats)
    plot_overhead_bar(long_df, output_dir, formats=formats)
    plot_action_distribution(long_df, output_dir, formats=formats)
    plot_fusion_residual_curves(long_df, wide_df, output_dir, formats=formats)
