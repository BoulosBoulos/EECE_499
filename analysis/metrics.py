"""Outcome metric computation (Phase 1E component 3)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_final_metrics(
    run_iterations: pd.DataFrame, window_frac: float = 0.10,
) -> dict:
    """Aggregate the last `window_frac` of training iterations into outcome
    metrics. Returns a dict suitable for merging into wide_df.
    """
    if run_iterations is None or len(run_iterations) == 0:
        return {
            "final_collision_rate": None,
            "final_success_rate": None,
            "final_timeout_rate": None,
            "final_abort_rate": None,
            "final_mean_reward": None,
            "mean_action_dist_stop": None,
            "mean_action_dist_creep": None,
            "mean_action_dist_yield": None,
            "mean_action_dist_go": None,
            "mean_action_dist_abort": None,
        }
    df = run_iterations
    if "iteration" in df.columns:
        df = df.sort_values("iteration").reset_index(drop=True)
    n_total = len(df)
    n_window = max(1, int(n_total * window_frac))
    window = df.tail(n_window)
    total_eps = float(window["n_episodes"].sum()) if "n_episodes" in window else 0.0

    def rate(col):
        if col not in window.columns or total_eps == 0:
            return 0.0
        return float(window[col].sum()) / total_eps

    return {
        "final_collision_rate": rate("n_collisions"),
        "final_success_rate":   rate("n_successes"),
        "final_timeout_rate":   rate("n_timeouts"),
        "final_abort_rate":     rate("n_aborts"),
        "final_mean_reward":    float(window["mean_reward"].mean()) if "mean_reward" in window else None,
        "mean_action_dist_stop":  float(window["action_dist_stop"].mean())  if "action_dist_stop"  in window else None,
        "mean_action_dist_creep": float(window["action_dist_creep"].mean()) if "action_dist_creep" in window else None,
        "mean_action_dist_yield": float(window["action_dist_yield"].mean()) if "action_dist_yield" in window else None,
        "mean_action_dist_go":    float(window["action_dist_go"].mean())    if "action_dist_go"    in window else None,
        "mean_action_dist_abort": float(window["action_dist_abort"].mean()) if "action_dist_abort" in window else None,
    }


def compute_eval_metrics(eval_df: pd.DataFrame) -> dict:
    """Aggregate eval_metrics.csv into per-run summaries."""
    if eval_df is None or len(eval_df) == 0:
        return {
            "eval_collision_rate": None, "eval_success_rate": None,
            "eval_timeout_rate":   None, "eval_abort_rate":   None,
            "mean_return_eval":    None,
            "min_ttc_eval":        None,
            "mean_ttc_eval":       None,
            "n_eval_episodes":     0,
        }
    n = len(eval_df)
    ts = eval_df.get("terminal_state")

    def frac(state):
        if ts is None:
            return None
        return float((ts == state).sum()) / n

    return {
        "eval_collision_rate": frac("collision"),
        "eval_success_rate":   frac("success"),
        "eval_timeout_rate":   frac("timeout"),
        "eval_abort_rate":     frac("abort"),
        "mean_return_eval":    float(eval_df["return_total"].mean()) if "return_total" in eval_df else None,
        "min_ttc_eval":        float(eval_df["min_ttc"].min())        if "min_ttc"        in eval_df else None,
        "mean_ttc_eval":       float(eval_df["mean_ttc"].mean())      if "mean_ttc"       in eval_df else None,
        "n_eval_episodes":     int(n),
    }
