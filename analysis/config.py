"""Analysis pipeline configuration constants.

All Phase 1E modules import their settings from here so the colour palette,
method ordering, and statistical hyperparameters stay consistent across
tables and figures.

Phase 1F: result-affecting analysis constants (FINAL_WINDOW_FRAC, ALPHA,
BOOTSTRAP_*, PRIMARY_METRICS, etc.) are now sourced from
``config_frozen_v1.yaml :: analysis``. Plot styling (METHOD_COLORS, etc.)
and schema constants (EXPECTED_*) stay hard-coded per Phase 1E decision #1.
"""

from __future__ import annotations

# Phase 1F: pull result-affecting constants from the frozen config.
from config_loader import get_config as _get_frozen_config

_FROZEN = _get_frozen_config()
_ANALYSIS_CFG = _FROZEN["analysis"]

# ── Final-window metrics ────────────────────────────────────────────────
FINAL_WINDOW_FRAC = float(_ANALYSIS_CFG["final_window_frac"])  # last N% of iters

# ── Statistical tests ───────────────────────────────────────────────────
ALPHA = float(_ANALYSIS_CFG["alpha"])
HOLM_FAMILIES = tuple(_ANALYSIS_CFG["holm_families"])
PRIMARY_METRICS = tuple(_ANALYSIS_CFG["primary_metrics"])
BOOTSTRAP_N = int(_ANALYSIS_CFG["bootstrap_n"])
BOOTSTRAP_CI = float(_ANALYSIS_CFG["bootstrap_ci"])
RNG_SEED_BOOTSTRAP = int(_ANALYSIS_CFG["bootstrap_rng_seed"])  # determinism

# ── Plotting (color palette + ordering shared across all figures) ──────
METHOD_COLORS = {
    "drppo":        "#888888",   # gray (baseline)
    "hjb_aux":      "#1f77b4",   # blue
    "soft_hjb_aux": "#ff7f0e",   # orange
    "eikonal_aux":  "#2ca02c",   # green
    "cbf_aux":      "#d62728",   # red
    "fusion_aux":   "#9467bd",   # purple
    "rule_based":   "#000000",   # black (reference)
}

METHOD_LABELS = {
    "drppo":        "DRPPO",
    "hjb_aux":      "HJB",
    "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux":  "Eikonal",
    "cbf_aux":      "CBF",
    "fusion_aux":   "Fusion",
    "rule_based":   "Rule-based",
}

METHOD_ORDER = (
    "drppo",
    "hjb_aux",
    "soft_hjb_aux",
    "eikonal_aux",
    "cbf_aux",
    "fusion_aux",
    "rule_based",
)

PDE_METHODS = ("hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux")

MATPLOTLIB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,        # screen render dpi (savefig overrides for output)
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# ── Loader ──────────────────────────────────────────────────────────────
RESULTS_ROOT_DEFAULT = "results/ablation"
ANALYSIS_OUTPUT_DEFAULT = "results/analysis"

# Failure rate gate: if more than this fraction of discovered runs fail
# quality checks, the orchestrator prompts before continuing.
FAILURE_RATE_GATE = float(_ANALYSIS_CFG["failure_rate_gate"])

# ── Schema (canonical column lists for quality checks) ─────────────────
EXPECTED_METRICS_COLUMNS = (
    "iteration", "total_steps", "wall_time_seconds", "iter_time_seconds",
    "env_step_time_seconds", "learn_step_time_seconds",
    "residual_compute_time_seconds",
    "L_total", "L_policy", "L_value", "L_entropy",
    "L_residual_optimality", "L_residual_safety", "L_distill",
    "mean_reward", "mean_episode_length", "n_episodes",
    "n_collisions", "n_successes", "n_timeouts", "n_aborts",
    "action_dist_stop", "action_dist_creep", "action_dist_yield",
    "action_dist_go", "action_dist_abort",
)

EXPECTED_META_KEYS = (
    "run_id", "start_time_iso", "end_time_iso", "wall_time_seconds",
    "method", "scenario", "ego_maneuver", "seed", "intent_on",
    "total_steps_target", "total_steps_actual", "convergence_reason",
    "git_commit", "git_branch", "git_dirty", "hostname", "device",
    "torch_version", "python_version", "config", "result_summary",
)

EXPECTED_CONFIG_KEYS = (
    "lr", "gamma", "gae_lambda", "clip_eps", "ent_coef", "vf_coef",
    "max_grad_norm", "n_epochs_per_update", "batch_size", "n_steps",
    "policy_hidden_size", "policy_n_layers",
    "gru_hidden_size", "gru_n_layers",
    "alpha_cbf", "tau_soft", "w_fail", "barrier_offset",
    "lambda_residual", "lambda_distill", "lambda_actor_kl",
    "collocation_size",
    "w_optimality", "w_safety",  # Phase 1C fusion
)

EVAL_METRICS_COLUMNS = (
    "iteration", "eval_episode_idx", "seed", "scenario", "ego_maneuver",
    "return_total", "episode_length", "terminal_state",
    "min_ttc", "mean_ttc", "min_distance_to_collision",
    "ego_max_speed", "n_action_changes",
)
