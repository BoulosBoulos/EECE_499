"""Provenance metadata helper + shared logging schema for PDE training.

Defines the canonical metrics.csv / eval_metrics.csv column ordering used by
every Phase 1A-instrumented training script and eval.py. Any script that
deviates will silently break downstream analysis (Phase 1E), so all scripts
import these constants rather than redeclaring the schema.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:
    torch = None

# Phase 1F: CLI defaults are sourced from config_frozen_v1.yaml. Any change to
# a value below must be made in the YAML; the lock check at orchestrator/
# training startup will warn if the config drifts post-freeze.
from config_loader import get_config as _get_frozen_config  # noqa: E402

_FROZEN = _get_frozen_config()
_PPO = _FROZEN["ppo"]
_ARCH = _FROZEN["architecture"]
_TRAIN = _FROZEN["training"]


METRICS_COLUMNS = [
    "iteration",
    "total_steps",
    "wall_time_seconds",
    "iter_time_seconds",
    "env_step_time_seconds",
    "learn_step_time_seconds",
    "residual_compute_time_seconds",
    "L_total",
    "L_policy",
    "L_value",
    "L_entropy",
    "L_residual_optimality",
    "L_residual_safety",
    "L_distill",
    "mean_reward",
    "mean_episode_length",
    "n_episodes",
    "n_collisions",
    "n_successes",
    "n_timeouts",
    "n_aborts",
    "action_dist_stop",
    "action_dist_creep",
    "action_dist_yield",
    "action_dist_go",
    "action_dist_abort",
]


EVAL_METRICS_COLUMNS = [
    "iteration",
    "eval_episode_idx",
    "seed",
    "scenario",
    "ego_maneuver",
    "return_total",
    "episode_length",
    "terminal_state",
    "min_ttc",
    "mean_ttc",
    "min_distance_to_collision",
    "ego_max_speed",
    "n_action_changes",
]


# ---- Phase 1B uniform config schema -----------------------------------------
# Every script's meta.json["config"] must contain *all* of these keys, with
# values populated from the matching CLI flag, or null where the flag does
# not apply to that method. Downstream analysis (Phase 1E) keys off this set.

UNIFORM_CONFIG_KEYS = (
    # Core PPO
    "lr", "gamma", "gae_lambda", "clip_eps",
    "ent_coef", "vf_coef", "max_grad_norm",
    "n_epochs_per_update", "batch_size", "n_steps",
    # Architecture
    "policy_hidden_size", "policy_n_layers",
    "gru_hidden_size", "gru_n_layers",
    # PDE-specific (null where not applicable)
    "alpha_cbf", "tau_soft", "w_fail", "barrier_offset",
    "lambda_residual", "lambda_distill", "lambda_actor_kl",
    "collocation_size",
    # Fusion-specific (Phase 1C; null for non-fusion methods)
    "w_optimality", "w_safety",
)


VALID_METHODS = (
    "drppo",
    "hjb_aux",
    "soft_hjb_aux",
    "eikonal_aux",
    "cbf_aux",
    "fusion_aux",
)


SCENARIO_CHOICES = ["1a", "1b", "1c", "1d", "2", "3", "4",
                    "2_dense", "3_dense", "4_dense"]
EGO_MANEUVER_CHOICES = ["stem_right", "stem_left", "right_left",
                        "right_stem", "left_right", "left_stem"]


def add_common_cli_args(parser):
    """Register the CLI flags that every Phase 1B-instrumented training
    script must accept identically. Returns the parser for chaining.

    PDE-specific flags (lambda_residual, lambda_distill, tau_soft, alpha_cbf,
    w_fail, barrier_offset, lambda_actor_kl, collocation_size) are NOT added
    here -- each PDE-method script registers only the ones that apply.
    """
    # --- Core PPO (defaults from config_frozen_v1.yaml :: ppo) -----------
    parser.add_argument("--lr", type=float, default=float(_PPO["lr"]),
                        help="PPO learning rate")
    parser.add_argument("--gamma", type=float, default=float(_PPO["gamma"]),
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=float(_PPO["gae_lambda"]),
                        help="GAE lambda")
    parser.add_argument("--clip_eps", type=float, default=float(_PPO["clip_eps"]),
                        help="PPO clip range (epsilon)")
    parser.add_argument("--ent_coef", type=float, default=float(_PPO["ent_coef"]),
                        help="Entropy bonus coefficient")
    parser.add_argument("--vf_coef", type=float, default=float(_PPO["vf_coef"]),
                        help="Value function loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=float(_PPO["max_grad_norm"]),
                        help="Gradient norm clip")
    parser.add_argument("--n_epochs_per_update", type=int, default=int(_PPO["n_epochs_per_update"]),
                        help="PPO epochs per update")
    parser.add_argument("--batch_size", type=int, default=int(_PPO["batch_size"]),
                        help="Minibatch size for PPO")
    parser.add_argument("--n_steps", type=int, default=int(_PPO["n_steps"]),
                        help="Steps per rollout")
    # --- Architecture (defaults from config_frozen_v1.yaml :: architecture) ---
    parser.add_argument("--policy_hidden_size", type=int, default=int(_ARCH["policy_hidden_size"]),
                        help="Policy MLP hidden size")
    parser.add_argument("--policy_n_layers", type=int, default=int(_ARCH["policy_n_layers"]),
                        help="Number of MLP layers in policy")
    parser.add_argument("--gru_hidden_size", type=int, default=int(_ARCH["gru_hidden_size"]),
                        help="DRPPO GRU hidden size")
    parser.add_argument("--gru_n_layers", type=int, default=int(_ARCH["gru_n_layers"]),
                        help="DRPPO GRU layers")
    # --- Training control (defaults from config_frozen_v1.yaml :: training) --
    parser.add_argument("--total_steps", type=int, default=int(_TRAIN["default_total_steps"]),
                        help="Target environment steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for everything")
    parser.add_argument("--scenario", default="1a",
                        choices=SCENARIO_CHOICES,
                        help="Scenario name")
    parser.add_argument("--ego_maneuver", default="stem_right",
                        choices=EGO_MANEUVER_CHOICES,
                        help="Ego maneuver")
    parser.add_argument("--use_intent", action="store_true",
                        help="Enable intent encoder")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write outputs (Phase 1B canonical name)")
    parser.add_argument("--n_eval_episodes", type=int, default=int(_TRAIN["n_eval_episodes"]),
                        help="Eval episodes per checkpoint")
    parser.add_argument("--eval_every_n_iter", type=int, default=int(_TRAIN["eval_every_n_iter"]),
                        help="Run eval every N iterations")
    parser.add_argument("--save_every_n_iter", type=int, default=int(_TRAIN["save_every_n_iter"]),
                        help="Save checkpoint every N iterations")
    # --- Legacy args kept for backward compatibility ---
    parser.add_argument("--out_dir", type=str, default=None,
                        help="(Legacy) alias for --output_dir; --output_dir wins if both given")
    parser.add_argument("--algo_config", default="configs/algo/default.yaml",
                        help="(Legacy) yaml config; explicit CLI flags override yaml values")
    parser.add_argument("--sumo_gui", action="store_true",
                        help="(Legacy) launch SUMO GUI")
    parser.add_argument("--no_buildings", action="store_true",
                        help="Disable static occlusion buildings")
    parser.add_argument("--style_filter", default=None,
                        choices=["nominal", "adversarial"],
                        help="Filter agent behavioral styles for robustness ablation")
    parser.add_argument("--state_ablation", default=None,
                        choices=["no_visibility"],
                        help="State ablation: remove specific feature groups")
    return parser


def resolve_output_dir(args) -> str:
    """Return the canonical output_dir, honouring the new --output_dir flag
    and falling back to the legacy --out_dir for backward compatibility.
    Errors out if neither is set (the new flag is documented as required).
    """
    out = getattr(args, "output_dir", None) or getattr(args, "out_dir", None)
    if not out:
        raise SystemExit("error: --output_dir is required (or use legacy --out_dir).")
    return out


def build_uniform_config(args, *, method: str,
                         lambda_residual=None, lambda_distill=None,
                         lambda_actor_kl=None, tau_soft=None,
                         alpha_cbf=None, w_fail=None,
                         barrier_offset=None, collocation_size=None,
                         w_optimality=None, w_safety=None) -> Dict[str, Any]:
    """Construct the meta.json["config"] dict with the uniform key set
    required by Phase 1B test 24.2 / Phase 1C test 25.x. PDE-specific values
    are passed in explicitly per-method; leave them at None to record null.

    Phase 1C kwargs:
      w_optimality, w_safety -- non-null only for fusion_aux. For all other
      methods both stay None and serialize to null (uniform schema).
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"unknown method {method!r}; must be one of {VALID_METHODS}"
        )
    return {
        # Core PPO
        "lr": float(args.lr),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_eps": float(args.clip_eps),
        "ent_coef": float(args.ent_coef),
        "vf_coef": float(args.vf_coef),
        "max_grad_norm": float(args.max_grad_norm),
        "n_epochs_per_update": int(args.n_epochs_per_update),
        "batch_size": int(args.batch_size),
        "n_steps": int(args.n_steps),
        # Architecture
        "policy_hidden_size": int(args.policy_hidden_size),
        "policy_n_layers": int(args.policy_n_layers),
        "gru_hidden_size": int(args.gru_hidden_size),
        "gru_n_layers": int(args.gru_n_layers),
        # PDE-specific (null where not applicable)
        "alpha_cbf": (None if alpha_cbf is None else float(alpha_cbf)),
        "tau_soft": (None if tau_soft is None else float(tau_soft)),
        "w_fail": (None if w_fail is None else float(w_fail)),
        "barrier_offset": (None if barrier_offset is None else float(barrier_offset)),
        "lambda_residual": (None if lambda_residual is None else float(lambda_residual)),
        "lambda_distill": (None if lambda_distill is None else float(lambda_distill)),
        "lambda_actor_kl": (None if lambda_actor_kl is None else float(lambda_actor_kl)),
        "collocation_size": (None if collocation_size is None else int(collocation_size)),
        # Fusion-specific (null for all non-fusion methods)
        "w_optimality": (None if w_optimality is None else float(w_optimality)),
        "w_safety": (None if w_safety is None else float(w_safety)),
    }


def _git_info() -> Dict[str, Any]:
    """Walk up the directory tree looking for a .git folder, then capture
    commit / branch / dirty state from that repository. Falls back to a
    schema-valid placeholder if no git repo is found anywhere up the tree.
    """
    repo_dir = None
    cur = Path(os.getcwd())
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            repo_dir = parent
            break

    if repo_dir is None:
        return {"git_commit": "00000000", "git_branch": "unknown", "git_dirty": True}

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(dirty_output)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "00000000", "git_branch": "unknown", "git_dirty": True}

    return {"git_commit": commit[:8], "git_branch": branch, "git_dirty": dirty}


def _device_string() -> str:
    if torch is None:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _torch_version() -> str:
    return torch.__version__ if torch is not None else "unavailable"


def write_meta_start(
    output_dir: str,
    method: str,
    scenario: str,
    ego_maneuver: str,
    seed: Optional[int],
    intent_on: bool,
    total_steps_target: int,
    config: Dict[str, Any],
) -> str:
    """Write meta.json at training start. Returns run_id."""
    # Phase 3F Step 12: stamp criterion_version directly onto every job's
    # meta.json at training-start time so downstream analysis can key off
    # the run-time criterion without a separate post-run annotation step.
    from analysis.calibration_analysis import CRITERION_VERSION
    run_id = str(uuid.uuid4())
    meta = {
        "run_id": run_id,
        "start_time_iso": datetime.now(timezone.utc).isoformat(),
        "end_time_iso": None,
        "wall_time_seconds": None,
        "method": method,
        "scenario": scenario,
        "ego_maneuver": ego_maneuver,
        "seed": seed,
        "intent_on": bool(intent_on),
        "total_steps_target": int(total_steps_target),
        "total_steps_actual": None,
        "convergence_reason": None,
        "criterion_version": CRITERION_VERSION,
        **_git_info(),
        "hostname": socket.gethostname(),
        "device": _device_string(),
        "torch_version": _torch_version(),
        "python_version": platform.python_version(),
        "config": config,
        "result_summary": None,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return run_id


def write_meta_end(
    output_dir: str,
    total_steps_actual: int,
    convergence_reason: str,
    result_summary: Dict[str, Any],
) -> None:
    """Update meta.json at training end."""
    path = os.path.join(output_dir, "meta.json")
    with open(path) as f:
        meta = json.load(f)
    end_time = datetime.now(timezone.utc)
    start_iso = meta.get("start_time_iso") or end_time.isoformat()
    start_time = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    meta["end_time_iso"] = end_time.isoformat()
    meta["wall_time_seconds"] = (end_time - start_time).total_seconds()
    meta["total_steps_actual"] = int(total_steps_actual)
    meta["convergence_reason"] = str(convergence_reason)
    meta["result_summary"] = result_summary
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
