"""Evaluate PDE-family and DRPPO baseline trained models."""

from __future__ import annotations

import argparse
import os
import csv
import json
import time
import collections
from pathlib import Path
import numpy as np

try:
    import torch
    torch.set_num_threads(2)
except ImportError:
    torch = None

from env.sumo_env import SumoEnv, ACTION_NAMES
from experiments.pde.run_metadata import EVAL_METRICS_COLUMNS
from experiments.pde.trajectory_logger import TrajectoryLogger


# ── SPEC_PHASE_2_FOLLOWUP_EVAL_FIX (Step 2) ────────────────────────────────
# Architecture resolution: CLI > checkpoint's sibling meta.json > YAML defaults.
# Returns dict with: gru_hidden_size, policy_hidden_size, gru_n_layers, policy_n_layers.
def _resolve_architecture(args, checkpoint_path: str | None) -> dict:
    from config_loader import get_config
    arch_yaml = get_config()["architecture"]
    resolved = {
        "gru_hidden_size":    int(arch_yaml["gru_hidden_size"]),
        "policy_hidden_size": int(arch_yaml["policy_hidden_size"]),
        "gru_n_layers":       int(arch_yaml["gru_n_layers"]),
        "policy_n_layers":    int(arch_yaml["policy_n_layers"]),
    }

    meta_arch_loaded = False
    if checkpoint_path:
        meta_path = Path(checkpoint_path).parent / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                meta_cfg = meta.get("config") or {}
                for key in resolved:
                    if key in meta_cfg and meta_cfg[key] is not None:
                        resolved[key] = int(meta_cfg[key])
                meta_arch_loaded = True
            except Exception as e:
                print(f"[eval] WARN: could not parse meta.json next to checkpoint ({e}); falling back to YAML defaults.")

    cli_overrides = []
    for key in resolved:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            resolved[key] = int(cli_value)
            cli_overrides.append(f"{key}={cli_value}")

    src_parts = []
    if cli_overrides:
        src_parts.append(f"CLI ({', '.join(cli_overrides)})")
    src_parts.append("meta.json" if meta_arch_loaded else "YAML")
    print(f"[eval] Architecture resolved from: {' + '.join(src_parts)}")
    print(f"[eval]   gru_hidden_size={resolved['gru_hidden_size']}")
    print(f"[eval]   policy_hidden_size={resolved['policy_hidden_size']}")
    print(f"[eval]   gru_n_layers={resolved['gru_n_layers']}")
    print(f"[eval]   policy_n_layers={resolved['policy_n_layers']}")
    return resolved


def _classify_terminal_state(info: dict, term: bool, trunc: bool) -> str:
    """Resolve canonical terminal_state string for Phase 1A eval_metrics.csv."""
    if info.get("collision"):
        return "collision"
    if info.get("success"):
        return "success"
    if info.get("aborted"):
        return "abort"
    if info.get("timeout") or (trunc and not term):
        return "timeout"
    return "timeout"


def eval_model(env, policy, n_episodes: int, deterministic: bool, seed: int,
               save_failures: bool = False, max_failures: int = 10,
               fail_dir: str | None = None, fail_prefix: str = "",
               scenario: str = "", ego_maneuver: str = "",
               iteration: int = 0):
    """Run n_episodes of eval. Returns (aggregated_metrics, per_episode_rows,
    collision_episodes) where collision_episodes is a list of dicts ready for
    the Phase 1A TrajectoryLogger.
    """
    returns, coll_eps, success_eps, pothole_hits = [], [], [], []
    ttc_means, ttc_mins, ttc_p10s = [], [], []
    action_entropies, hard_brake_counts, row_violation_counts = [], [], []
    action_counts_all = np.zeros(5, dtype=int)
    switching_rates = []
    decision_latencies = []
    failure_count = 0

    # Phase 1A per-episode rows + collision step buffers.
    per_episode_rows = []
    collision_episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy.reset_hidden()
        r_tot, coll, pot = 0, 0, 0
        ttc_list = []
        trajectory = []
        ep_success = False
        actions_this_ep = []
        hard_brakes = 0
        prev_ego_v = None
        row_violations = 0
        prev_action = None
        switches_this_ep = 0
        zone_entry_step = None
        first_go_step = None

        # Phase 1A: per-step trajectory buffer for the current episode.
        ep_step_buffer = []
        ep_max_speed = 0.0
        ep_action_changes = 0
        last_term = False
        last_trunc = False
        last_info = {}

        for step_i in range(500):
            a, _, _, _ = policy.get_action(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            r_tot += r
            last_term = bool(term)
            last_trunc = bool(trunc)
            last_info = info

            actions_this_ep.append(int(a))

            ego_raw = info.get("raw_obs", {}).get("ego", {}) if isinstance(info.get("raw_obs"), dict) else {}
            ego_p = ego_raw.get("p", (0.0, 0.0))
            try:
                ego_x_now = float(ego_p[0])
                ego_y_now = float(ego_p[1])
            except (TypeError, IndexError):
                ego_x_now = ego_y_now = 0.0
            n_agents_now = len(info.get("raw_obs", {}).get("agents", []) or []) if isinstance(info.get("raw_obs"), dict) else 0
            ego_v_now = float(info.get("ego_speed", ego_raw.get("v", 0.0)))
            ep_max_speed = max(ep_max_speed, ego_v_now)
            if prev_action is not None and int(a) != prev_action:
                ep_action_changes += 1
            ep_step_buffer.append({
                "step": step_i,
                "ego_x": ego_x_now,
                "ego_y": ego_y_now,
                "ego_psi": float(ego_raw.get("psi", 0.0)),
                "ego_v": ego_v_now,
                "ego_a": float(ego_raw.get("a", 0.0)),
                "action": int(a),
                "reward": float(r),
                "min_ttc": float(info.get("ttc_min", float("inf"))),
                "n_agents": int(n_agents_now),
                "collision_agent_id": "none",
                "terminal_flag": int(bool(term or trunc) and bool(info.get("collision", False))),
            })

            # Hard-brake detection
            ego_v = info.get("ego_speed", 0.0)
            if prev_ego_v is not None and (prev_ego_v - ego_v) > 3.0:
                hard_brakes += 1
            prev_ego_v = ego_v

            # ROW violation heuristic
            nearest = info.get("nearest_agent_dist", 1e9)
            if nearest < 10.0 and a in (1, 3):
                row_violations += 1

            # Action switching
            if prev_action is not None and a != prev_action:
                switches_this_ep += 1
            prev_action = a

            # Decision latency: track when ego enters conflict zone and
            # when it first chooses GO (action==3) after zone entry
            d_cz = float(info.get("built", {}).get("s_geom", np.zeros(12))[1])
            if zone_entry_step is None and d_cz < 1.0:
                zone_entry_step = step_i
            if zone_entry_step is not None and first_go_step is None and int(a) == 3:
                first_go_step = step_i

            if save_failures:
                ego = info.get("raw_obs", {}).get("ego", {})
                p = ego.get("p", np.zeros(2))
                trajectory.append({
                    "step": step_i, "action": a,
                    "action_name": ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else "?",
                    "reward": r,
                    "ego_x": float(p[0]) if hasattr(p, '__len__') else 0,
                    "ego_y": float(p[1]) if hasattr(p, '__len__') else 0,
                    "ego_v": ego.get("v", 0),
                    "ttc_min": info.get("ttc_min", 10.0),
                    "collision": 1 if info.get("collision", False) else 0,
                    "d_cz": float(info.get("built", {}).get("s_geom", np.zeros(12))[1]),
                })

            if info.get("collision", False):
                coll += 1
            if info.get("in_pothole", False):
                pot += 1
            ttc_list.append(info.get("ttc_min", 10.0))
            if term or trunc:
                ep_success = info.get("success", False)
                break

        # Save failure trajectory
        if save_failures and coll > 0 and failure_count < max_failures and fail_dir:
            os.makedirs(fail_dir, exist_ok=True)
            fail_path = os.path.join(fail_dir, f"{fail_prefix}ep{ep}.csv")
            if trajectory:
                with open(fail_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(trajectory[0].keys()))
                    writer.writeheader()
                    writer.writerows(trajectory)
                failure_count += 1

        # Phase 1A: per-episode row + collision step buffer collection.
        terminal_state = _classify_terminal_state(last_info, last_term, last_trunc)
        ttc_arr_local = np.array(ttc_list) if ttc_list else np.array([float("inf")])
        per_episode_rows.append({
            "iteration": int(iteration),
            "eval_episode_idx": int(ep),
            "seed": int(seed + ep),
            "scenario": str(scenario),
            "ego_maneuver": str(ego_maneuver),
            "return_total": float(r_tot),
            "episode_length": int(len(actions_this_ep)),
            "terminal_state": terminal_state,
            "min_ttc": float(np.min(ttc_arr_local)),
            "mean_ttc": float(np.mean(ttc_arr_local)),
            "min_distance_to_collision": float(last_info.get("nearest_agent_dist", float("inf"))) if isinstance(last_info, dict) else float("inf"),
            "ego_max_speed": float(ep_max_speed),
            "n_action_changes": int(ep_action_changes),
        })
        if terminal_state == "collision":
            collision_episodes.append({
                "steps": ep_step_buffer,
                "scenario": str(scenario),
                "ego_maneuver": str(ego_maneuver),
                "seed": int(seed + ep),
                "episode_idx": int(ep),
                "terminal_step": max(len(ep_step_buffer) - 1, 0),
                "collision_agent_id": str(last_info.get("collision_agent_id", "unknown")) if isinstance(last_info, dict) else "unknown",
            })

        returns.append(r_tot)
        coll_eps.append(1 if coll > 0 else 0)
        success_eps.append(1 if ep_success else 0)
        pothole_hits.append(pot)
        ttc_arr = np.array(ttc_list) if ttc_list else np.array([10.0])
        ttc_means.append(float(np.mean(ttc_arr)))
        ttc_mins.append(float(np.min(ttc_arr)))
        ttc_p10s.append(float(np.percentile(ttc_arr, 10)))

        # Action entropy
        if actions_this_ep:
            counts = collections.Counter(actions_this_ep)
            total = len(actions_this_ep)
            probs = [c / total for c in counts.values()]
            ent = -sum(p * np.log(p + 1e-12) for p in probs)
            action_entropies.append(ent)
        else:
            action_entropies.append(0.0)

        hard_brake_counts.append(hard_brakes)
        row_violation_counts.append(row_violations)

        # Accumulate action counts
        for act in actions_this_ep:
            if 0 <= act < 5:
                action_counts_all[act] += 1

        # Switching rate for this episode
        n_steps_ep = len(actions_this_ep)
        if n_steps_ep > 1:
            switching_rates.append(switches_this_ep / (n_steps_ep - 1))
        else:
            switching_rates.append(0.0)

        # Decision latency for this episode
        if zone_entry_step is not None and first_go_step is not None:
            decision_latencies.append(float(first_go_step - zone_entry_step))
        else:
            decision_latencies.append(float("nan"))

    total_actions = int(action_counts_all.sum())
    if total_actions > 0:
        action_fracs = action_counts_all / total_actions
    else:
        action_fracs = np.zeros(5)

    valid_latencies = [v for v in decision_latencies if v == v]

    aggregated = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "collision_rate": float(np.mean(coll_eps)),
        "success_rate": float(np.mean(success_eps)),
        "pothole_hits_mean": float(np.mean(pothole_hits)),
        "mean_ttc": float(np.mean(ttc_means)),
        "min_ttc": float(np.mean(ttc_mins)),
        "ttc_p10_mean": float(np.mean(ttc_p10s)),
        "action_entropy_mean": float(np.mean(action_entropies)),
        "hard_brakes_per_ep_mean": float(np.mean(hard_brake_counts)),
        "row_violations_per_ep_mean": float(np.mean(row_violation_counts)),
        "action_stop_frac": float(action_fracs[0]),
        "action_creep_frac": float(action_fracs[1]),
        "action_yield_frac": float(action_fracs[2]),
        "action_go_frac": float(action_fracs[3]),
        "action_abort_frac": float(action_fracs[4]),
        "switching_rate_mean": float(np.mean(switching_rates)),
        "decision_latency_mean": float(np.mean(valid_latencies)) if valid_latencies else float("nan"),
        "decision_latency_frac_defined": float(len(valid_latencies) / max(len(decision_latencies), 1)),
    }
    return aggregated, per_episode_rows, collision_episodes


def main():
    eval_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (not required for rule_based)")
    parser.add_argument("--method", required=True,
                        choices=["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux",
                                 "drppo", "fusion_aux", "rule_based"])
    parser.add_argument("--episodes", type=int, default=100,
                        help="Eval episodes per seed per mode (3 seeds x 100 eps x 2 modes = 600 total)")
    parser.add_argument("--out_dir", default="results/pde")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4", "2_dense", "3_dense", "4_dense"])
    parser.add_argument("--ego_maneuver", default="stem_right",
                        choices=["stem_right", "stem_left", "right_left",
                                 "right_stem", "left_right", "left_stem"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no_buildings", action="store_true",
                        help="Disable static occlusion buildings (full visibility)")
    parser.add_argument("--style_filter", default=None, choices=["nominal", "adversarial"],
                        help="Filter agent behavioral styles for robustness ablation")
    parser.add_argument("--state_ablation", default=None, choices=["no_visibility"],
                        help="State ablation: remove specific feature groups")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Three eval seeds by default (disjoint from training)")
    parser.add_argument("--save_failures", action="store_true",
                        help="Save trajectory CSVs for episodes that end in collision")
    parser.add_argument("--max_failures", type=int, default=10)
    # SPEC_PHASE_2_FOLLOWUP Step 1: architecture override flags. Default=None
    # so we can distinguish "user didn't specify" from "user passed value
    # matching YAML." Resolution precedence in _resolve_architecture():
    # CLI > checkpoint's sibling meta.json > YAML defaults.
    parser.add_argument("--gru_hidden_size", type=int, default=None,
                        help="GRU hidden dim. Default: auto-detect from checkpoint's meta.json, then YAML.")
    parser.add_argument("--policy_hidden_size", type=int, default=None,
                        help="Policy MLP hidden dim. Default: auto-detect from checkpoint's meta.json, then YAML.")
    parser.add_argument("--gru_n_layers", type=int, default=None,
                        help="GRU layer count. Default: auto-detect from checkpoint's meta.json, then YAML.")
    parser.add_argument("--policy_n_layers", type=int, default=None,
                        help="Policy MLP layer count. Default: auto-detect from checkpoint's meta.json, then YAML.")
    args = parser.parse_args()

    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    env = SumoEnv(use_gui=args.gui, scenario_name=args.scenario,
                  ego_maneuver=args.ego_maneuver, use_intent=args.use_intent,
                  buildings=not args.no_buildings, style_filter=args.style_filter,
                  state_ablation=args.state_ablation)
    obs_dim = int(env.observation_space.shape[0])

    if args.method == "rule_based":
        from models.rule_based_policy import RuleBasedTTCPolicy
        policy = RuleBasedTTCPolicy(obs_dim=obs_dim, device=device)
    else:
        if args.checkpoint is None:
            parser.error(f"--checkpoint is required for method '{args.method}'")
        # Phase 3 followup: prefer Agent.from_checkpoint() — reconstructs the
        # agent using the architecture saved with the checkpoint. The
        # _resolve_architecture helper above is retained for backward-compat
        # (legacy checkpoints without an `arch` dict) but the canonical path
        # is now from_checkpoint().
        agent_cls_map = {
            "hjb_aux":      ("models.pde.hjb_aux_agent",      "HJBAuxAgent"),
            "soft_hjb_aux": ("models.pde.soft_hjb_aux_agent", "SoftHJBAuxAgent"),
            "eikonal_aux":  ("models.pde.eikonal_aux_agent",  "EikonalAuxAgent"),
            "cbf_aux":      ("models.pde.cbf_aux_agent",      "CBFAuxAgent"),
            "drppo":        ("models.drppo",                  "DRPPO"),
            "fusion_aux":   ("models.pde.fusion_aux_agent",   "FusionAuxAgent"),
        }
        if args.method not in agent_cls_map:
            parser.error(f"Unknown method '{args.method}' for agent construction")
        mod_name, cls_name = agent_cls_map[args.method]
        import importlib
        cls = getattr(importlib.import_module(mod_name), cls_name)
        try:
            policy = cls.from_checkpoint(args.checkpoint, device=device)
        except ValueError as e:
            if "no 'arch.obs_dim'" in str(e):
                # Legacy checkpoint path: fall back to the older
                # _resolve_architecture flow (CLI > meta.json > YAML).
                print(f"[eval] {e} — falling back to legacy arch-resolution path.")
                arch = _resolve_architecture(args, args.checkpoint)
                hidden_dim = arch["gru_hidden_size"]
                policy = cls(obs_dim=obs_dim, hidden_dim=hidden_dim, device=device)
                policy.load(args.checkpoint, strict_arch=False)
            else:
                raise

    os.makedirs(args.out_dir, exist_ok=True)
    fail_dir = os.path.join(args.out_dir, "failures") if args.save_failures else None
    fail_prefix = f"fail_{args.method}_{args.scenario}_{args.ego_maneuver}_"

    # Phase 1A: shared trajectory logger (collisions during eval ring-buffer
    # alongside training collisions in the same out_dir).
    trajectory_logger = TrajectoryLogger(output_dir=args.out_dir, max_episodes=50)
    eval_metrics_path = os.path.join(args.out_dir, "eval_metrics.csv")
    new_eval_metrics_file = not os.path.isfile(eval_metrics_path)
    eval_metrics_f = open(eval_metrics_path, "a", newline="")
    eval_metrics_writer = csv.writer(eval_metrics_f)
    if new_eval_metrics_file:
        eval_metrics_writer.writerow(EVAL_METRICS_COLUMNS)

    all_results = {}
    for seed in args.seeds:
        for mode, det in [("deterministic", True), ("stochastic", False)]:
            print(f"Eval seed={seed} [{mode}]...")
            m, per_ep_rows, coll_eps = eval_model(
                env, policy, args.episodes, det, seed,
                save_failures=args.save_failures,
                max_failures=args.max_failures,
                fail_dir=fail_dir, fail_prefix=f"{fail_prefix}s{seed}_{mode}_",
                scenario=args.scenario, ego_maneuver=args.ego_maneuver,
                iteration=0,
            )
            all_results[(seed, mode)] = m
            for row in per_ep_rows:
                eval_metrics_writer.writerow([row[col] for col in EVAL_METRICS_COLUMNS])
            for ep_payload in coll_eps:
                trajectory_logger.log_collision_episode(
                    steps=ep_payload.get("steps", []),
                    scenario=ep_payload.get("scenario", args.scenario),
                    ego_maneuver=ep_payload.get("ego_maneuver", args.ego_maneuver),
                    seed=int(ep_payload.get("seed", seed)),
                    episode_idx=int(ep_payload.get("episode_idx", 0)),
                    terminal_step=int(ep_payload.get("terminal_step", 0)),
                    collision_agent_id=str(ep_payload.get("collision_agent_id", "unknown")),
                )
            print(f"  return={m['mean_return']:.2f} coll={m['collision_rate']:.3f} "
                  f"success={m['success_rate']:.3f} ttc={m['mean_ttc']:.2f} "
                  f"ent={m['action_entropy_mean']:.2f} brakes={m['hard_brakes_per_ep_mean']:.1f}")
    eval_metrics_f.close()

    csv_path = os.path.join(args.out_dir, f"eval_{args.method}_{args.scenario}_{args.ego_maneuver}.csv")
    header = ["seed", "eval_mode", "mean_return", "std_return",
              "collision_rate", "success_rate", "pothole_hits_mean",
              "mean_ttc", "min_ttc", "ttc_p10_mean",
              "action_entropy_mean", "hard_brakes_per_ep_mean",
              "row_violations_per_ep_mean",
              "action_stop_frac", "action_creep_frac", "action_yield_frac",
              "action_go_frac", "action_abort_frac",
              "switching_rate_mean", "decision_latency_mean",
              "decision_latency_frac_defined"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (seed, mode), m in all_results.items():
            w.writerow([seed, mode] + [m[h] for h in header[2:]])
    print(f"Saved {csv_path}")

    # Eval provenance metadata
    eval_meta = {
        "method": args.method,
        "checkpoint_path": args.checkpoint if args.checkpoint else "N/A (rule_based)",
        "scenario": args.scenario,
        "ego_maneuver": args.ego_maneuver,
        "no_buildings": args.no_buildings,
        "style_filter": args.style_filter,
        "state_ablation": args.state_ablation,
        "n_eval_seeds": len(args.seeds),
        "eval_seeds": list(args.seeds),
        "episodes_per_seed": args.episodes,
        "total_eval_episodes": len(args.seeds) * args.episodes * 2,
        "eval_wall_time_seconds": time.time() - eval_start_time,
    }
    eval_meta_path = os.path.join(args.out_dir,
        f"meta_eval_{args.method}_{args.scenario}_{args.ego_maneuver}.json")
    with open(eval_meta_path, "w") as f:
        json.dump(eval_meta, f, indent=2, default=str)

    env.close()


if __name__ == "__main__":
    main()
