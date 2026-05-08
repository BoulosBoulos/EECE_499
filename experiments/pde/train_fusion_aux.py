"""Train fusion (Soft-HJB optimality + CBF safety) auxiliary critic on SUMO T-intersection.

Phase 1C — sixth PDE method. See SPEC_PHASE_1C_FUSION_ARCHITECTURE.md.

Note: fusion_aux's metrics.csv has a SUPERSET schema vs the canonical
METRICS_COLUMNS — it appends `distill_loss_optimality_component` and
`distill_loss_safety_component`, the per-critic distillation diagnostics
introduced by the post-Phase-1C dual-critic fix (verification/fusion_bug_artifact.md).
Other methods' metrics.csv files retain the canonical 25-column shape.
"""

from __future__ import annotations

import argparse
import os
import csv
import time
import json
import subprocess
from pathlib import Path
import numpy as np

try:
    import torch
    torch.set_num_threads(2)
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

from experiments.pde.run_metadata import (
    METRICS_COLUMNS,
    add_common_cli_args,
    build_uniform_config,
    resolve_output_dir,
    write_meta_start,
    write_meta_end,
)
from experiments.pde.trajectory_logger import TrajectoryLogger
from config_loader import check_config_lock, get_method_config

METHOD_NAME = "fusion_aux"
_METHOD_CFG = get_method_config(METHOD_NAME)

# Fusion-specific metrics.csv schema: canonical METRICS_COLUMNS + the two
# per-critic distillation-component diagnostics produced by FusionAuxAgent
# after the dual-critic bug fix. Analysis code that reads fusion metrics
# should key on column name (DictReader) rather than positional index.
FUSION_METRICS_COLUMNS = list(METRICS_COLUMNS) + [
    "distill_loss_optimality_component",
    "distill_loss_safety_component",
]


def _load_config(path: str) -> dict:
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    add_common_cli_args(parser)
    parser.add_argument("--config", default="configs/pde/fusion_aux.yaml",
                        help="(Optional) yaml; explicit CLI flags below override yaml values")
    # Soft-HJB-specific flags (Decision H; defaults from config_frozen_v1.yaml).
    parser.add_argument("--lambda_residual", type=float,
                        default=float(_METHOD_CFG["lambda_residual"]),
                        help="Multiplier on the (w_o*L_soft + w_s*L_cbf) sum")
    parser.add_argument("--lambda_distill", type=float,
                        default=float(_METHOD_CFG["lambda_distill"]),
                        help="Weight for distillation loss (vs U_optimality)")
    parser.add_argument("--lambda_actor_kl", type=float,
                        default=float(_METHOD_CFG["lambda_actor_kl"]),
                        help="Weight for actor-KL alignment to Soft-HJB-implied policy")
    parser.add_argument("--tau_soft", type=float,
                        default=float(_METHOD_CFG["tau_soft"]),
                        help="Soft-HJB temperature")
    parser.add_argument("--collocation_size", type=int,
                        default=int(_METHOD_CFG["collocation_size"]),
                        help="Collocation samples per iteration")
    # CBF-specific flags (Decision H).
    parser.add_argument("--alpha_cbf", type=float,
                        default=float(_METHOD_CFG["alpha_cbf"]),
                        help="CBF class-K function gain (applied to U_safety)")
    parser.add_argument("--barrier_offset", type=float,
                        default=float(_METHOD_CFG["barrier_offset"]),
                        help="Offset for h(xi) = U_safety(xi) + offset")
    # Fusion-specific flags (Decision G — NEW in Phase 1C).
    parser.add_argument("--w_optimality", type=float,
                        default=float(_METHOD_CFG["w_optimality"]),
                        help="Weight on Soft-HJB residual in fusion sum")
    parser.add_argument("--w_safety", type=float,
                        default=float(_METHOD_CFG["w_safety"]),
                        help="Weight on CBF residual in fusion sum")
    args = parser.parse_args()
    _lock = check_config_lock(strict=False)
    if not _lock["matches"] and _lock.get("warning"):
        print(_lock["warning"])
    output_dir = resolve_output_dir(args)

    script_start_time = time.time()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    if args.seed is not None:
        np.random.seed(args.seed)
        if torch:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

    pde_cfg = _load_config(args.config)
    algo_cfg = _load_config(args.algo_config)
    os.makedirs(output_dir, exist_ok=True)

    from env.sumo_env import SumoEnv
    env_kwargs = {"use_gui": args.sumo_gui, "scenario_name": args.scenario,
                  "ego_maneuver": args.ego_maneuver, "use_intent": args.use_intent,
                  "buildings": not args.no_buildings, "style_filter": args.style_filter,
                  "state_ablation": args.state_ablation}
    obs_dim = int(SumoEnv(**env_kwargs).observation_space.shape[0])
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    from models.pde.fusion_aux_agent import FusionAuxAgent
    policy = FusionAuxAgent(
        obs_dim=obs_dim,
        n_actions=5,
        lr=float(args.lr),
        aux_lr=float(pde_cfg.get("aux_lr", _METHOD_CFG["aux_lr"])),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_eps),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        # Anchor / boundary (carry over from yaml; not exposed on CLI yet).
        lambda_anchor=float(pde_cfg.get("lambda_anchor", 1.0)),
        lambda_bc=float(pde_cfg.get("lambda_bc", 0.5)),
        # Fusion / Soft-HJB / CBF
        lambda_residual=float(args.lambda_residual),
        lambda_distill=float(args.lambda_distill),
        lambda_actor_kl=float(args.lambda_actor_kl),
        w_optimality=float(args.w_optimality),
        w_safety=float(args.w_safety),
        tau_soft=float(args.tau_soft),
        alpha_cbf=float(args.alpha_cbf),
        barrier_offset=float(args.barrier_offset),
        # Critic / collocation
        aux_hidden_dim=int(pde_cfg.get("aux_hidden_dim", 256)),
        collocation_ratio=float(pde_cfg.get("collocation_ratio", 0.7)),
        # Policy
        hidden_dim=int(args.gru_hidden_size),
        device=device,
        seed=int(args.seed) if args.seed is not None else 42,
    )

    n_steps = int(args.n_steps)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs_per_update)
    gamma = float(args.gamma)
    gae_lambda = float(args.gae_lambda)

    # ── Legacy CSV (per-iteration eval log; same shape as soft_hjb's) ─────
    csv_path = os.path.join(output_dir,
                            f"train_fusion_aux_{args.scenario}_{args.ego_maneuver}.csv")
    header = [
        "step", "episode_return", "episode_len", "actor_loss", "vf_loss",
        "collision_count", "collision_rate", "mean_ttc", "min_ttc", "entropy",
        "soft_residual_mean", "cbf_residual_mean",
        "anchor_loss_optimality", "anchor_loss_safety",
        "bc_loss_optimality", "bc_loss_safety",
        "distill_loss", "distill_gap", "actor_align_kl",
        "train_time_per_iter",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    # ── Phase 1A canonical metrics.csv + meta.json ────────────────────────
    # fusion_aux uses FUSION_METRICS_COLUMNS (= METRICS_COLUMNS + 2 fusion-
    # specific diagnostic columns); see module docstring.
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(FUSION_METRICS_COLUMNS)
    config_dict = build_uniform_config(
        args, method=METHOD_NAME,
        # Both Soft-HJB and CBF keys are populated for fusion (Decision I).
        lambda_residual=args.lambda_residual,
        lambda_distill=args.lambda_distill,
        lambda_actor_kl=args.lambda_actor_kl,
        tau_soft=args.tau_soft,
        alpha_cbf=args.alpha_cbf,
        barrier_offset=args.barrier_offset,
        collocation_size=args.collocation_size,
        # NEW Phase 1C fusion weights.
        w_optimality=args.w_optimality,
        w_safety=args.w_safety,
        # w_fail stays None (only used by Eikonal).
    )
    run_id = write_meta_start(
        output_dir=output_dir,
        method=METHOD_NAME,
        scenario=args.scenario,
        ego_maneuver=args.ego_maneuver,
        seed=args.seed,
        intent_on=bool(args.use_intent),
        total_steps_target=int(args.total_steps),
        config=config_dict,
    )
    trajectory_logger = TrajectoryLogger(output_dir=output_dir, max_episodes=50)
    training_start_time = time.time()

    from experiments.pde.collect_rollouts import collect_rollouts

    env = SumoEnv(**env_kwargs)
    eval_env = SumoEnv(**env_kwargs)
    EVAL_SEED_OFFSET = 10_000
    EVAL_EPISODES_PER_LOG = int(args.n_eval_episodes)
    log_interval_steps = int(args.eval_every_n_iter) * int(args.n_steps)
    step = 0
    ep_returns = []
    next_log_step = 0
    ckpt_frac = [0.5, 0.75, 1.0]
    ckpt_steps = [int(args.total_steps * f) for f in ckpt_frac]
    next_ckpt_idx = 0
    intermediate_ckpts = {}

    iteration = 0
    while step < args.total_steps:
        iteration += 1
        t_iter_start = time.time()
        rollout_t0 = time.time()
        obs_arr, actions_arr, log_probs_arr, returns_arr, advantages_arr, extra, hidden_arr = \
            collect_rollouts(env, policy, n_steps, gamma, gae_lambda)
        rollout_time = time.time() - rollout_t0
        env_step_time = float(extra.get("env_step_time", rollout_time))
        step += n_steps

        learn_t0 = time.time()
        residual_compute_time = 0.0
        for _ in range(n_epochs):
            perm = np.random.permutation(len(obs_arr))
            for start in range(0, len(obs_arr), batch_size):
                idx = perm[start:start + batch_size]
                if len(idx) == 0:
                    continue
                ex = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(obs_arr) else v)
                      for k, v in extra.items()} if extra else None
                h = hidden_arr[idx] if hidden_arr is not None else None
                _residual_t0 = time.time()
                metrics = policy.train_step(
                    obs_arr[idx], actions_arr[idx], log_probs_arr[idx],
                    returns_arr[idx], advantages_arr[idx],
                    hiddens=h, extra=ex,
                )
                residual_compute_time += time.time() - _residual_t0
        learn_step_time = time.time() - learn_t0
        train_time = time.time() - t_iter_start

        action_counts = extra.get("action_counts", [0, 0, 0, 0, 0])
        total_actions = float(sum(action_counts))
        action_dist = (
            [c / total_actions for c in action_counts] if total_actions > 0 else [0.0] * 5
        )
        n_eps = int(extra.get("n_episodes", 0))
        mean_reward_iter = float(np.sum(returns_arr) / max(n_eps, 1)) if n_eps > 0 else float(np.mean(returns_arr))
        mean_ep_len = float(extra.get("mean_episode_length", 0.0))
        # Decision J: BOTH residuals non-zero in metrics.csv for fusion.
        L_residual_opt = float(metrics.get("soft_residual_mean", 0.0))
        L_residual_saf = float(metrics.get("cbf_residual_mean", 0.0))
        L_distill_iter = float(metrics.get("distill_loss", 0.0))
        L_total_iter = (
            float(metrics.get("actor_loss", 0.0))
            + float(metrics.get("vf_loss", 0.0))
            + float(args.lambda_residual) * (
                float(args.w_optimality) * L_residual_opt
                + float(args.w_safety) * L_residual_saf
            )
            + L_distill_iter
        )
        with open(metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                iteration, step,
                time.time() - training_start_time, train_time,
                env_step_time, learn_step_time, residual_compute_time,
                L_total_iter,
                float(metrics.get("actor_loss", 0.0)),
                float(metrics.get("vf_loss", 0.0)),
                float(metrics.get("entropy", 0.0)),
                L_residual_opt, L_residual_saf, L_distill_iter,
                mean_reward_iter, mean_ep_len, n_eps,
                int(extra.get("n_collisions", 0)),
                int(extra.get("n_successes", 0)),
                int(extra.get("n_timeouts", 0)),
                int(extra.get("n_aborts", 0)),
                action_dist[0], action_dist[1], action_dist[2], action_dist[3], action_dist[4],
                float(metrics.get("distill_loss_optimality_component", 0.0)),
                float(metrics.get("distill_loss_safety_component", 0.0)),
            ])
        for ep_payload in extra.get("completed_collision_episodes", []) or []:
            trajectory_logger.log_collision_episode(
                steps=ep_payload.get("steps", []),
                scenario=args.scenario,
                ego_maneuver=args.ego_maneuver,
                seed=int(args.seed) if args.seed is not None else -1,
                episode_idx=trajectory_logger.episode_counter,
                terminal_step=int(ep_payload.get("terminal_step", 0)),
                collision_agent_id=str(ep_payload.get("collision_agent_id", "unknown")),
            )

        while next_ckpt_idx < len(ckpt_steps) and step >= ckpt_steps[next_ckpt_idx]:
            ck_step = ckpt_steps[next_ckpt_idx]
            ck_path = os.path.join(output_dir,
                f"model_fusion_aux_{args.scenario}_{args.ego_maneuver}_step{ck_step}.pt")
            policy.save(ck_path)
            intermediate_ckpts[ck_step] = ck_path
            next_ckpt_idx += 1

        if step >= next_log_step:
            eval_returns = []
            eval_colls = []
            eval_ttcs_all = []
            eval_ttcs_min = []
            iter_id = step // n_steps
            for ep_idx in range(EVAL_EPISODES_PER_LOG):
                eval_seed = (args.seed or 0) + EVAL_SEED_OFFSET + iter_id * 100 + ep_idx
                obs_e, _ = eval_env.reset(seed=eval_seed)
                policy.reset_hidden()
                ep_reward, ep_coll = 0.0, 0
                ep_ttcs = []
                for _ in range(500):
                    action_e, _, _, _ = policy.get_action(obs_e, deterministic=True)
                    obs_e, r_e, term_e, trunc_e, info_e = eval_env.step(action_e)
                    ep_reward += r_e
                    ep_ttcs.append(info_e.get("ttc_min", 10.0))
                    if info_e.get("collision", False):
                        ep_coll = 1
                    if term_e or trunc_e:
                        break
                eval_returns.append(ep_reward)
                eval_colls.append(ep_coll)
                if ep_ttcs:
                    eval_ttcs_all.extend(ep_ttcs)
                    eval_ttcs_min.append(min(ep_ttcs))
            ep_returns.append(float(np.mean(eval_returns)))
            mean_ttc = float(np.mean(eval_ttcs_all)) if eval_ttcs_all else float("nan")
            min_ttc = float(np.min(eval_ttcs_min)) if eval_ttcs_min else float("nan")
            coll_rate = float(np.mean(eval_colls))
            coll_count = int(sum(eval_colls))

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    step, np.mean(ep_returns[-10:]) if ep_returns else 0,
                    len(eval_returns), metrics.get("actor_loss", 0), metrics.get("vf_loss", 0),
                    coll_count, coll_rate, mean_ttc, min_ttc, metrics.get("entropy", 0),
                    metrics.get("soft_residual_mean", 0), metrics.get("cbf_residual_mean", 0),
                    metrics.get("anchor_loss_optimality", 0), metrics.get("anchor_loss_safety", 0),
                    metrics.get("bc_loss_optimality", 0), metrics.get("bc_loss_safety", 0),
                    metrics.get("distill_loss", 0), metrics.get("distill_gap", 0),
                    metrics.get("actor_align_kl", 0),
                    train_time,
                ])
            print(f"[fusion_aux] step={step} ret={np.mean(ep_returns[-10:]):.2f} "
                  f"coll={coll_count} ttc={mean_ttc:.2f} "
                  f"soft_res={metrics.get('soft_residual_mean', 0):.4f} "
                  f"cbf_res={metrics.get('cbf_residual_mean', 0):.4f} "
                  f"align_kl={metrics.get('actor_align_kl', 0):.4f}")
            next_log_step = step + log_interval_steps

    ckpt_path = os.path.join(output_dir, f"model_fusion_aux_{args.scenario}_{args.ego_maneuver}.pt")
    policy.save(ckpt_path)
    print(f"Saved fusion_aux to {ckpt_path}")

    intermediate_ckpts["final"] = ckpt_path

    if args.total_steps >= 50_000 and len(intermediate_ckpts) > 1:
        import shutil
        print("\nRunning post-training checkpoint selection (30 eps each)...")
        best_score = -np.inf
        best_ckpt = ckpt_path
        best_step = "final"
        final_score_val = None
        for ck_step, ck_path_i in intermediate_ckpts.items():
            # Phase 3 followup: from_checkpoint reconstructs with saved arch.
            tmp_policy = type(policy).from_checkpoint(ck_path_i, device=device)
            ck_returns = []
            ck_colls = 0
            for ep in range(30):
                eval_seed_ck = (args.seed or 0) + 100_000 + ep
                obs_c, _ = eval_env.reset(seed=eval_seed_ck)
                tmp_policy.reset_hidden()
                ep_r, ep_c = 0.0, 0
                for _ in range(500):
                    a_c, _, _, _ = tmp_policy.get_action(obs_c, deterministic=True)
                    obs_c, r_c, term_c, trunc_c, info_c = eval_env.step(a_c)
                    ep_r += r_c
                    if info_c.get("collision", False):
                        ep_c = 1; break
                    if term_c or trunc_c: break
                ck_returns.append(ep_r)
                ck_colls += ep_c
            mean_r = np.mean(ck_returns)
            coll_rate = ck_colls / 30
            score = -coll_rate * 100 + mean_r
            print(f"  step={ck_step}: return={mean_r:+.2f}, coll={coll_rate:.2f}, score={score:.2f}")
            if ck_step == "final":
                final_score_val = score
            if score > best_score:
                best_score = score
                best_ckpt = ck_path_i
                best_step = ck_step

        SWAP_MARGIN = 1.0
        if best_ckpt != ckpt_path and final_score_val is not None and (best_score - final_score_val) > SWAP_MARGIN:
            shutil.copy(best_ckpt, ckpt_path)
            print(f"  Best: step={best_step} (margin={best_score - final_score_val:.2f}). Copied to {ckpt_path}")
        else:
            print(f"  Keeping final checkpoint (best_step={best_step}, margin={best_score - (final_score_val or best_score):.2f})")

    # Provenance metadata (legacy file kept for backward compat).
    meta = {
        "method": METHOD_NAME,
        "scenario": args.scenario,
        "ego_maneuver": args.ego_maneuver,
        "seed": args.seed,
        "use_intent": args.use_intent,
        "total_steps": args.total_steps,
        "config_file": args.config,
        "algo_config_file": args.algo_config,
        "pde_config": pde_cfg,
        "algo_config": algo_cfg,
        "device": device,
        "lambda_residual": args.lambda_residual,
        "w_optimality": args.w_optimality,
        "w_safety": args.w_safety,
        "tau_soft": args.tau_soft,
        "alpha_cbf": args.alpha_cbf,
        "barrier_offset": args.barrier_offset,
        "lambda_actor_kl": args.lambda_actor_kl,
        "lambda_distill": args.lambda_distill,
        "start_time": start_time_iso,
        "wall_time_seconds": time.time() - script_start_time,
    }
    meta["best_checkpoint_step"] = best_step if args.total_steps >= 50_000 else "final"
    meta["best_checkpoint_score"] = float(best_score) if args.total_steps >= 50_000 else float("nan")
    try:
        meta["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        meta["git_hash"] = "unknown"
    meta_path = os.path.join(output_dir, f"meta_fusion_aux_{args.scenario}_{args.ego_maneuver}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    final_eval_return_val = float(ep_returns[-1]) if ep_returns else 0.0
    best_eval_return_val = float(max(ep_returns)) if ep_returns else 0.0
    best_iteration_val = int(np.argmax(ep_returns) + 1) if ep_returns else 0
    final_residual_opt = float(metrics.get("soft_residual_mean", 0.0)) if "metrics" in locals() else 0.0
    final_residual_saf = float(metrics.get("cbf_residual_mean", 0.0)) if "metrics" in locals() else 0.0
    final_distill_gap = float(metrics.get("distill_gap", 0.0)) if "metrics" in locals() else 0.0
    result_summary = {
        "best_eval_return": best_eval_return_val,
        "final_eval_return": final_eval_return_val,
        "best_iteration": best_iteration_val,
        "final_collision_rate": float(coll_rate) if "coll_rate" in locals() else 0.0,
        "final_success_rate": None,
        "final_timeout_rate": None,
        "best_distillation_gap": final_distill_gap,
        "final_residual_loss_optimality": final_residual_opt,
        "final_residual_loss_safety": final_residual_saf,
    }
    write_meta_end(
        output_dir=output_dir,
        total_steps_actual=int(step),
        convergence_reason="max_steps_reached",
        result_summary=result_summary,
    )

    env.close()
    eval_env.close()
    print(f"Training complete. Metrics in {csv_path}")


if __name__ == "__main__":
    main()
