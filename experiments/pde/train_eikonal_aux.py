"""Train Eikonal-residual auxiliary critic on SUMO T-intersection."""

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

METHOD_NAME = "eikonal_aux"
_METHOD_CFG = get_method_config(METHOD_NAME)


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
    parser.add_argument("--config", default="configs/pde/eikonal_aux.yaml",
                        help="Yaml with Phase 3F-A hyperparameters; CLI flags below override yaml values")
    # Phase 3F-A four-term loss weights (CLI overrides; defaults read from yaml).
    parser.add_argument("--w_eik", type=float, default=None,
                        help="Phase 3F-A: weight for Eikonal PDE residual L_eik (default: yaml)")
    parser.add_argument("--w_bc", type=float, default=None,
                        help="Phase 3F-A: weight for boundary-condition loss L_bc (default: yaml)")
    parser.add_argument("--w_ground", type=float, default=None,
                        help="Phase 3F-A: weight for observed-time grounding loss L_ground (default: yaml)")
    parser.add_argument("--w_distill", type=float, default=None,
                        help="Phase 3F-A: weight for soft-KL policy distillation L_distill (default: yaml)")
    parser.add_argument("--beta_KL", type=float, default=None,
                        help="Phase 3F-A: KL distillation strength (default: yaml)")
    parser.add_argument("--tau_distill", type=float, default=None,
                        help="Phase 3F-A: softmax temperature for A_eik (default: yaml)")
    parser.add_argument("--T_max", type=float, default=None,
                        help="Phase 3F-A: finite-horizon collision boundary value (default: yaml)")
    parser.add_argument("--collocation_size", type=int,
                        default=int(_METHOD_CFG["collocation_size"]),
                        help="Collocation samples per iteration")
    # Phase 3F-A Step 7C: Kendall et al. uncertainty weighting + replay buffers.
    parser.add_argument("--log_sigma_init", type=float, default=None,
                        help="Phase 3F-A Step 7C: initial value for log σ params (default: yaml)")
    parser.add_argument("--replay_buffer_size", type=int, default=None,
                        help="Phase 3F-A Step 7C: capacity per terminal/intermediate ring buffer (default: yaml)")
    parser.add_argument("--replay_batch_size", type=int, default=None,
                        help="Phase 3F-A Step 7C: minibatch size sampled from replay buffer (default: yaml)")
    # Phase 3F-A Step 7D: Augmented Lagrangian for L_eik constraint.
    parser.add_argument("--alm_lambda_init", type=float, default=None,
                        help="Phase 3F-A Step 7D: initial Lagrange multiplier λ (default: yaml or 0.0)")
    parser.add_argument("--alm_mu_init", type=float, default=None,
                        help="Phase 3F-A Step 7D: initial penalty μ (default: yaml or 1.0)")
    parser.add_argument("--alm_mu_max", type=float, default=None,
                        help="Phase 3F-A Step 7D: max μ before clamp (default: yaml or 1e4)")
    parser.add_argument("--alm_alpha", type=float, default=None,
                        help="Phase 3F-A Step 7D: required improvement ratio per ALM epoch (default: yaml or 0.25)")
    parser.add_argument("--alm_beta", type=float, default=None,
                        help="Phase 3F-A Step 7D: μ growth factor (default: yaml or 5.0)")
    parser.add_argument("--alm_update_interval", type=int, default=None,
                        help="Phase 3F-A Step 7D: training steps between λ, μ updates (default: yaml or 5000)")
    # Backwards-compatibility aliases (Phase 3 prior); silently ignored under
    # Phase 3F-A since the legacy semantics no longer apply. Kept so older
    # orchestrator scripts that still pass --lambda_residual/--lambda_distill
    # do not crash; Phase 3F-A reads its weights from yaml/CLI flags above.
    parser.add_argument("--lambda_residual", type=float, default=None,
                        help="DEPRECATED (Phase 3F-A): use --w_eik instead.")
    parser.add_argument("--lambda_distill", type=float, default=None,
                        help="DEPRECATED (Phase 3F-A): use --w_distill instead.")
    parser.add_argument("--w_fail", type=float, default=None,
                        help="DEPRECATED (Phase 3F-A): collision BC value is now T_max.")
    parser.add_argument("--lambda_aux", type=float, default=None,
                        help="DEPRECATED (Phase 3F-A): use --w_eik instead.")
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

    from models.pde.eikonal_aux_agent import EikonalAuxAgent
    # Phase 3F-A hyperparameters: read from yaml, override with CLI if present.
    def _pick(cli_val, yaml_key, default):
        if cli_val is not None:
            return float(cli_val)
        return float(pde_cfg.get(yaml_key, default))
    policy = EikonalAuxAgent(
        obs_dim=obs_dim,
        lr=float(args.lr),
        aux_lr=float(pde_cfg.get("aux_lr", 1e-3)),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_eps),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        # Phase 3F-A four-term loss weights (legacy fixed weights, kept for
        # provenance only — Step 7C replaces critic-side weights with σ params):
        w_eik=_pick(args.w_eik, "w_eik", 1.0),
        w_bc=_pick(args.w_bc, "w_bc", 0.5),
        w_ground=_pick(args.w_ground, "w_ground", 1.0),
        w_distill=_pick(args.w_distill, "w_distill", 0.5),
        beta_KL=_pick(args.beta_KL, "beta_KL", 0.1),
        tau=_pick(args.tau_distill, "tau", 1.0),
        T_max=_pick(args.T_max, "T_max", 100.0),
        # Phase 3F-A Step 7C: Kendall uncertainty weighting (BC, GROUND only in 7D)
        # + replay buffers.
        log_sigma_init=_pick(args.log_sigma_init, "log_sigma_init", 0.0),
        replay_buffer_size=int(args.replay_buffer_size if args.replay_buffer_size is not None
                               else pde_cfg.get("replay_buffer_size", 1000)),
        replay_batch_size=int(args.replay_batch_size if args.replay_batch_size is not None
                              else pde_cfg.get("replay_batch_size", 256)),
        # Phase 3F-A Step 7D: Augmented Lagrangian for L_eik.
        alm_lambda_init=_pick(args.alm_lambda_init, "alm_lambda_init", 0.0),
        alm_mu_init=_pick(args.alm_mu_init, "alm_mu_init", 1.0),
        alm_mu_max=_pick(args.alm_mu_max, "alm_mu_max", 1.0e4),
        alm_alpha=_pick(args.alm_alpha, "alm_alpha", 0.25),
        alm_beta=_pick(args.alm_beta, "alm_beta", 5.0),
        alm_update_interval=int(args.alm_update_interval if args.alm_update_interval is not None
                                 else pde_cfg.get("alm_update_interval", 5000)),
        # Existing hyperparameters retained:
        aux_hidden_dim=int(pde_cfg.get("aux_hidden_dim", 256)),
        collocation_ratio=float(pde_cfg.get("collocation_ratio", 0.7)),
        v_min=float(pde_cfg.get("v_min", 0.5)),
        hidden_dim=int(args.gru_hidden_size),
        device=device,
    )

    n_steps = int(args.n_steps)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs_per_update)
    gamma = float(args.gamma)
    gae_lambda = float(args.gae_lambda)

    csv_path = os.path.join(output_dir, f"train_eikonal_aux_{args.scenario}_{args.ego_maneuver}.csv")
    # Phase 3F-A Step 7D header: hybrid ALM/KGC diagnostics. sigma_eik retained
    # at 0 for back-compat readers; ALM scalars λ, μ added.
    header = [
        "step", "episode_return", "episode_len", "actor_loss", "vf_loss",
        "collision_count", "collision_rate", "mean_ttc", "min_ttc", "entropy",
        "L_eik",
        "L_bc_raw", "L_bc_normalized",
        "L_ground_raw", "L_ground_normalized",
        "L_distill",
        "mean_rho", "mean_T_succ", "mean_T_coll", "n_grounding_points",
        # Step 7C diagnostics (KGC for BC/GROUND retained; sigma_eik=0 in 7D)
        "sigma_eik", "sigma_bc", "sigma_ground",
        # Step 7D diagnostics (ALM for L_eik)
        "alm_lambda", "alm_mu", "L_eik_alm_contribution",
        "n_success_buffer", "n_collision_buffer", "n_intermediate_buffer",
        "train_time_per_iter",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(METRICS_COLUMNS)
    # Phase 3F-A: build_uniform_config's schema is shared across methods and
    # only accepts the legacy kwarg set. Map the relevant new weights onto
    # the closest legacy slots so the uniform-schema test 24.2 still passes.
    # The full Phase 3F-A weight set is also recorded inside meta.json under
    # the dedicated `phase_3F_A_loss_weights` block below.
    config_dict = build_uniform_config(
        args, method=METHOD_NAME,
        lambda_residual=policy.w_eik,    # Phase 3F-A: w_eik replaces lambda_residual
        lambda_distill=policy.w_distill, # Phase 3F-A: w_distill replaces lambda_distill
        w_fail=policy.T_max,             # Phase 3F-A: collision BC anchor is now T_max
        collocation_size=args.collocation_size,
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

        # Phase 3F-A Step 7D: Augmented Lagrangian update on the L_eik constraint.
        # Triggered every ``alm_update_interval`` training steps; canonical
        # Bertsekas 1996 §4.2 update rule. See math derivation §10.2.
        if (
            policy.alm_update_interval > 0
            and step >= policy.alm_last_update_step + policy.alm_update_interval
        ):
            alm_record = policy.update_alm_parameters(current_step=step)
            print(
                f"[eikonal_aux ALM] step={step}  "
                f"λ={alm_record.get('alm_lambda'):.4g}  "
                f"μ={alm_record.get('alm_mu'):.4g}  "
                f"mean L_eik (epoch)={alm_record.get('mean_L_eik_epoch')}  "
                f"μ_grew={alm_record.get('mu_grew', False)}  "
                f"μ@max={alm_record.get('mu_at_max', False)}"
            )

        action_counts = extra.get("action_counts", [0, 0, 0, 0, 0])
        total_actions = float(sum(action_counts))
        action_dist = (
            [c / total_actions for c in action_counts] if total_actions > 0 else [0.0] * 5
        )
        n_eps = int(extra.get("n_episodes", 0))
        mean_reward_iter = float(np.sum(returns_arr) / max(n_eps, 1)) if n_eps > 0 else float(np.mean(returns_arr))
        mean_ep_len = float(extra.get("mean_episode_length", 0.0))
        L_residual_safety = float(metrics.get("eikonal_residual_mean", 0.0))
        L_distill_iter = float(metrics.get("distill_loss", 0.0))
        L_total_iter = (
            float(metrics.get("actor_loss", 0.0))
            + float(metrics.get("vf_loss", 0.0))
            + L_residual_safety
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
                0.0, L_residual_safety, L_distill_iter,
                mean_reward_iter, mean_ep_len, n_eps,
                int(extra.get("n_collisions", 0)),
                int(extra.get("n_successes", 0)),
                int(extra.get("n_timeouts", 0)),
                int(extra.get("n_aborts", 0)),
                action_dist[0], action_dist[1], action_dist[2], action_dist[3], action_dist[4],
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
                f"model_eikonal_aux_{args.scenario}_{args.ego_maneuver}_step{ck_step}.pt")
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
                # Phase 3F-A Step 7D: include ALM scalars (λ, μ, L_eik_alm).
                csv.writer(f).writerow([
                    step, np.mean(ep_returns[-10:]) if ep_returns else 0,
                    len(eval_returns), metrics.get("actor_loss", 0), metrics.get("vf_loss", 0),
                    coll_count, coll_rate, mean_ttc, min_ttc, metrics.get("entropy", 0),
                    metrics.get("L_eik", 0),
                    metrics.get("L_bc_raw", 0), metrics.get("L_bc_normalized", 0),
                    metrics.get("L_ground_raw", 0), metrics.get("L_ground_normalized", 0),
                    metrics.get("L_distill", 0),
                    metrics.get("mean_rho", 0),
                    metrics.get("mean_T_succ", 0), metrics.get("mean_T_coll", 0),
                    metrics.get("n_grounding_points", 0),
                    metrics.get("sigma_eik", 0.0), metrics.get("sigma_bc", 1.0), metrics.get("sigma_ground", 1.0),
                    metrics.get("alm_lambda", 0.0), metrics.get("alm_mu", 1.0),
                    metrics.get("L_eik_alm_contribution", 0.0),
                    metrics.get("n_success_buffer", 0),
                    metrics.get("n_collision_buffer", 0),
                    metrics.get("n_intermediate_buffer", 0),
                    train_time,
                ])
            print(
                f"[eikonal_aux] step={step} ret={np.mean(ep_returns[-10:]):.2f} "
                f"coll={coll_count} ttc={mean_ttc:.2f} "
                f"L_eik={metrics.get('L_eik', 0):.3g} "
                f"L_bc={metrics.get('L_bc_raw', 0):.3g} L_ground={metrics.get('L_ground_raw', 0):.3g} "
                f"|rho|={metrics.get('mean_rho', 0):.3f} "
                f"ALM[λ={metrics.get('alm_lambda',0):.3g},μ={metrics.get('alm_mu',1):.3g}] "
                f"σ_bc={metrics.get('sigma_bc',1):.2f} σ_g={metrics.get('sigma_ground',1):.2f} "
                f"buf=[{metrics.get('n_success_buffer',0)}/{metrics.get('n_collision_buffer',0)}/{metrics.get('n_intermediate_buffer',0)}] "
                f"T_succ~{metrics.get('mean_T_succ', 0):.1f} T_coll~{metrics.get('mean_T_coll', 0):.1f}"
            )
            next_log_step = step + log_interval_steps

    ckpt_path = os.path.join(output_dir, f"model_eikonal_aux_{args.scenario}_{args.ego_maneuver}.pt")
    policy.save(ckpt_path)
    print(f"Saved eikonal_aux to {ckpt_path}")

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

    # Provenance metadata (Phase 3F-A: include four-term loss weights).
    meta = {
        "method": "eikonal_aux",
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
        "phase_3F_A_loss_weights": {
            "w_eik": policy.w_eik,
            "w_bc": policy.w_bc,
            "w_ground": policy.w_ground,
            "w_distill": policy.w_distill,
            "beta_KL": policy.beta_KL,
            "tau": policy.tau,
            "T_max": policy.T_max,
        },
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
    meta_path = os.path.join(output_dir, f"meta_eikonal_aux_{args.scenario}_{args.ego_maneuver}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    final_eval_return_val = float(ep_returns[-1]) if ep_returns else 0.0
    best_eval_return_val = float(max(ep_returns)) if ep_returns else 0.0
    best_iteration_val = int(np.argmax(ep_returns) + 1) if ep_returns else 0
    # Phase 3F-A: read residual from new metric keys; legacy aliases also
    # populated by the agent so older orchestrator scripts keep working.
    final_residual_safety = float(metrics.get("mean_rho", metrics.get("eikonal_residual_mean", 0.0))) if "metrics" in locals() else 0.0
    final_distill_gap = float(metrics.get("L_distill", metrics.get("distill_gap", 0.0))) if "metrics" in locals() else 0.0
    result_summary = {
        "best_eval_return": best_eval_return_val,
        "final_eval_return": final_eval_return_val,
        "best_iteration": best_iteration_val,
        "final_collision_rate": float(coll_rate) if "coll_rate" in locals() else 0.0,
        "final_success_rate": None,
        "final_timeout_rate": None,
        "best_distillation_gap": final_distill_gap,
        "final_residual_loss_optimality": 0.0,
        "final_residual_loss_safety": final_residual_safety,
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
