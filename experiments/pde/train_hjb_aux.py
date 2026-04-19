"""Train HJB-residual auxiliary critic on SUMO T-intersection."""

from __future__ import annotations

import argparse
import os
import csv
import time
import json
import subprocess
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None


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
    parser.add_argument("--config", default="configs/pde/hjb_aux.yaml")
    parser.add_argument("--algo_config", default="configs/algo/default.yaml")
    parser.add_argument("--out_dir", default="results/pde")
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4", "2_dense", "3_dense", "4_dense"])
    parser.add_argument("--ego_maneuver", default="stem_right",
                        choices=["stem_right", "stem_left", "right_left",
                                 "right_stem", "left_right", "left_stem"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sumo_gui", action="store_true")
    parser.add_argument("--log_interval_steps", type=int, default=2000,
                        help="Log eval metrics every N training steps")
    parser.add_argument("--no_buildings", action="store_true",
                        help="Disable static occlusion buildings (full visibility)")
    parser.add_argument("--style_filter", default=None, choices=["nominal", "adversarial"],
                        help="Filter agent behavioral styles for robustness ablation")
    parser.add_argument("--state_ablation", default=None, choices=["no_visibility"],
                        help="State ablation: remove specific feature groups")
    parser.add_argument("--lambda_aux", type=float, default=None,
                        help="Override PDE residual weight (for lambda sweep ablation)")
    args = parser.parse_args()

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
    os.makedirs(args.out_dir, exist_ok=True)

    from env.sumo_env import SumoEnv
    env_kwargs = {"use_gui": args.sumo_gui, "scenario_name": args.scenario,
                  "ego_maneuver": args.ego_maneuver, "use_intent": args.use_intent,
                  "buildings": not args.no_buildings, "style_filter": args.style_filter,
                  "state_ablation": args.state_ablation}
    obs_dim = int(SumoEnv(**env_kwargs).observation_space.shape[0])
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    from models.pde.hjb_aux_agent import HJBAuxAgent
    policy = HJBAuxAgent(
        obs_dim=obs_dim,
        lr=float(algo_cfg.get("lr", 3e-4)),
        aux_lr=float(pde_cfg.get("aux_lr", 1e-3)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(algo_cfg.get("clip_range", 0.2)),
        ent_coef=float(algo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
        lambda_anchor=float(pde_cfg.get("lambda_anchor", 1.0)),
        lambda_hjb=args.lambda_aux if args.lambda_aux is not None else float(pde_cfg.get("lambda_hjb", 0.2)),
        lambda_bc=float(pde_cfg.get("lambda_bc", 0.5)),
        lambda_distill=float(pde_cfg.get("lambda_distill", 0.25)),
        aux_hidden_dim=int(pde_cfg.get("aux_hidden_dim", 256)),
        collocation_ratio=float(pde_cfg.get("collocation_ratio", 0.7)),
        hidden_dim=int(algo_cfg.get("gru_hidden", 128)),
        device=device,
    )

    n_steps = int(algo_cfg.get("n_steps", 4096))
    batch_size = int(algo_cfg.get("batch_size", 128))
    n_epochs = int(algo_cfg.get("n_epochs", 8))
    gamma = float(algo_cfg.get("gamma", 0.99))
    gae_lambda = float(algo_cfg.get("gae_lambda", 0.95))

    csv_path = os.path.join(args.out_dir, f"train_hjb_aux_{args.scenario}_{args.ego_maneuver}.csv")
    header = [
        "step", "episode_return", "episode_len", "actor_loss", "vf_loss",
        "collision_count", "collision_rate", "mean_ttc", "min_ttc", "entropy",
        "hjb_residual_mean", "anchor_loss", "bc_loss", "distill_loss", "distill_gap",
        "train_time_per_iter",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    from experiments.pde.collect_rollouts import collect_rollouts

    env = SumoEnv(**env_kwargs)
    eval_env = SumoEnv(**env_kwargs)
    EVAL_SEED_OFFSET = 10_000
    EVAL_EPISODES_PER_LOG = 5
    step = 0
    ep_returns = []
    next_log_step = 0

    while step < args.total_steps:
        t_iter_start = time.time()
        obs_arr, actions_arr, log_probs_arr, returns_arr, advantages_arr, extra, hidden_arr = \
            collect_rollouts(env, policy, n_steps, gamma, gae_lambda)
        step += n_steps

        for _ in range(n_epochs):
            perm = np.random.permutation(len(obs_arr))
            for start in range(0, len(obs_arr), batch_size):
                idx = perm[start:start + batch_size]
                if len(idx) == 0:
                    continue
                ex = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(obs_arr) else v)
                      for k, v in extra.items()} if extra else None
                h = hidden_arr[idx] if hidden_arr is not None else None
                metrics = policy.train_step(
                    obs_arr[idx], actions_arr[idx], log_probs_arr[idx],
                    returns_arr[idx], advantages_arr[idx],
                    hiddens=h, extra=ex,
                )

        train_time = time.time() - t_iter_start

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
                    metrics.get("hjb_residual_mean", 0), metrics.get("anchor_loss", 0),
                    metrics.get("bc_loss", 0), metrics.get("distill_loss", 0),
                    metrics.get("distill_gap", 0), train_time,
                ])
            print(f"[hjb_aux] step={step} ret={np.mean(ep_returns[-10:]):.2f} "
                  f"coll={coll_count} ttc={mean_ttc:.2f} hjb_res={metrics.get('hjb_residual_mean', 0):.4f}")
            next_log_step = step + args.log_interval_steps

    ckpt_path = os.path.join(args.out_dir, f"model_hjb_aux_{args.scenario}_{args.ego_maneuver}.pt")
    policy.save(ckpt_path)
    print(f"Saved hjb_aux to {ckpt_path}")

    # Provenance metadata
    meta = {
        "method": "hjb_aux",
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
        "lambda_aux_override": args.lambda_aux,
        "lambda_hjb_effective": args.lambda_aux if args.lambda_aux is not None else float(pde_cfg.get("lambda_hjb", 0.2)),
        "start_time": start_time_iso,
        "wall_time_seconds": time.time() - script_start_time,
    }
    try:
        meta["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        meta["git_hash"] = "unknown"
    meta_path = os.path.join(args.out_dir, f"meta_hjb_aux_{args.scenario}_{args.ego_maneuver}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    env.close()
    eval_env.close()
    print(f"Training complete. Metrics in {csv_path}")


if __name__ == "__main__":
    main()
