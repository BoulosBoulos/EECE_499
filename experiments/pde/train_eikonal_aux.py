"""Train Eikonal-residual auxiliary critic on SUMO T-intersection."""

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
    parser.add_argument("--config", default="configs/pde/eikonal_aux.yaml")
    parser.add_argument("--algo_config", default="configs/algo/default.yaml")
    parser.add_argument("--out_dir", default="results/pde")
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--ego_maneuver", default="stem_right",
                        choices=["stem_right", "stem_left", "right_left",
                                 "right_stem", "left_right", "left_stem"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sumo_gui", action="store_true")
    args = parser.parse_args()

    script_start_time = time.time()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    if args.seed is not None:
        np.random.seed(args.seed)
        if torch:
            torch.manual_seed(args.seed)

    pde_cfg = _load_config(args.config)
    algo_cfg = _load_config(args.algo_config)
    os.makedirs(args.out_dir, exist_ok=True)

    from env.sumo_env import SumoEnv
    env_kwargs = {"use_gui": args.sumo_gui, "scenario_name": args.scenario,
                  "ego_maneuver": args.ego_maneuver, "use_intent": args.use_intent}
    obs_dim = int(SumoEnv(**env_kwargs).observation_space.shape[0])
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    from models.pde.eikonal_aux_agent import EikonalAuxAgent
    policy = EikonalAuxAgent(
        obs_dim=obs_dim,
        lr=float(algo_cfg.get("lr", 3e-4)),
        aux_lr=float(pde_cfg.get("aux_lr", 1e-3)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(algo_cfg.get("clip_range", 0.2)),
        ent_coef=float(algo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
        lambda_anchor=float(pde_cfg.get("lambda_anchor", 1.0)),
        lambda_eik=float(pde_cfg.get("lambda_eik", 0.2)),
        lambda_bc=float(pde_cfg.get("lambda_bc", 0.5)),
        lambda_distill=float(pde_cfg.get("lambda_distill", 0.25)),
        aux_hidden_dim=int(pde_cfg.get("aux_hidden_dim", 256)),
        collocation_ratio=float(pde_cfg.get("collocation_ratio", 0.7)),
        v_min=float(pde_cfg.get("v_min", 0.5)),
        hidden_dim=int(algo_cfg.get("gru_hidden", 128)),
        device=device,
    )

    n_steps = int(algo_cfg.get("n_steps", 4096))
    batch_size = int(algo_cfg.get("batch_size", 128))
    n_epochs = int(algo_cfg.get("n_epochs", 8))
    gamma = float(algo_cfg.get("gamma", 0.99))
    gae_lambda = float(algo_cfg.get("gae_lambda", 0.95))

    csv_path = os.path.join(args.out_dir, f"train_eikonal_aux_{args.scenario}_{args.ego_maneuver}.csv")
    header = [
        "step", "episode_return", "episode_len", "actor_loss", "vf_loss",
        "collision_count", "collision_rate", "mean_ttc", "min_ttc", "entropy",
        "eikonal_residual_mean", "anchor_loss", "bc_loss", "distill_loss", "distill_gap",
        "train_time_per_iter",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    from experiments.pde.collect_rollouts import collect_rollouts

    env = SumoEnv(**env_kwargs)
    step = 0
    ep_returns = []

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

        rewards = []
        ttc_list = []
        coll_count = 0
        obs, _ = env.reset()
        policy.reset_hidden()
        for _ in range(500):
            action, _, _, _ = policy.get_action(obs)
            obs, r, term, trunc, info = env.step(action)
            rewards.append(r)
            ttc_list.append(info.get("ttc_min", 10.0))
            if info.get("collision", False):
                coll_count += 1
            if term or trunc:
                break
        ep_returns.append(sum(rewards))
        ttc_arr = np.array(ttc_list)
        mean_ttc = float(np.mean(ttc_arr)) if len(ttc_arr) > 0 else float("nan")
        min_ttc = float(np.min(ttc_arr)) if len(ttc_arr) > 0 else float("nan")
        coll_rate = coll_count / max(len(rewards), 1)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                step, np.mean(ep_returns[-10:]) if ep_returns else 0,
                len(rewards), metrics.get("actor_loss", 0), metrics.get("vf_loss", 0),
                coll_count, coll_rate, mean_ttc, min_ttc, metrics.get("entropy", 0),
                metrics.get("eikonal_residual_mean", 0), metrics.get("anchor_loss", 0),
                metrics.get("bc_loss", 0), metrics.get("distill_loss", 0),
                metrics.get("distill_gap", 0), train_time,
            ])
        if step % 5000 == 0:
            print(f"[eikonal_aux] step={step} ret={np.mean(ep_returns[-10:]):.2f} "
                  f"coll={coll_count} ttc={mean_ttc:.2f} eik_res={metrics.get('eikonal_residual_mean', 0):.4f}")

    ckpt_path = os.path.join(args.out_dir, f"model_eikonal_aux_{args.scenario}_{args.ego_maneuver}.pt")
    policy.save(ckpt_path)
    print(f"Saved eikonal_aux to {ckpt_path}")

    # Provenance metadata
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
        "start_time": start_time_iso,
        "wall_time_seconds": time.time() - script_start_time,
    }
    try:
        meta["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        meta["git_hash"] = "unknown"
    meta_path = os.path.join(args.out_dir, f"meta_eikonal_aux_{args.scenario}_{args.ego_maneuver}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    env.close()
    print(f"Training complete. Metrics in {csv_path}")


if __name__ == "__main__":
    main()
