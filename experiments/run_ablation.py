"""
Ablation study: train and eval across scenarios (1a-1d, 2-4) and variants.
Variants: nopinn, pinn (Design A only), pinn_ego (Design A + L_ego dynamics).

NOTE: Ablation results can depend on hyperparameters (reward, lr, physics weights).
See docs/ABLATION_HYPERPARAMETERS.md. Use --reward_config to override reward params.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

SCENARIOS = ["1a", "1b", "1c", "1d", "2", "3", "4"]
# Variants: (name, use_pinn, use_l_ego)
VARIANTS = [
    ("nopinn", False, False),
    ("pinn", True, False),
    ("pinn_ego", True, True),
]


def _load_config(path: str) -> dict:
    if yaml is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def collect_rollouts(env, policy, n_steps: int, gamma: float, gae_lambda: float):
    from experiments.run_train import collect_rollouts as _c
    return _c(env, policy, n_steps, gamma, gae_lambda)


def train_one(env, scenario: str, variant_name: str, use_pinn: bool, use_l_ego: bool,
              total_steps: int, out_dir: str, device: str) -> str:
    from models.drppo import DRPPO

    cfg = _load_config("configs/algo/default.yaml")
    res_cfg = _load_config("configs/residuals/default.yaml")
    obs_dim = int(env.observation_space.shape[0])
    policy = DRPPO(
        obs_dim=obs_dim,
        n_actions=5,
        lr=float(cfg.get("lr", 3e-4)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        use_pinn=use_pinn,
        use_l_ego=use_l_ego,
        lambda_physics_critic=float(res_cfg.get("lambda_physics_critic", 0.5)),
        lambda_physics_ttc=float(res_cfg.get("lambda_physics_ttc", 1.0)),
        lambda_physics_stop=float(res_cfg.get("lambda_physics_stop", 1.0)),
        lambda_physics_fric=float(res_cfg.get("lambda_physics_fric", 1.0)),
        lambda_physics_ego=float(res_cfg.get("lambda_physics_ego", 0.1)),
        physics_ttc_thr=float(res_cfg.get("physics_ttc_thr", 3.0)),
        physics_tau=float(res_cfg.get("physics_tau", 0.5)),
        a_max=float(res_cfg.get("a_max", 5.0)),
        mu=float(res_cfg.get("mu", 0.8)),
        g=float(res_cfg.get("g", 9.81)),
        device=device,
    )
    n_steps = int(cfg.get("n_steps", 256))
    batch_size = int(cfg.get("batch_size", 64))
    n_epochs = int(cfg.get("n_epochs", 5))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    step = 0
    while step < total_steps:
        obs_arr, actions_arr, log_probs_arr, returns_arr, advantages_arr, extra = collect_rollouts(
            env, policy, n_steps, gamma, gae_lambda
        )
        step += n_steps
        for _ in range(n_epochs):
            perm = np.random.permutation(len(obs_arr))
            for start in range(0, len(obs_arr), batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) == 0:
                    continue
                o, a = obs_arr[idx], actions_arr[idx]
                lp, ret, adv = log_probs_arr[idx], returns_arr[idx], advantages_arr[idx]
                ex = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(obs_arr) else v)
                     for k, v in extra.items()} if extra else None
                policy.train_step(o, a, lp, ret, adv, ex)

    ckpt = os.path.join(out_dir, f"ablation_{scenario}_{variant_name}.pt")
    policy.save(ckpt)
    return ckpt


def eval_one(env, checkpoint: str, n_episodes: int, device: str) -> dict:
    from models.drppo import DRPPO

    obs_dim = int(env.observation_space.shape[0])
    policy = DRPPO(obs_dim=obs_dim, n_actions=5, use_pinn=False, device=device)
    policy.load(checkpoint)

    returns, coll_eps, pothole_hits = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        r_tot, coll = 0, 0
        pot = 0
        for _ in range(500):
            a, _, _, _ = policy.get_action(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            r_tot += r
            if r < -5:
                coll += 1
            if info.get("in_pothole", False):
                pot += 1
            if term or trunc:
                break
        returns.append(r_tot)
        coll_eps.append(1 if coll > 0 else 0)
        pothole_hits.append(pot)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "collision_rate": float(np.mean(coll_eps)),
        "pothole_hits_mean": float(np.mean(pothole_hits)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/ablation")
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--skip_train", action="store_true", help="Only eval existing checkpoints")
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--reward_config", default=None, help="Override reward config (affects which variant may win)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    from env.sumo_env import SumoEnv

    results = []
    for scenario in args.scenarios:
        env = SumoEnv(scenario_name=scenario, use_intent=args.use_intent, reward_config=args.reward_config)
        for variant_name, use_pinn, use_l_ego in VARIANTS:
            ckpt = os.path.join(args.out_dir, f"ablation_{scenario}_{variant_name}.pt")
            if not args.skip_train:
                print(f"Training {scenario} {variant_name}...")
                ckpt = train_one(env, scenario, variant_name, use_pinn, use_l_ego,
                                 args.total_steps, args.out_dir, device)
            if os.path.isfile(ckpt):
                print(f"Eval {scenario} {variant_name}...")
                m = eval_one(env, ckpt, args.eval_episodes, device)
                results.append({
                    "scenario": scenario,
                    "variant": variant_name,
                    "use_pinn": use_pinn,
                    "use_l_ego": use_l_ego,
                    **m,
                })
        env.close()

    csv_path = os.path.join(args.out_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "variant", "use_pinn", "use_l_ego", "mean_return", "std_return", "collision_rate", "pothole_hits_mean"])
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(args.out_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation done. Results: {csv_path}")


if __name__ == "__main__":
    main()
