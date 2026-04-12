"""Ablation runner for PDE-family methods.

Trains and evaluates hjb_aux and soft_hjb_aux across scenarios, seeds, and lambda sweeps.
Writes to results/pde_ablation/ -- does NOT touch results/ablation/.
"""

from __future__ import annotations

import argparse
import csv
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
PDE_VARIANTS = ["hjb_aux", "soft_hjb_aux"]
DEFAULT_SEEDS = [42, 123, 456, 789, 999]

EVAL_FIELDS = [
    "scenario", "variant", "lambda_hjb", "seed", "eval_mode",
    "mean_return", "std_return", "collision_rate", "pothole_hits_mean",
    "mean_ttc", "min_ttc",
]

TRAIN_LOG_FIELDS = [
    "scenario", "variant", "lambda_hjb", "seed", "step",
    "actor_loss", "vf_loss", "entropy", "total_loss",
    "hjb_residual_mean", "soft_residual_mean", "anchor_loss",
    "bc_loss", "distill_loss", "distill_gap", "actor_align_kl",
]


def _load_config(path: str) -> dict:
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _append_csv(path: str, row: dict, fieldnames: list[str]):
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def train_one(env, scenario, variant, total_steps, out_dir, device, seed,
              lambda_hjb=0.2, train_csv=None):
    from experiments.pde.collect_rollouts import collect_rollouts

    np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)

    algo_cfg = _load_config("configs/algo/default.yaml")
    obs_dim = int(env.observation_space.shape[0])

    common_kwargs = dict(
        obs_dim=obs_dim,
        lr=float(algo_cfg.get("lr", 3e-4)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(algo_cfg.get("clip_range", 0.2)),
        hidden_dim=int(algo_cfg.get("gru_hidden", 128)),
        device=device,
    )

    if variant == "hjb_aux":
        from models.pde.hjb_aux_agent import HJBAuxAgent
        policy = HJBAuxAgent(lambda_hjb=lambda_hjb, **common_kwargs)
    else:
        from models.pde.soft_hjb_aux_agent import SoftHJBAuxAgent
        policy = SoftHJBAuxAgent(lambda_soft=lambda_hjb, **common_kwargs)

    n_steps = int(algo_cfg.get("n_steps", 256))
    batch_size = int(algo_cfg.get("batch_size", 64))
    n_epochs = int(algo_cfg.get("n_epochs", 5))
    gamma = float(algo_cfg.get("gamma", 0.99))
    gae_lam = float(algo_cfg.get("gae_lambda", 0.95))

    step = 0
    while step < total_steps:
        obs_arr, actions_arr, log_probs_arr, returns_arr, adv_arr, extra, hidden_arr = \
            collect_rollouts(env, policy, n_steps, gamma, gae_lam)
        step += n_steps
        metrics = {}
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
                    returns_arr[idx], adv_arr[idx], hiddens=h, extra=ex)

        if train_csv and metrics:
            row = {"scenario": scenario, "variant": variant,
                   "lambda_hjb": lambda_hjb, "seed": seed, "step": step}
            row.update(metrics)
            _append_csv(train_csv, row, TRAIN_LOG_FIELDS)

    ckpt = os.path.join(out_dir, f"ablation_{scenario}_{variant}_lh{lambda_hjb}_s{seed}.pt")
    policy.save(ckpt)
    return ckpt


def eval_one(env, checkpoint, variant, n_episodes, device, seed, deterministic=True):
    obs_dim = int(env.observation_space.shape[0])
    if variant == "hjb_aux":
        from models.pde.hjb_aux_agent import HJBAuxAgent
        policy = HJBAuxAgent(obs_dim=obs_dim, device=device)
    else:
        from models.pde.soft_hjb_aux_agent import SoftHJBAuxAgent
        policy = SoftHJBAuxAgent(obs_dim=obs_dim, device=device)
    policy.load(checkpoint)

    returns, coll_eps, pothole_hits, ttc_means, ttc_mins = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy.reset_hidden()
        r_tot, coll, pot = 0, 0, 0
        ttc_list = []
        for _ in range(500):
            a, _, _, _ = policy.get_action(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            r_tot += r
            if info.get("collision", False):
                coll += 1
            if info.get("in_pothole", False):
                pot += 1
            ttc_list.append(info.get("ttc_min", 10.0))
            if term or trunc:
                break
        returns.append(r_tot)
        coll_eps.append(1 if coll > 0 else 0)
        pothole_hits.append(pot)
        ttc_arr = np.array(ttc_list)
        ttc_means.append(float(np.mean(ttc_arr)))
        ttc_mins.append(float(np.min(ttc_arr)))

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "collision_rate": float(np.mean(coll_eps)),
        "pothole_hits_mean": float(np.mean(pothole_hits)),
        "mean_ttc": float(np.mean(ttc_means)),
        "min_ttc": float(np.mean(ttc_mins)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/pde_ablation")
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--variants", nargs="+", default=PDE_VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--lambda_hjb", type=float, nargs="+", default=[0.2])
    parser.add_argument("--use_intent", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    from env.sumo_env import SumoEnv
    results_csv = os.path.join(args.out_dir, "ablation_results.csv")
    train_csv = os.path.join(args.out_dir, "ablation_train_log.csv")

    for scenario in args.scenarios:
        env = SumoEnv(scenario_name=scenario, use_intent=args.use_intent)
        for variant in args.variants:
            for lh in args.lambda_hjb:
                for seed in args.seeds:
                    run_tag = f"{scenario}_{variant}_lh{lh}_s{seed}"
                    print(f"Training {run_tag}...")
                    ckpt = train_one(env, scenario, variant, args.total_steps,
                                     args.out_dir, device, seed, lambda_hjb=lh,
                                     train_csv=train_csv)

                    if os.path.isfile(ckpt):
                        for mode, det in [("deterministic", True), ("stochastic", False)]:
                            print(f"Eval {run_tag} [{mode}]...")
                            m = eval_one(env, ckpt, variant, args.eval_episodes,
                                         device, seed, deterministic=det)
                            row = {
                                "scenario": scenario, "variant": variant,
                                "lambda_hjb": lh, "seed": seed, "eval_mode": mode, **m,
                            }
                            _append_csv(results_csv, row, EVAL_FIELDS)
        env.close()

    print(f"\nPDE ablation done. Results: {results_csv}")
    print(f"Training log: {train_csv}")


if __name__ == "__main__":
    main()
