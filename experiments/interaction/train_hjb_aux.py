"""Train HJB-residual auxiliary critic on the interaction benchmark."""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

from env.sumo_env_interaction import InteractionEnv, N_ACTIONS
from experiments.interaction.metrics_util import episode_event_rates
from experiments.interaction.run_train import collect_rollouts


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
    parser.add_argument("--out_dir", default="results/interaction")
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--scenario", default="1a",
                        choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        if torch:
            torch.manual_seed(args.seed)

    pde_cfg = _load_config(args.config)
    algo_cfg = _load_config(args.algo_config)
    os.makedirs(args.out_dir, exist_ok=True)

    env = InteractionEnv(scenario_name=args.scenario, use_gui=args.gui, seed=args.seed)
    obs_dim = env._state_dim
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
        lambda_hjb=float(pde_cfg.get("lambda_hjb", 0.2)),
        lambda_bc=float(pde_cfg.get("lambda_bc", 0.5)),
        lambda_distill=float(pde_cfg.get("lambda_distill", 0.25)),
        aux_hidden_dim=int(pde_cfg.get("aux_hidden_dim", 256)),
        collocation_ratio=float(pde_cfg.get("collocation_ratio", 0.7)),
        hidden_dim=int(algo_cfg.get("gru_hidden", 128)),
        device=device,
    )

    n_steps = int(algo_cfg.get("n_steps", 2048))
    batch_size = int(algo_cfg.get("batch_size", 128))
    n_epochs = int(algo_cfg.get("n_epochs", 8))
    gamma = float(algo_cfg.get("gamma", 0.99))
    gae_lambda = float(algo_cfg.get("gae_lambda", 0.95))

    csv_path = os.path.join(args.out_dir, f"train_interaction_hjb_aux_{args.scenario}.csv")
    header = [
        "step", "mean_return", "success_rate", "collision_rate", "row_violation_rate",
        "actor_loss", "vf_loss", "entropy",
        "hjb_residual_mean", "anchor_loss", "distill_loss",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    step = 0
    while step < args.total_steps:
        rollout = collect_rollouts(env, policy, n_steps, gamma, gae_lambda)
        obs_arr, act_arr, lp_arr, returns, advantages, extra, hid_arr, infos = rollout
        step += len(obs_arr)

        metrics = {}
        for _ in range(n_epochs):
            perm = np.random.permutation(len(obs_arr))
            for start in range(0, len(obs_arr), batch_size):
                idx = perm[start:start + batch_size]
                if len(idx) == 0:
                    continue
                ex = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(obs_arr) else v)
                      for k, v in extra.items()} if extra else None
                metrics = policy.train_step(
                    obs_arr[idx], act_arr[idx], lp_arr[idx],
                    returns[idx], advantages[idx],
                    hiddens=hid_arr[idx], extra=ex,
                )

        ep_returns = []
        n_eps = 0
        cur_ret = 0.0
        for i, info in enumerate(infos):
            cur_ret += info.get("reward", 0)
            done = (i < len(infos) - 1 and infos[i+1].get("step", 1) <= info.get("step", 0))
            if done or i == len(infos) - 1:
                ep_returns.append(cur_ret)
                n_eps += 1
                cur_ret = 0.0

        n_eps = max(n_eps, 1)
        evr, _ = episode_event_rates(infos)
        mean_ret = np.mean(ep_returns) if ep_returns else 0

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                step, f"{mean_ret:.2f}",
                f"{evr['success']:.3f}", f"{evr['collision']:.3f}",
                f"{evr['row_violation']:.3f}",
                f"{metrics.get('actor_loss', 0):.4f}",
                f"{metrics.get('vf_loss', 0):.4f}",
                f"{metrics.get('entropy', 0):.4f}",
                f"{metrics.get('hjb_residual_mean', 0):.4f}",
                f"{metrics.get('anchor_loss', 0):.4f}",
                f"{metrics.get('distill_loss', 0):.4f}",
            ])
        print(f"[HJB-aux/interaction] step={step} ret={mean_ret:.1f} "
              f"succ={evr['success']:.2f} coll={evr['collision']:.2f} "
              f"row={evr['row_violation']:.2f}")

    ckpt = os.path.join(args.out_dir, f"model_interaction_hjb_aux_{args.scenario}.pt")
    policy.save(ckpt)
    env.close()
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
