"""Evaluate trained models. See docs/RUNNING.md."""

from __future__ import annotations

import argparse
import os
import csv

try:
    import numpy as np
except ImportError:
    np = None

from env.sumo_env import SumoEnv
from models.drppo import DRPPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/model_pinn_1a.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="results/eval_metrics.csv")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui during eval (slower)")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"],
                        help="SUMO scenario to evaluate on")
    parser.add_argument("--use_intent", action="store_true", help="Use intent LSTM (must match training)")
    args = parser.parse_args()

    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    env = SumoEnv(use_gui=args.gui, scenario_name=args.scenario, use_intent=args.use_intent)
    obs_dim = int(env.observation_space.shape[0])
    policy = DRPPO(
        obs_dim=obs_dim,
        n_actions=5,
        use_pinn=False,
        device=device,
    )
    policy.load(args.checkpoint)

    returns = []
    lengths = []
    collision_steps = []
    collision_episodes = []
    mean_ttc_per_ep = []
    min_ttc_per_ep = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_r = 0
        coll = 0
        steps = 0
        ttc_list = []
        for _ in range(500):
            action, _, _, _ = policy.get_action(obs, deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            steps += 1
            ttc_list.append(info.get("ttc_min", 10.0))
            if r < -5:
                coll += 1
            if term or trunc:
                break
        returns.append(total_r)
        lengths.append(steps)
        collision_steps.append(coll)
        collision_episodes.append(1 if coll > 0 else 0)
        ttc_arr = np.array(ttc_list)
        mean_ttc_per_ep.append(float(np.mean(ttc_arr)) if len(ttc_arr) > 0 else float("nan"))
        min_ttc_per_ep.append(float(np.min(ttc_arr)) if len(ttc_arr) > 0 else float("nan"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "return", "length", "collision_steps", "collision_episode",
            "mean_ttc", "min_ttc",
        ])
        for i in range(args.episodes):
            writer.writerow([
                i, returns[i], lengths[i], collision_steps[i], collision_episodes[i],
                mean_ttc_per_ep[i], min_ttc_per_ep[i],
            ])

    n_coll_ep = sum(collision_episodes)
    total_coll_steps = sum(collision_steps)
    print(f"\n=== Eval Summary ===")
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Collision episodes: {n_coll_ep}/{args.episodes} ({100*n_coll_ep/args.episodes:.1f}%)")
    print(f"Total collision steps: {total_coll_steps}")
    print(f"Mean TTC per episode: {np.nanmean(mean_ttc_per_ep):.2f}s")
    print(f"Mean min TTC per episode: {np.nanmean(min_ttc_per_ep):.2f}s")
    print(f"Metrics saved to {args.out}")
    env.close()


if __name__ == "__main__":
    main()
