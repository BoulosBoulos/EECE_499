"""Evaluate trained models. See docs/RUNNING.md."""

from __future__ import annotations

import argparse
import os
import csv

try:
    import numpy as np
except ImportError:
    np = None

from env.t_intersection_env import TIntersectionEnv
from models.drppo import DRPPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/model_pinn.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="results/eval_metrics.csv")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    env = TIntersectionEnv()
    policy = DRPPO(
        obs_dim=134,
        n_actions=5,
        use_pinn=False,
        device=device,
    )
    policy.load(args.checkpoint)

    returns = []
    lengths = []
    collisions = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_r = 0
        coll = 0
        steps = 0
        for _ in range(500):
            action, _, _, _ = policy.get_action(obs, deterministic=args.deterministic)
            obs, r, _, done, info = env.step(action)
            total_r += r
            steps += 1
            if r < -5:
                coll += 1
            if done:
                break
        returns.append(total_r)
        lengths.append(steps)
        collisions.append(coll)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length", "collisions"])
        for i, (r, l, c) in enumerate(zip(returns, lengths, collisions)):
            writer.writerow([i, r, l, c])

    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Collision episodes: {sum(1 for c in collisions if c > 0)}")
    print(f"Metrics saved to {args.out}")


if __name__ == "__main__":
    main()
