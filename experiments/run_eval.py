"""Evaluate trained models with proper collision detection and multi-seed support.
See docs/RUNNING.md and docs/FRAMEWORK.md."""

from __future__ import annotations

import argparse
import os
import csv
import json

try:
    import numpy as np
except ImportError:
    np = None

from env.sumo_env import SumoEnv
from models.drppo import DRPPO


def eval_single_seed(env, policy, n_episodes: int, deterministic: bool, seed: int) -> dict:
    """Run n_episodes with a fixed seed. Returns per-episode metrics."""
    returns, lengths, collision_eps, collision_counts = [], [], [], []
    mean_ttc_per_ep, min_ttc_per_ep = [], []
    pothole_hits = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy.reset_hidden()
        total_r, steps, coll_count, pot = 0, 0, 0, 0
        ttc_list = []
        for _ in range(500):
            action, _, _, _ = policy.get_action(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            steps += 1
            ttc_list.append(info.get("ttc_min", 10.0))
            if info.get("collision", False):
                coll_count += 1
            if info.get("in_pothole", False):
                pot += 1
            if term or trunc:
                break
        returns.append(total_r)
        lengths.append(steps)
        collision_eps.append(1 if coll_count > 0 else 0)
        collision_counts.append(coll_count)
        pothole_hits.append(pot)
        ttc_arr = np.array(ttc_list)
        mean_ttc_per_ep.append(float(np.mean(ttc_arr)) if len(ttc_arr) > 0 else float("nan"))
        min_ttc_per_ep.append(float(np.min(ttc_arr)) if len(ttc_arr) > 0 else float("nan"))

    return {
        "returns": returns, "lengths": lengths,
        "collision_episodes": collision_eps,
        "collision_counts": collision_counts,
        "mean_ttc": mean_ttc_per_ep,
        "min_ttc": min_ttc_per_ep,
        "pothole_hits": pothole_hits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/model_pinn_1a.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="results/eval_metrics.csv")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy action selection (argmax). Default: stochastic.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Run both deterministic AND stochastic eval for comparison.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--pinn_placement", default="none",
                        choices=["critic", "actor", "both", "none"],
                        help="PINN placement used during training (needed for safety filter)")
    parser.add_argument("--use_safety_filter", action="store_true",
                        help="Enable safety filter (override to STOP on physics violations)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds for multi-seed eval (e.g. --seeds 42 123 456)")
    args = parser.parse_args()

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    env = SumoEnv(use_gui=args.gui, scenario_name=args.scenario, use_intent=args.use_intent)
    obs_dim = int(env.observation_space.shape[0])
    policy = DRPPO(
        obs_dim=obs_dim, n_actions=5,
        pinn_placement=args.pinn_placement,
        use_safety_filter=args.use_safety_filter,
        device=device,
    )
    policy.load(args.checkpoint)

    eval_modes = []
    if args.stochastic:
        eval_modes = [("deterministic", True), ("stochastic", False)]
    else:
        eval_modes = [("deterministic" if args.deterministic else "stochastic", args.deterministic)]

    all_seed_results = {}
    for mode_name, det in eval_modes:
        print(f"\n--- Eval mode: {mode_name} ---")
        for seed in args.seeds:
            print(f"Evaluating seed={seed} [{mode_name}]...")
            result = eval_single_seed(env, policy, args.episodes, det, seed)
            all_seed_results[(seed, mode_name)] = result

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "eval_mode", "episode", "return", "length",
            "collision_steps", "collision_episode",
            "mean_ttc", "min_ttc", "pothole_hits",
        ])
        for (seed, mode_name), result in all_seed_results.items():
            for i in range(len(result["returns"])):
                writer.writerow([
                    seed, mode_name, i, result["returns"][i], result["lengths"][i],
                    result["collision_counts"][i], result["collision_episodes"][i],
                    result["mean_ttc"][i], result["min_ttc"][i], result["pothole_hits"][i],
                ])

    all_returns = [r for res in all_seed_results.values() for r in res["returns"]]
    all_coll_eps = [c for res in all_seed_results.values() for c in res["collision_episodes"]]
    all_mean_ttc = [t for res in all_seed_results.values() for t in res["mean_ttc"]]
    all_min_ttc = [t for res in all_seed_results.values() for t in res["min_ttc"]]

    n_total = len(all_returns)
    print(f"\n=== Eval Summary ({n_total} total episodes) ===")
    print(f"Mean return: {np.mean(all_returns):.2f} +/- {np.std(all_returns):.2f}")
    if n_total > 1:
        print(f"95% CI: [{np.mean(all_returns) - 1.96*np.std(all_returns)/np.sqrt(n_total):.2f}, "
              f"{np.mean(all_returns) + 1.96*np.std(all_returns)/np.sqrt(n_total):.2f}]")
    n_coll = sum(all_coll_eps)
    print(f"Collision episodes: {n_coll}/{n_total} ({100*n_coll/max(n_total,1):.1f}%)")
    print(f"Mean TTC: {np.nanmean(all_mean_ttc):.2f}s")
    print(f"Mean min TTC: {np.nanmean(all_min_ttc):.2f}s")
    print(f"Metrics saved to {args.out}")

    summary_path = args.out.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_total_episodes": n_total,
            "episodes_per_seed": args.episodes,
            "mean_return": float(np.mean(all_returns)),
            "std_return": float(np.std(all_returns)),
            "collision_rate": float(np.mean(all_coll_eps)),
            "mean_ttc": float(np.nanmean(all_mean_ttc)),
            "mean_min_ttc": float(np.nanmean(all_min_ttc)),
        }, f, indent=2)
    print(f"Summary saved to {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
