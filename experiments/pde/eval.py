"""Evaluate PDE-family trained models."""

from __future__ import annotations

import argparse
import os
import csv
import json
import numpy as np

from env.sumo_env import SumoEnv


def eval_model(env, policy, n_episodes: int, deterministic: bool, seed: int) -> dict:
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=["hjb_aux", "soft_hjb_aux"], required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out_dir", default="results/pde")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--ego_maneuver", default="stem_right",
                        choices=["stem_right", "stem_left", "right_left",
                                 "right_stem", "left_right", "left_stem"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        torch = None
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    env = SumoEnv(use_gui=args.gui, scenario_name=args.scenario,
                  ego_maneuver=args.ego_maneuver, use_intent=args.use_intent)
    obs_dim = int(env.observation_space.shape[0])

    if args.method == "hjb_aux":
        from models.pde.hjb_aux_agent import HJBAuxAgent
        policy = HJBAuxAgent(obs_dim=obs_dim, device=device)
    else:
        from models.pde.soft_hjb_aux_agent import SoftHJBAuxAgent
        policy = SoftHJBAuxAgent(obs_dim=obs_dim, device=device)
    policy.load(args.checkpoint)

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}
    for seed in args.seeds:
        for mode, det in [("deterministic", True), ("stochastic", False)]:
            print(f"Eval seed={seed} [{mode}]...")
            m = eval_model(env, policy, args.episodes, det, seed)
            all_results[(seed, mode)] = m
            print(f"  return={m['mean_return']:.2f} coll={m['collision_rate']:.3f} ttc={m['mean_ttc']:.2f}")

    csv_path = os.path.join(args.out_dir, f"eval_{args.method}_{args.scenario}_{args.ego_maneuver}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "eval_mode", "mean_return", "std_return", "collision_rate",
                     "pothole_hits_mean", "mean_ttc", "min_ttc"])
        for (seed, mode), m in all_results.items():
            w.writerow([seed, mode, m["mean_return"], m["std_return"], m["collision_rate"],
                        m["pothole_hits_mean"], m["mean_ttc"], m["min_ttc"]])
    print(f"Saved {csv_path}")
    env.close()


if __name__ == "__main__":
    main()
