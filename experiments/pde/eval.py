"""Evaluate PDE-family and DRPPO baseline trained models."""

from __future__ import annotations

import argparse
import os
import csv
import json
import numpy as np

from env.sumo_env import SumoEnv, ACTION_NAMES


def eval_model(env, policy, n_episodes: int, deterministic: bool, seed: int,
               save_failures: bool = False, max_failures: int = 10,
               fail_dir: str | None = None, fail_prefix: str = "") -> dict:
    returns, coll_eps, success_eps, pothole_hits, ttc_means, ttc_mins = [], [], [], [], [], []
    failure_count = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy.reset_hidden()
        r_tot, coll, pot = 0, 0, 0
        ttc_list = []
        trajectory = []
        ep_success = False

        for step_i in range(500):
            a, _, _, _ = policy.get_action(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            r_tot += r

            if save_failures:
                ego = info.get("raw_obs", {}).get("ego", {})
                p = ego.get("p", np.zeros(2))
                trajectory.append({
                    "step": step_i,
                    "action": a,
                    "action_name": ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else "?",
                    "reward": r,
                    "ego_x": float(p[0]) if hasattr(p, '__len__') else 0,
                    "ego_y": float(p[1]) if hasattr(p, '__len__') else 0,
                    "ego_v": ego.get("v", 0),
                    "ttc_min": info.get("ttc_min", 10.0),
                    "collision": 1 if info.get("collision", False) else 0,
                    "d_cz": float(info.get("built", {}).get("s_geom", np.zeros(12))[1]),
                })

            if info.get("collision", False):
                coll += 1
            if info.get("in_pothole", False):
                pot += 1
            ttc_list.append(info.get("ttc_min", 10.0))
            if term or trunc:
                ep_success = info.get("success", False)
                break

        # Save failure trajectory
        if save_failures and coll > 0 and failure_count < max_failures and fail_dir:
            os.makedirs(fail_dir, exist_ok=True)
            fail_path = os.path.join(fail_dir, f"{fail_prefix}ep{ep}.csv")
            if trajectory:
                with open(fail_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(trajectory[0].keys()))
                    writer.writeheader()
                    writer.writerows(trajectory)
                failure_count += 1

        returns.append(r_tot)
        coll_eps.append(1 if coll > 0 else 0)
        success_eps.append(1 if ep_success else 0)
        pothole_hits.append(pot)
        ttc_arr = np.array(ttc_list)
        ttc_means.append(float(np.mean(ttc_arr)))
        ttc_mins.append(float(np.min(ttc_arr)))

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "collision_rate": float(np.mean(coll_eps)),
        "success_rate": float(np.mean(success_eps)),
        "pothole_hits_mean": float(np.mean(pothole_hits)),
        "mean_ttc": float(np.mean(ttc_means)),
        "min_ttc": float(np.mean(ttc_mins)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", required=True,
                        choices=["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out_dir", default="results/pde")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4", "2_dense", "3_dense", "4_dense"])
    parser.add_argument("--ego_maneuver", default="stem_right",
                        choices=["stem_right", "stem_left", "right_left",
                                 "right_stem", "left_right", "left_stem"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no_buildings", action="store_true",
                        help="Disable static occlusion buildings (full visibility)")
    parser.add_argument("--style_filter", default=None, choices=["nominal", "adversarial"],
                        help="Filter agent behavioral styles for robustness ablation")
    parser.add_argument("--state_ablation", default=None, choices=["no_visibility"],
                        help="State ablation: remove specific feature groups")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--save_failures", action="store_true",
                        help="Save trajectory CSVs for episodes that end in collision")
    parser.add_argument("--max_failures", type=int, default=10)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        torch = None
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    env = SumoEnv(use_gui=args.gui, scenario_name=args.scenario,
                  ego_maneuver=args.ego_maneuver, use_intent=args.use_intent,
                  buildings=not args.no_buildings, style_filter=args.style_filter,
                  state_ablation=args.state_ablation)
    obs_dim = int(env.observation_space.shape[0])

    if args.method == "hjb_aux":
        from models.pde.hjb_aux_agent import HJBAuxAgent
        policy = HJBAuxAgent(obs_dim=obs_dim, device=device)
    elif args.method == "soft_hjb_aux":
        from models.pde.soft_hjb_aux_agent import SoftHJBAuxAgent
        policy = SoftHJBAuxAgent(obs_dim=obs_dim, device=device)
    elif args.method == "eikonal_aux":
        from models.pde.eikonal_aux_agent import EikonalAuxAgent
        policy = EikonalAuxAgent(obs_dim=obs_dim, device=device)
    elif args.method == "cbf_aux":
        from models.pde.cbf_aux_agent import CBFAuxAgent
        policy = CBFAuxAgent(obs_dim=obs_dim, device=device)
    elif args.method == "drppo":
        from models.drppo import DRPPO
        policy = DRPPO(obs_dim=obs_dim, device=device)
    policy.load(args.checkpoint)

    os.makedirs(args.out_dir, exist_ok=True)
    fail_dir = os.path.join(args.out_dir, "failures") if args.save_failures else None
    fail_prefix = f"fail_{args.method}_{args.scenario}_{args.ego_maneuver}_"

    all_results = {}
    for seed in args.seeds:
        for mode, det in [("deterministic", True), ("stochastic", False)]:
            print(f"Eval seed={seed} [{mode}]...")
            m = eval_model(env, policy, args.episodes, det, seed,
                           save_failures=args.save_failures,
                           max_failures=args.max_failures,
                           fail_dir=fail_dir, fail_prefix=f"{fail_prefix}s{seed}_{mode}_")
            all_results[(seed, mode)] = m
            print(f"  return={m['mean_return']:.2f} coll={m['collision_rate']:.3f} "
                  f"success={m['success_rate']:.3f} ttc={m['mean_ttc']:.2f}")

    csv_path = os.path.join(args.out_dir, f"eval_{args.method}_{args.scenario}_{args.ego_maneuver}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "eval_mode", "mean_return", "std_return", "collision_rate",
                     "success_rate", "pothole_hits_mean", "mean_ttc", "min_ttc"])
        for (seed, mode), m in all_results.items():
            w.writerow([seed, mode, m["mean_return"], m["std_return"], m["collision_rate"],
                        m["success_rate"], m["pothole_hits_mean"], m["mean_ttc"], m["min_ttc"]])
    print(f"Saved {csv_path}")
    env.close()


if __name__ == "__main__":
    main()
