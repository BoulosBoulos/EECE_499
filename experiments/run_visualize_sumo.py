"""Visualize SUMO T-intersection: run episodes with GUI or headless, log & plot."""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

from env.sumo_env import SumoEnv, ACTION_NAMES


def run_episode(env, policy_fn, max_steps: int = 500):
    obs, _ = env.reset()
    log = []
    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        try:
            ego = env._get_ego()
            agents = env._get_agents()
        except Exception:
            ego = {"p": np.array([0.0, 0.0]), "v": 0.0, "a": 0.0}
            agents = []
        ttc_min = info.get("ttc_min", float("nan"))
        collision = any(np.linalg.norm(np.array(ag["p"]) - ego["p"]) < 2.0 for ag in agents)

        row = {
            "step": step,
            "ego_x": ego["p"][0],
            "ego_y": ego["p"][1],
            "ego_v": ego["v"],
            "ego_a": ego.get("a", 0),
            "action": action,
            "action_name": ACTION_NAMES[action],
            "reward": reward,
            "ttc_min": ttc_min,
            "collision": 1 if collision else 0,
            "n_agents": len(agents),
        }
        for i, ag in enumerate(agents):
            row[f"ag{i}_x"] = ag["p"][0]
            row[f"ag{i}_y"] = ag["p"][1]
            row[f"ag{i}_v"] = ag["v"]
        log.append(row)
        if done:
            break
    return log


def main():
    parser = argparse.ArgumentParser(description="Visualize SUMO T-intersection")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui to see simulation")
    parser.add_argument("--policy", choices=["random", "go", "checkpoint"], default="go")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario_dir", default=None, help="Override scenario directory (optional)")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"],
                        help="SUMO scenario (1a-1d, 2, 3, 4)")
    parser.add_argument("--use_intent", action="store_true", help="Use intent LSTM (must match checkpoint)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    env_for_dim = SumoEnv(scenario_name=args.scenario, use_intent=args.use_intent)
    obs_dim = int(env_for_dim.observation_space.shape[0])
    if args.policy == "checkpoint" and args.checkpoint:
        from models.drppo import DRPPO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy_net = DRPPO(obs_dim=obs_dim, n_actions=5, use_pinn=False, device=device)
        policy_net.load(args.checkpoint)

        def policy_fn(obs):
            return policy_net.get_action(obs, deterministic=True)[0]
    elif args.policy == "go":
        def policy_fn(obs):
            return 3
    else:
        def policy_fn(obs):
            return np.random.randint(0, 5)

    env = SumoEnv(use_gui=args.gui, scenario_dir=args.scenario_dir, scenario_name=args.scenario, use_intent=args.use_intent)
    try:
        for ep in range(args.episodes):
            log = run_episode(env, policy_fn)
            if not log:
                continue
            csv_path = os.path.join(args.out_dir, f"sumo_trajectory_ep{ep}.csv")
            all_keys = sorted(set().union(*(set(r.keys()) for r in log)))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(log)
            ret = sum(r["reward"] for r in log)
            print(f"Ep{ep}: {len(log)} steps, return={ret:.1f}, saved {csv_path}")
    finally:
        env.close()
    print("Done. Use --gui to watch episodes with sumo-gui.")


if __name__ == "__main__":
    main()
