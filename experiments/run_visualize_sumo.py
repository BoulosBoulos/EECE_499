"""Visualize SUMO T-intersection: run episodes with GUI or headless, log & plot."""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

from env.sumo_env import SumoEnv, ACTION_NAMES


def run_episode(env, policy_fn, max_steps: int = 500):
    obs, info = env.reset()
    log = []
    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        ego = info.get("raw_obs", {}).get("ego", {})
        agents = info.get("raw_obs", {}).get("agents", [])
        collision = info.get("collision", False)

        ego_p = ego.get("p", np.array([0.0, 0.0]))
        if isinstance(ego_p, np.ndarray):
            ego_x, ego_y = float(ego_p[0]), float(ego_p[1])
        elif isinstance(ego_p, (list, tuple)) and len(ego_p) >= 2:
            ego_x, ego_y = float(ego_p[0]), float(ego_p[1])
        else:
            ego_x, ego_y = 0.0, 0.0

        row = {
            "step": step,
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_v": ego.get("v", 0),
            "ego_a": ego.get("a", 0),
            "action": action,
            "action_name": ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else "?",
            "reward": reward,
            "ttc_min": info.get("ttc_min", float("nan")),
            "collision": 1 if collision else 0,
            "nearest_agent_dist": info.get("nearest_agent_dist", float("inf")),
            "n_agents": len(agents),
        }
        for i, ag in enumerate(agents[:5]):
            p = ag.get("p", [0, 0])
            row[f"ag{i}_x"] = p[0] if isinstance(p, (list, np.ndarray)) else 0
            row[f"ag{i}_y"] = p[1] if isinstance(p, (list, np.ndarray)) else 0
            row[f"ag{i}_v"] = ag.get("v", 0)
            row[f"ag{i}_type"] = ag.get("type", "?")
        log.append(row)
        if done:
            break
    return log


def main():
    parser = argparse.ArgumentParser(description="Visualize SUMO T-intersection")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui to see simulation")
    parser.add_argument("--policy", choices=["random", "go", "checkpoint"], default="checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained model checkpoint (required when --policy=checkpoint)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario_dir", default=None, help="Override scenario directory (optional)")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"],
                        help="SUMO scenario (1a-1d, 2, 3, 4)")
    parser.add_argument("--use_intent", action="store_true", help="Use intent LSTM (must match checkpoint)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.policy == "checkpoint" and not args.checkpoint:
        print("ERROR: --policy=checkpoint requires --checkpoint <path>.")
        print("Train a model first with: make train")
        print("Then: python3 experiments/run_visualize_sumo.py --gui --policy checkpoint --checkpoint results/model_pinn_1a.pt")
        print("Or use --policy go for a non-RL baseline (always accelerates).")
        return

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
            colls = sum(r["collision"] for r in log)
            print(f"Ep{ep}: {len(log)} steps, return={ret:.1f}, collisions={colls}, saved {csv_path}")
    finally:
        env.close()
    print("Done. Use --gui to watch episodes with sumo-gui.")


if __name__ == "__main__":
    main()
