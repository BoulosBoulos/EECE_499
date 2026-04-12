"""Visualize PDE-family trained models in SUMO GUI with action overlays."""

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
        row = {
            "step": step,
            "ego_x": ego.get("p", [0, 0])[0] if isinstance(ego.get("p"), (list, np.ndarray)) else 0,
            "ego_y": ego.get("p", [0, 0])[1] if isinstance(ego.get("p"), (list, np.ndarray)) else 0,
            "ego_v": ego.get("v", 0),
            "ego_a": ego.get("a", 0),
            "action": action,
            "action_name": ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else "?",
            "reward": reward,
            "ttc_min": info.get("ttc_min", float("nan")),
            "collision": 1 if info.get("collision", False) else 0,
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
    parser = argparse.ArgumentParser(description="Visualize PDE-family model in SUMO")
    parser.add_argument("--checkpoint", required=True, help="Path to PDE-family checkpoint")
    parser.add_argument("--method", choices=["hjb_aux", "soft_hjb_aux"], required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out_dir", default="results/pde")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import torch
    except ImportError:
        torch = None
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    env_for_dim = SumoEnv(scenario_name=args.scenario, use_intent=args.use_intent)
    obs_dim = int(env_for_dim.observation_space.shape[0])

    if args.method == "hjb_aux":
        from models.pde.hjb_aux_agent import HJBAuxAgent
        policy = HJBAuxAgent(obs_dim=obs_dim, device=device)
    else:
        from models.pde.soft_hjb_aux_agent import SoftHJBAuxAgent
        policy = SoftHJBAuxAgent(obs_dim=obs_dim, device=device)
    policy.load(args.checkpoint)

    def policy_fn(obs):
        return policy.get_action(obs, deterministic=True)[0]

    env = SumoEnv(use_gui=args.gui, scenario_name=args.scenario, use_intent=args.use_intent)
    try:
        for ep in range(args.episodes):
            policy.reset_hidden()
            log = run_episode(env, policy_fn)
            if not log:
                continue
            csv_path = os.path.join(args.out_dir, f"pde_trajectory_{args.method}_ep{ep}.csv")
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
    print("Done.")


if __name__ == "__main__":
    main()
