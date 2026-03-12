"""Visualize ablation checkpoints: run SUMO with each model on each scenario.

Run all scenario×variant combinations, or select specific ones.
Use --gui to watch with sumo-gui (run one at a time for viewing).
"""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

SCENARIOS = ["1a", "1b", "1c", "1d", "2", "3", "4"]
VARIANTS = ["nopinn", "pinn", "pinn_ego"]


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
            "action": action,
            "reward": reward,
            "ttc_min": ttc_min,
            "collision": 1 if collision else 0,
        }
        log.append(row)
        if done:
            break
    return log


def main():
    parser = argparse.ArgumentParser(description="Visualize ablation models on all scenarios")
    parser.add_argument("--ablation_dir", default="results/ablation")
    parser.add_argument("--out_dir", default="results/ablation_viz")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per scenario×variant")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui (run one at a time)")
    parser.add_argument("--scenario", default=None,
                        help=f"Single scenario ({SCENARIOS}); if not set, run all")
    parser.add_argument("--variant", default=None,
                        help=f"Single variant ({VARIANTS}); if not set, run all")
    parser.add_argument("--use_intent", action="store_true")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS
    variants = [args.variant] if args.variant else VARIANTS

    from env.sumo_env import SumoEnv
    from models.drppo import DRPPO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    total = len(scenarios) * len(variants)
    count = 0
    for scenario in scenarios:
        for variant in variants:
            ckpt = os.path.join(args.ablation_dir, f"ablation_{scenario}_{variant}.pt")
            if not os.path.isfile(ckpt):
                print(f"Skip {scenario}/{variant}: checkpoint not found")
                continue
            count += 1
            viz_dir = os.path.join(args.out_dir, f"{scenario}_{variant}")
            os.makedirs(viz_dir, exist_ok=True)

            env_for_dim = SumoEnv(scenario_name=scenario, use_intent=args.use_intent)
            obs_dim = int(env_for_dim.observation_space.shape[0])
            del env_for_dim

            policy = DRPPO(obs_dim=obs_dim, n_actions=5, use_pinn=False, device=device)
            policy.load(ckpt)

            def policy_fn(obs):
                return policy.get_action(obs, deterministic=True)[0]

            print(f"[{count}/{total}] {scenario} {variant} ...")
            env = SumoEnv(use_gui=args.gui, scenario_name=scenario, use_intent=args.use_intent)
            try:
                for ep in range(args.episodes):
                    log = run_episode(env, policy_fn)
                    if not log:
                        continue
                    csv_path = os.path.join(viz_dir, f"ep{ep}.csv")
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=log[0].keys())
                        writer.writeheader()
                        writer.writerows(log)
                    ret = sum(r["reward"] for r in log)
                    coll = sum(r["collision"] for r in log)
                    print(f"  ep{ep}: return={ret:.1f} coll={coll} -> {csv_path}")
            finally:
                env.close()

    print(f"\nDone. Trajectories in {args.out_dir}/")
    if args.gui:
        print("(With --gui you watched the simulation. Use without --gui for batch headless runs.)")


if __name__ == "__main__":
    main()
