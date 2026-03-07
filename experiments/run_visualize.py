"""Visualize synthetic T-intersection scenario: run episodes, log & plot ego + agents."""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

from env.t_intersection_env import TIntersectionEnv, ACTION_NAMES


def run_episode(env, policy_fn, max_steps: int = 500):
    """Run one episode, return full trajectory log."""
    obs, _ = env.reset()
    log = []
    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        ego = env._ego
        agents = env._agents
        ttc_min = info.get("ttc_min", float("nan"))
        collision = any(np.linalg.norm(ag["p"] - ego["p"]) < 2.0 for ag in agents)

        row = {
            "step": step,
            "ego_x": ego["p"][0],
            "ego_y": ego["p"][1],
            "ego_v": ego["v"],
            "ego_a": ego["a"],
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


def random_policy(obs):
    return np.random.randint(0, 5)


def main():
    parser = argparse.ArgumentParser(description="Visualize T-intersection scenario")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=["random", "go", "checkpoint"], default="random",
                        help="random, always-GO, or load policy from --checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Path to model for policy")
    parser.add_argument("--no_plot", action="store_true", help="Skip trajectory plots")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.policy == "checkpoint" and args.checkpoint:
        from models.drppo import DRPPO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy_net = DRPPO(obs_dim=134, n_actions=5, use_pinn=False, device=device)
        policy_net.load(args.checkpoint)

        def policy_fn(obs):
            a, _, _, _ = policy_net.get_action(obs, deterministic=True)
            return a
    elif args.policy == "go":
        def policy_fn(obs):
            return 3  # GO
    else:
        policy_fn = random_policy

    env = TIntersectionEnv()

    all_episode_metrics = []

    for ep in range(args.episodes):
        log = run_episode(env, policy_fn)
        if not log:
            continue

        # Save per-episode CSV
        csv_path = os.path.join(args.out_dir, f"trajectory_ep{ep}.csv")
        fieldnames = list(log[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log)
        print(f"Saved {csv_path} ({len(log)} steps)")

        # Compute episode metrics
        returns = sum(r["reward"] for r in log)
        lengths = len(log)
        collisions = sum(r["collision"] for r in log)
        ttc_vals = [r["ttc_min"] for r in log if not np.isnan(r["ttc_min"]) and r["ttc_min"] < 1e5]
        mean_ttc = np.mean(ttc_vals) if ttc_vals else float("nan")
        min_ttc = np.min(ttc_vals) if ttc_vals else float("nan")

        all_episode_metrics.append({
            "episode": ep,
            "return": returns,
            "length": lengths,
            "collision_steps": collisions,
            "collision_episode": 1 if collisions > 0 else 0,
            "mean_ttc": mean_ttc,
            "min_ttc": min_ttc,
        })
        print(f"  Ep{ep}: return={returns:.1f}, len={lengths}, coll_steps={collisions}, "
              f"mean_ttc={mean_ttc:.2f}, min_ttc={min_ttc:.2f}")

        # Plot trajectory
        if not args.no_plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ego_x = [r["ego_x"] for r in log]
                ego_y = [r["ego_y"] for r in log]
                ax.plot(ego_x, ego_y, "b-o", markersize=3, label="ego", alpha=0.8)
                n_ag = log[0].get("n_agents", 0)
                for i in range(n_ag):
                    if f"ag{i}_x" in log[0]:
                        ax_x = [r[f"ag{i}_x"] for r in log]
                        ax_y = [r[f"ag{i}_y"] for r in log]
                        ax.plot(ax_x, ax_y, "r--", alpha=0.6, label=f"agent{i}")
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_title(f"Ep{ep}: return={returns:.1f}, coll_steps={collisions}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axis("equal")
                plot_path = os.path.join(args.out_dir, f"trajectory_ep{ep}.png")
                fig.savefig(plot_path, dpi=100)
                plt.close()
                print(f"  Saved plot {plot_path}")

                # Action timeline plot
                fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
                steps_arr = [r["step"] for r in log]
                actions_arr = [r["action"] for r in log]
                ax2.step(steps_arr, actions_arr, where="mid", color="steelblue", alpha=0.9)
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Action")
                ax2.set_yticks(range(5))
                ax2.set_yticklabels(ACTION_NAMES)
                ax2.set_title(f"Ep{ep}: Discrete actions over time")
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(-0.5, 4.5)
                plot_path_actions = os.path.join(args.out_dir, f"trajectory_ep{ep}_actions.png")
                fig2.savefig(plot_path_actions, dpi=100)
                plt.close()
                print(f"  Saved action plot {plot_path_actions}")
            except ImportError:
                print("  (matplotlib not installed; skip --no_plot to avoid plot)")

    # Summary metrics CSV
    summary_path = os.path.join(args.out_dir, "visualize_metrics.csv")
    if all_episode_metrics:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_episode_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_episode_metrics)
        print(f"\nSummary: {summary_path}")
        print(f"  Mean return: {np.mean([m['return'] for m in all_episode_metrics]):.2f}")
        print(f"  Mean length: {np.mean([m['length'] for m in all_episode_metrics]):.1f}")
        print(f"  Collision episodes: {sum(m['collision_episode'] for m in all_episode_metrics)}/{len(all_episode_metrics)}")


if __name__ == "__main__":
    main()
